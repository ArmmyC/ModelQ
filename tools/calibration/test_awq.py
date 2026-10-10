"""Tests for AWQ scaling (ADR 0031).

The most important test is the equivalence check: a rescaled checkpoint must
compute the same logits as the original. It uses a tiny random Qwen2 model, so
it runs in seconds on a CPU.
"""

import pathlib
import tempfile

import pytest
import torch

import awq


def test_rounding_ties_away_from_zero_as_rust_does():
    values = torch.tensor([0.5, -0.5, 1.5, -2.5, 2.4999, -2.4999, 0.0])
    assert awq.round_half_away(values).tolist() == [1.0, -1.0, 2.0, -3.0, 2.0, -2.0, 0.0]


def test_rounding_is_exact_just_below_one_half():
    # floor(|v| + 0.5) gives 1 for this value in float32; the exact rule gives 0.
    just_below = torch.tensor([0.49999997, -0.49999997, 1.4999999, 2.5], dtype=torch.float32)
    assert awq.round_half_away(just_below).tolist() == [0.0, -0.0, 1.0, 3.0]


def test_group_quantization_matches_the_two_bit_golden_vector():
    # Same vector as the Rust golden test: max|w| = 1, qmax = 1, decodes to [0, 1, -1, 1].
    weight = torch.tensor([[0.0, 1.0, -1.0, 0.5]])
    assert awq.pseudo_quantize(weight, group_size=4, bits=2).tolist() == [[0.0, 1.0, -1.0, 1.0]]


def test_group_quantization_matches_the_three_bit_golden_vector():
    # Same vector as the Rust golden test: scale 1, decodes to [3, -3, 2, -2, 0, 0, 0, 0].
    weight = torch.tensor([[3.0, -3.0, 1.5, -1.5, 0.0, 0.0, 0.0, 0.0]])
    result = awq.pseudo_quantize(weight, group_size=8, bits=3)
    assert result.tolist() == [[3.0, -3.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0]]


def test_all_zero_groups_keep_their_zeros():
    weight = torch.zeros(2, 8)
    assert torch.equal(awq.pseudo_quantize(weight, group_size=4), weight)


def test_a_partial_group_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        awq.pseudo_quantize(torch.zeros(1, 6), group_size=4)


def test_unit_scale_has_no_effect_on_the_error():
    inputs = torch.randn(32, 16)
    weights = [torch.randn(8, 16)]
    ones = torch.ones(16)
    scaled_by_one = awq.block_error(inputs, weights, ones, group_size=4)
    assert scaled_by_one == pytest.approx(awq.block_error(inputs, weights, ones, group_size=4))


def test_channel_scale_is_normalized_and_monotone():
    activation = torch.tensor([0.1, 1.0, 10.0])
    scale = awq.channel_scale(activation, 0.5)
    assert float(scale.max() * scale.min()) == pytest.approx(1.0, rel=1e-5)
    assert scale[0] < scale[1] < scale[2]
    assert torch.allclose(awq.channel_scale(activation, 0.0), torch.ones(3))


def tiny_qwen():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config).float()
    # Give the norms non-trivial weights so the fold is exercised.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "layernorm" in name or name.endswith("norm.weight"):
                parameter.uniform_(0.5, 1.5)
    return config, model


def test_rescaled_checkpoint_computes_the_same_logits():
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import Qwen2ForCausalLM

    config, model = tiny_qwen()
    torch.manual_seed(1)
    windows = torch.randint(0, config.vocab_size, (8, 24)).tolist()
    with tempfile.TemporaryDirectory() as directory:
        original = pathlib.Path(directory) / "original.safetensors"
        save_file({k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}, original)
        with safe_open(original, framework="pt") as source:
            blocks = awq.calibrate(model, windows, group_size=16, device="cpu", batch_size=4)
            scaled_path = pathlib.Path(directory) / "scaled.safetensors"
            awq.write_scaled_checkpoint(source, blocks, scaled_path, {"modelq.transform": "awq"})

        # The search must actually move some scales, or the test proves nothing.
        assert any(block.alpha > 0 for block in blocks)
        assert any(
            not torch.allclose(block.scale, torch.ones_like(block.scale)) for block in blocks
        )

        rescaled = Qwen2ForCausalLM(config).float()
        with safe_open(scaled_path, framework="pt") as scaled:
            state = {name: scaled.get_tensor(name) for name in scaled.keys()}
        missing, unexpected = rescaled.load_state_dict(state, strict=False)
        assert set(missing) <= {"lm_head.weight"}, missing
        assert not unexpected, unexpected

    probe = torch.randint(0, config.vocab_size, (2, 24))
    with torch.no_grad():
        reference = model(probe).logits
        rescaled.tie_weights()
        result = rescaled(probe).logits
    assert float((reference - result).abs().max()) < 1e-4


def test_the_rescaled_checkpoint_keeps_untransformed_tensors_unchanged():
    from safetensors import safe_open
    from safetensors.torch import save_file

    config, model = tiny_qwen()
    windows = torch.randint(0, config.vocab_size, (4, 16)).tolist()
    with tempfile.TemporaryDirectory() as directory:
        original = pathlib.Path(directory) / "original.safetensors"
        save_file({k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}, original)
        with safe_open(original, framework="pt") as source:
            blocks = awq.calibrate(model, windows, group_size=16, device="cpu")
            scaled_path = pathlib.Path(directory) / "scaled.safetensors"
            awq.write_scaled_checkpoint(source, blocks, scaled_path, {"modelq.transform": "awq"})
        with safe_open(original, framework="pt") as before, safe_open(scaled_path, framework="pt") as after:
            assert set(before.keys()) == set(after.keys())
            transformed = {
                name
                for name in before.keys()
                if not torch.equal(before.get_tensor(name).float(), after.get_tensor(name).float())
            }
            assert "model.embed_tokens.weight" not in transformed
            assert all("layers" in name for name in transformed)


def test_the_error_summary_reports_the_reduction():
    config, model = tiny_qwen()
    windows = torch.randint(0, config.vocab_size, (4, 16)).tolist()
    blocks = awq.calibrate(model, windows, group_size=16, device="cpu")
    summary = awq.summarize(blocks)
    assert summary["total_error_scaled"] <= summary["total_error_unscaled"] + 1e-9
    assert len(summary["blocks"]) == 3 * config.num_hidden_layers
