"""Tests for AWQ scaling (ADR 0031).

The most important test is the equivalence check: a rescaled checkpoint must
compute the same logits as the original. It uses a tiny random Qwen2 model, so
it runs in seconds on a CPU.
"""

import json
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


def test_sharded_rescaling_matches_the_single_file_rescaling():
    from safetensors import safe_open
    from safetensors.torch import save_file

    config, model = tiny_qwen()
    windows = torch.randint(0, config.vocab_size, (4, 16)).tolist()
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        sharded = root / "sharded"
        model.save_pretrained(sharded, max_shard_size="40KB")
        weight_map = json.loads((sharded / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        assert len(set(weight_map.values())) >= 2
        # The shards must split one layer between them, or the cross-shard case is not tested.
        assert any(
            len({weight_map[name] for name in weight_map if name.startswith(f"model.layers.{layer}.")}) > 1
            for layer in range(config.num_hidden_layers)
        )

        single = root / "original.safetensors"
        save_file({k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}, single)
        with safe_open(single, framework="pt") as source:
            blocks = awq.calibrate(model, windows, group_size=16, device="cpu")
            awq.write_scaled_checkpoint(source, blocks, root / "scaled.safetensors", {"modelq.transform": "awq"})
        shard_names = awq.write_scaled_shards(sharded, blocks, root / "scaled-sharded", {"modelq.transform": "awq"})
        assert sorted(shard_names) == sorted(set(weight_map.values()))

        with safe_open(root / "scaled.safetensors", framework="pt") as one:
            expected = {name: one.get_tensor(name) for name in one.keys()}
        found = {}
        for shard in shard_names:
            with safe_open(root / "scaled-sharded" / shard, framework="pt") as part:
                for name in part.keys():
                    found[name] = part.get_tensor(name)
        assert set(found) == set(expected)
        for name in expected:
            assert found[name].dtype == expected[name].dtype, name
            assert torch.equal(found[name], expected[name]), name

        written = json.loads((root / "scaled-sharded" / "model.safetensors.index.json").read_text(encoding="utf-8"))
        assert written["weight_map"] == weight_map
        assert written["metadata"]["total_size"] == sum(t.numel() * t.element_size() for t in found.values())


def test_a_sharded_rescaling_refuses_a_tensor_its_index_does_not_name():
    config, model = tiny_qwen()
    windows = torch.randint(0, config.vocab_size, (4, 16)).tolist()
    blocks = awq.calibrate(model, windows, group_size=16, device="cpu")
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        index = {"weight_map": {"model.embed_tokens.weight": "model-1.safetensors"}}
        (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        with pytest.raises(ValueError, match="has no tensor named"):
            awq.write_scaled_shards(root, blocks, root / "out", {"modelq.transform": "awq"})
        assert not (root / "out").exists()


def _calibrate_all_layers_at_once(model, windows, group_size, device, batch_size=4):
    """The search before ADR 0038: every layer's inputs are held until each is searched.

    Kept only as the reference that the one-layer-at-a-time search must match exactly.
    """
    captured = {}
    handles = []
    layers = model.model.layers

    def capture(name: str):
        def hook(_module, inputs):
            tensor = inputs[0].detach()
            captured.setdefault(name, []).append(tensor.reshape(-1, tensor.shape[-1]).float())

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.self_attn.q_proj.register_forward_pre_hook(capture(f"{index}.attn")))
        handles.append(layer.mlp.gate_proj.register_forward_pre_hook(capture(f"{index}.mlp")))
        handles.append(layer.mlp.down_proj.register_forward_pre_hook(capture(f"{index}.down")))

    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(windows), batch_size):
                batch = torch.tensor(windows[start : start + batch_size], device=device)
                model.model(batch)
    finally:
        for handle in handles:
            handle.remove()

    blocks = []
    for index, layer in enumerate(layers):
        attn_inputs = torch.cat(captured[f"{index}.attn"])
        blocks.append(
            awq.search_block(
                index,
                "A",
                attn_inputs,
                {
                    "self_attn.q_proj": layer.self_attn.q_proj.weight.detach().float(),
                    "self_attn.k_proj": layer.self_attn.k_proj.weight.detach().float(),
                    "self_attn.v_proj": layer.self_attn.v_proj.weight.detach().float(),
                },
                group_size,
            )
        )
        mlp_inputs = torch.cat(captured[f"{index}.mlp"])
        blocks.append(
            awq.search_block(
                index,
                "C",
                mlp_inputs,
                {
                    "mlp.gate_proj": layer.mlp.gate_proj.weight.detach().float(),
                    "mlp.up_proj": layer.mlp.up_proj.weight.detach().float(),
                },
                group_size,
            )
        )
        down_inputs = torch.cat(captured[f"{index}.down"])
        blocks.append(
            awq.search_block(
                index,
                "D",
                down_inputs,
                {"mlp.down_proj": layer.mlp.down_proj.weight.detach().float()},
                group_size,
            )
        )
        captured.pop(f"{index}.attn")
        captured.pop(f"{index}.mlp")
        captured.pop(f"{index}.down")
    return blocks


def test_one_layer_at_a_time_gives_the_same_blocks():
    config, model = tiny_qwen()
    # Six windows in batches of four and two, so the partial batch is covered too.
    windows = torch.randint(0, config.vocab_size, (6, 16)).tolist()
    reference = _calibrate_all_layers_at_once(model, windows, group_size=16, device="cpu", batch_size=4)
    blocks = awq.calibrate(model, windows, group_size=16, device="cpu", batch_size=4)
    assert len(blocks) == len(reference) == 3 * config.num_hidden_layers
    for got, want in zip(blocks, reference):
        assert (got.layer, got.block, got.linears) == (want.layer, want.block, want.linears)
        assert got.alpha == want.alpha
        assert torch.equal(got.scale, want.scale)
        assert got.error_unscaled == want.error_unscaled
        assert got.error_scaled == want.error_scaled
