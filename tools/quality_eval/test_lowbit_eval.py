"""Tests for reading ModelQ group-wise low-bit files in the evaluator (ADR 0030)."""

import json
import pathlib
import sys
import tempfile

import numpy as np
import pytest
import torch

import quality_eval as qe

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "calibration"))
import awq  # noqa: E402


def test_unpacking_matches_the_rust_two_bit_golden_vector():
    # Rust: packed 0x74 holds codes 0, 1, 3, 1 (LSB-first).
    assert qe.unpack_codes(np.array([0x74], dtype=np.uint8), 4, 2).tolist() == [0, 1, 3, 1]


def test_decoding_matches_the_rust_golden_vectors():
    assert qe.decode_lowbit(np.array([0x74], np.uint8), [1.0], 4, 2, 4, "symmetric").tolist() == [0.0, 1.0, -1.0, 1.0]
    three = np.array([0xAB, 0x0C, 0x00], np.uint8)
    assert qe.decode_lowbit(three, [1.0], 8, 3, 8, "symmetric").tolist() == [3.0, -3.0, 2.0, -2.0, 0.0, 0.0, 0.0, 0.0]
    assert qe.decode_lowbit(np.array([0x05], np.uint8), [2.0], 4, 1, 4, "sign").tolist() == [2.0, -2.0, 2.0, -2.0]


def test_the_reserved_code_is_refused():
    # A three-bit code 0b100 is the reserved minimum -4.
    with pytest.raises(ValueError, match="reserved"):
        qe.decode_lowbit(np.array([0b100], np.uint8), [1.0], 2, 3, 2, "symmetric")


def test_a_partial_last_group_uses_its_own_scale():
    # Five 4-bit values in groups of two: the last value uses the third scale.
    packed = np.array([0x21, 0x43, 0x05], np.uint8)
    values = qe.decode_lowbit(packed, [1.0, 1.0, 2.0], 5, 4, 2, "symmetric")
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0, 10.0]


def test_an_int8_file_is_not_read_as_low_bit():
    assert qe.read_lowbit_manifest({"modelq.format": "modelq-native", "modelq.quantization": "int8"}) is None


def test_a_wrong_packed_length_is_an_error():
    with pytest.raises(ValueError):
        qe.unpack_codes(np.array([0x00, 0x00], np.uint8), 4, 2)


def encode_symmetric(matrix, group_size, bits):
    """Encodes a matrix with the rules of ADR 0030, for building a test file."""
    qmax = 2 ** (bits - 1) - 1
    flat = matrix.reshape(-1).astype(np.float32)
    codes, scales = [], []
    for start in range(0, flat.size, group_size):
        chunk = flat[start : start + group_size]
        largest = float(np.abs(chunk).max())
        scale = np.float32(largest / qmax) if largest > 0 else np.float32(1.0)
        scales.append(scale)
        scaled = chunk / scale
        rounded = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
        clamped = np.clip(rounded, -qmax, qmax).astype(np.int64)
        codes.extend((clamped & ((1 << bits) - 1)).tolist())
    stream = np.zeros(len(codes) * bits, dtype=np.uint8)
    for index, code in enumerate(codes):
        for bit in range(bits):
            stream[index * bits + bit] = (code >> bit) & 1
    return np.packbits(stream, bitorder="little"), np.asarray(scales, dtype=np.float32)


def test_a_written_low_bit_file_decodes_to_the_rescaling_quantizer_values():
    from safetensors.torch import save_file

    torch.manual_seed(3)
    matrix = torch.randn(32, 32)
    packed, scales = encode_symmetric(matrix.numpy(), group_size=16, bits=4)
    name = "model.layers.0.self_attn.q_proj.weight"
    metadata = {
        "modelq.format": "modelq-native",
        "modelq.format_version": "2",
        "modelq.quantization": "int4",
        "modelq.scheme": "symmetric-group-wise",
        "modelq.bits": "4",
        "modelq.group_size": "16",
        "modelq.manifest": json.dumps(
            {
                "schema": qe.LOWBIT_SCHEMA,
                "tensors": {
                    name: {
                        "action": "quantized",
                        "original_dtype": "F32",
                        "original_shape": [32, 32],
                        "qdata_name": name + ".qdata",
                        "qdata_dtype": "U8",
                        "qdata_shape": [len(packed)],
                        "scale_name": name + ".scale",
                        "scale_dtype": "F32",
                        "scale_shape": [len(scales)],
                        "elements": 32 * 32,
                        "bits": 4,
                        "group_size": 16,
                    }
                },
            }
        ),
    }
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
    )
    model = Qwen2ForCausalLM(config).float()
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "low.safetensors"
        save_file(
            {name + ".qdata": torch.from_numpy(packed.copy()), name + ".scale": torch.from_numpy(scales.copy())},
            path,
            metadata=metadata,
        )
        report = qe.substitute_lowbit_weights(model, path, torch)
    assert report.replaced == [name]
    decoded = dict(model.named_parameters())[name].detach()
    expected = awq.pseudo_quantize(matrix, group_size=16, bits=4)
    assert torch.allclose(decoded, expected, atol=1e-6)
