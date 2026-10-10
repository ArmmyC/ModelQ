"""Tests for the readers of every ModelQ container (ADR 0035). Nothing is downloaded.

The golden blocks are the reference blocks of the Rust writers' tests (ADR 0032, ADR 0033), so the
decoders are checked against the same bytes the writers are checked against.
"""

import json
import struct

import numpy as np
import pytest

import modelq_containers as mc
import validate_multi as vm

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402  (after the torch check)


# --- helpers ---------------------------------------------------------------------------------


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path, kvs, tensors, alignment=32):
    """Writes a GGUF version 3 file. `tensors` are (name, GGML dims, ggml type, raw bytes)."""
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for key, value in kvs.items():
        header += _string(key)
        if isinstance(value, str):
            header += struct.pack("<I", 8) + _string(value)
        else:
            header += struct.pack("<I", 4) + struct.pack("<I", value)
    data = bytearray()
    offsets = []
    for _, _, _, raw in tensors:
        data += b"\0" * ((-len(data)) % alignment)
        offsets.append(len(data))
        data += raw
    for (name, dims, ggml_type, _), offset in zip(tensors, offsets):
        header += _string(name) + struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", dimension) for dimension in dims)
        header += struct.pack("<IQ", ggml_type, offset)
    header += b"\0" * ((-len(header)) % alignment)
    path.write_bytes(bytes(header) + bytes(data))
    return path


def f16_bytes(value: float) -> bytes:
    return np.float16(value).tobytes()


# The Q4_0 reference block of the Rust writer's tests: x[i] = i - 16, scale 2, codes as written.
Q4_0_GOLDEN = bytes(
    [0x00, 0x40, 0x80, 0x91, 0x91, 0xA2, 0xA2, 0xB3, 0xB3, 0xC4, 0xC4, 0xD5, 0xD5, 0xE6, 0xE6, 0xF7, 0xF7, 0xF8]
)
Q4_0_GOLDEN_VALUES = np.array(
    [-16, -14, -14, -12, -12, -10, -10, -8, -8, -6, -6, -4, -4, -2, -2, 0,
     0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14, 14],
    dtype=np.float32,
)


def _tiny_model(shapes):
    """A module whose parameter names are the given dotted names, with zero values."""
    root = torch.nn.Module()
    for name, shape in shapes.items():
        *path, leaf = name.split(".")
        owner = root
        for part in path:
            if not hasattr(owner, part):
                setattr(owner, part, torch.nn.Module())
            owner = getattr(owner, part)
        owner.register_parameter(leaf, torch.nn.Parameter(torch.zeros(shape)))
    return root


# --- detection -------------------------------------------------------------------------------


def test_each_container_format_is_recognized(tmp_path):
    gguf_path = write_gguf(tmp_path / "a.gguf", {"general.architecture": "qwen2"}, [])
    assert mc.detect_container(gguf_path) == "gguf"

    def safetensors_with(name, metadata):
        path = tmp_path / name
        save_file({"x": torch.zeros(1)}, str(path), metadata=metadata)
        return path

    assert mc.detect_container(safetensors_with("te.safetensors", {"modelq.format": vm.FORMAT})) == "nvfp4-te"
    lowbit = {"modelq.format": "modelq-native", "modelq.quantization": "int4"}
    assert mc.detect_container(safetensors_with("int4.safetensors", lowbit)) == "lowbit"
    int8 = {"modelq.format": "modelq-native", "modelq.quantization": "int8"}
    assert mc.detect_container(safetensors_with("int8.safetensors", int8)) == "int8"
    native = {"modelq.format": "modelq-native", "modelq.quantization": "nvfp4"}
    assert mc.detect_container(safetensors_with("nvfp4.safetensors", native)) == "nvfp4"


def test_an_unknown_file_is_refused(tmp_path):
    plain = tmp_path / "plain.safetensors"
    save_file({"x": torch.zeros(1)}, str(plain))
    with pytest.raises(ValueError, match="not a ModelQ container"):
        mc.detect_container(plain)
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"not a model at all")
    with pytest.raises(ValueError, match="neither a GGUF file nor a SafeTensors"):
        mc.detect_container(junk)


# --- GGUF ------------------------------------------------------------------------------------


def test_gguf_blocks_decode_as_the_reference_blocks_do(tmp_path):
    f32_values = np.arange(4, dtype="<f4")
    q8_raw = f16_bytes(0.5) + np.arange(-16, 16, dtype=np.int8).tobytes()
    path = write_gguf(
        tmp_path / "blocks.gguf",
        {"general.architecture": "qwen2", "general.alignment": 32},
        [
            ("norm.weight", (4,), mc.GGML_F32, f32_values.tobytes()),
            ("q8.weight", (32,), mc.GGML_Q8_0, q8_raw),
            ("q4.weight", (32,), mc.GGML_Q4_0, Q4_0_GOLDEN),
        ],
    )
    gguf = mc.read_gguf(path)
    assert gguf.metadata["general.architecture"] == "qwen2"
    assert gguf.tensors["q4.weight"].ggml_type == mc.GGML_Q4_0
    np.testing.assert_array_equal(mc.gguf_values(gguf, gguf.tensors["norm.weight"]), f32_values.astype(np.float32))
    q8_expected = np.arange(-16, 16, dtype=np.float32) * np.float32(0.5)
    np.testing.assert_array_equal(mc.gguf_values(gguf, gguf.tensors["q8.weight"]), q8_expected)
    np.testing.assert_array_equal(mc.gguf_values(gguf, gguf.tensors["q4.weight"]), Q4_0_GOLDEN_VALUES)


def test_gguf_reader_refuses_other_versions_and_types(tmp_path):
    path = tmp_path / "v2.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 2, 0, 0))
    with pytest.raises(ValueError, match="version 2"):
        mc.read_gguf(path)
    path = write_gguf(tmp_path / "odd.gguf", {"general.architecture": "qwen2"}, [("t", (32,), 1, b"\0" * 64)])
    gguf = mc.read_gguf(path)
    with pytest.raises(ValueError, match="has type 1"):
        mc.gguf_values(gguf, gguf.tensors["t"])


def test_gguf_names_map_back_to_hugging_face_names():
    assert mc.hf_name_for_gguf("token_embd.weight") == "model.embed_tokens.weight"
    assert mc.hf_name_for_gguf("output.weight") == "lm_head.weight"
    assert mc.hf_name_for_gguf("blk.23.ffn_down.weight") == "model.layers.23.mlp.down_proj.weight"
    assert mc.hf_name_for_gguf("blk.0.attn_q.bias") == "model.layers.0.self_attn.q_proj.bias"
    assert mc.hf_name_for_gguf("blk.0.rope_freqs.weight") is None
    assert mc.hf_name_for_gguf("something.else") is None


def test_gguf_substitution_writes_decoded_weights_and_checks_names(tmp_path):
    shapes = {
        "model.embed_tokens.weight": (32, 64),
        "model.norm.weight": (64,),
    }
    embed_values = np.tile(np.arange(-16, 16, dtype=np.int8), 64)  # 2048 values, 64 blocks
    blocks = b"".join(f16_bytes(0.5) + embed_values[i * 32 : (i + 1) * 32].tobytes() for i in range(64))
    norm = np.linspace(0.5, 1.5, 64, dtype="<f4")
    path = write_gguf(
        tmp_path / "model.gguf",
        {"general.architecture": "qwen2"},
        [
            ("token_embd.weight", (64, 32), mc.GGML_Q8_0, blocks),
            ("output_norm.weight", (64,), mc.GGML_F32, norm.tobytes()),
        ],
    )
    model = _tiny_model(shapes)
    report = mc.substitute_gguf_weights(model, path, torch)
    parameters = dict(model.named_parameters())
    np.testing.assert_array_equal(
        parameters["model.embed_tokens.weight"].detach().numpy(),
        (embed_values.astype(np.float32) * np.float32(0.5)).reshape(32, 64),
    )
    np.testing.assert_array_equal(parameters["model.norm.weight"].detach().numpy(), norm)
    assert report.replaced == ["model.embed_tokens.weight"]
    assert report.relative_errors["model.embed_tokens.weight"] >= 0.0

    incomplete = _tiny_model({**shapes, "model.layers.0.input_layernorm.weight": (64,)})
    with pytest.raises(ValueError, match="lacks parameters"):
        mc.substitute_gguf_weights(incomplete, path, torch)


def test_gguf_substitution_refuses_other_architectures(tmp_path):
    path = write_gguf(tmp_path / "llama.gguf", {"general.architecture": "llama"}, [])
    with pytest.raises(ValueError, match="qwen2"):
        mc.substitute_gguf_weights(_tiny_model({"model.norm.weight": (4,)}), path, torch)


# --- INT8 ------------------------------------------------------------------------------------


def test_int8_substitution_decodes_qdata_times_scale(tmp_path):
    manifest = {
        "schema": mc.INT8_SCHEMA,
        "tensors": {
            "model.embed_tokens.weight": {
                "action": "quantized", "original_shape": [1, 5], "qdata_name": "model.embed_tokens.weight.qdata",
                "scale_name": "model.embed_tokens.weight.scale",
            },
            "model.norm.weight": {"action": "preserved", "original_shape": [5], "tensor_name": "model.norm.weight"},
        },
    }
    path = tmp_path / "int8.safetensors"
    save_file(
        {
            "model.embed_tokens.weight.qdata": torch.tensor([[-127, -1, 0, 1, 127]], dtype=torch.int8),
            "model.embed_tokens.weight.scale": torch.tensor(0.5, dtype=torch.float32),
            "model.norm.weight": torch.arange(5, dtype=torch.float32),
        },
        str(path),
        metadata={"modelq.format": "modelq-native", "modelq.quantization": "int8", "modelq.manifest": json.dumps(manifest)},
    )
    model = _tiny_model({"model.embed_tokens.weight": (1, 5), "model.norm.weight": (5,)})
    report = mc.substitute_int8_weights(model, path, torch)
    parameters = dict(model.named_parameters())
    np.testing.assert_array_equal(
        parameters["model.embed_tokens.weight"].detach().numpy(),
        np.array([[-63.5, -0.5, 0.0, 0.5, 63.5]], dtype=np.float32),
    )
    np.testing.assert_array_equal(parameters["model.norm.weight"].detach().numpy(), np.arange(5, dtype=np.float32))
    assert report.replaced == ["model.embed_tokens.weight"]


# --- NVFP4, native ---------------------------------------------------------------------------


def test_native_nvfp4_decodes_the_reference_block():
    # Codes 0..15 in order, low nibble first; block scale 1.0 (E4M3 0x38) and global scale 1.0.
    packed = np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=np.uint8)
    assert vm.E4M3[0x38] == 1.0 and vm.E4M3[0x40] == 2.0
    expected = np.array(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
    )
    np.testing.assert_array_equal(mc.decode_nvfp4(packed, np.array([0x38], np.uint8), 1.0, 16), expected)
    # Block scale 2.0 and global scale 0.25 scale every value by 0.5.
    np.testing.assert_array_equal(
        mc.decode_nvfp4(packed, np.array([0x40], np.uint8), 0.25, 16), expected * np.float32(0.5)
    )


def test_native_nvfp4_refuses_mismatched_fields():
    with pytest.raises(ValueError, match="packed bytes"):
        mc.decode_nvfp4(np.zeros(3, np.uint8), np.array([0x38], np.uint8), 1.0, 16)
    with pytest.raises(ValueError, match="block scales"):
        mc.decode_nvfp4(np.zeros(8, np.uint8), np.zeros(2, np.uint8), 1.0, 16)
    with pytest.raises(ValueError, match="finite E4M3"):
        mc.decode_nvfp4(np.zeros(8, np.uint8), np.array([0x7F], np.uint8), 1.0, 16)


def test_native_nvfp4_substitution_matches_the_decoder(tmp_path):
    packed = np.arange(16, dtype=np.uint8).reshape(2, 8) * np.uint8(17)
    block_scales = np.array([[0x38], [0x40]], dtype=np.uint8)
    manifest = {
        "schema": mc.NVFP4_SCHEMA,
        "tensors": {
            "w": {
                "action": "quantized", "original_shape": [2, 16], "qdata_name": "w.qdata",
                "block_scale_name": "w.block_scale", "global_scale_name": "w.global_scale",
            },
        },
    }
    path = tmp_path / "nvfp4.safetensors"
    save_file(
        {
            "w.qdata": torch.from_numpy(packed),
            "w.block_scale": torch.from_numpy(block_scales),
            "w.global_scale": torch.tensor(0.5, dtype=torch.float32),
        },
        str(path),
        metadata={"modelq.format": "modelq-native", "modelq.quantization": "nvfp4", "modelq.manifest": json.dumps(manifest)},
    )
    model = _tiny_model({"w": (2, 16)})
    mc.substitute_nvfp4_native_weights(model, path, torch)
    expected = np.stack(
        [
            mc.decode_nvfp4(packed[0], block_scales[0], 0.5, 16),
            mc.decode_nvfp4(packed[1], block_scales[1], 0.5, 16),
        ]
    )
    np.testing.assert_array_equal(dict(model.named_parameters())["w"].detach().numpy(), expected)


# --- dispatch --------------------------------------------------------------------------------


def test_substitute_container_dispatches_by_format(tmp_path):
    path = write_gguf(
        tmp_path / "dispatch.gguf",
        {"general.architecture": "qwen2"},
        [("output_norm.weight", (4,), mc.GGML_F32, np.arange(4, dtype="<f4").tobytes())],
    )
    model = _tiny_model({"model.norm.weight": (4,)})
    kind, report = mc.substitute_container(model, path, torch)
    assert kind == "gguf"
    assert report.replaced == []
    np.testing.assert_array_equal(dict(model.named_parameters())["model.norm.weight"].detach().numpy(), np.arange(4))

