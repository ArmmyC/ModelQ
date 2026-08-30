"""Tests for the optional Transformer Engine NVFP4 fixture helper."""

from __future__ import annotations

import importlib.util
import math
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).with_name("nvfp4_reference.py")
spec = importlib.util.spec_from_file_location("nvfp4_reference", SCRIPT_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"could not load {SCRIPT_PATH}")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def _valid_document() -> dict:
    return {
        "schema": "modelq.nvfp4.reference.v1",
        "producer": {
            "name": "transformer-engine",
            "version": "2.20.0.dev0",
            "commit": "ace1873f0bca9ab52e23385f38c673164546d28a",
            "pytorch": "2.7.0",
            "cuda": "12.8",
            "device": "NVIDIA Blackwell",
            "compute_capability": "10.0",
        },
        "recipe": {
            "quantization_dim": "1x16",
            "deterministic": True,
            "disable_2d_quantization": True,
            "disable_rht": True,
            "disable_stochastic_rounding": True,
            "nvfp4_4over6": "none",
        },
        "representation": {
            "fp4": "E2M1",
            "block_scale": "E4M3",
            "block_size": 16,
            "global_scale": "F32_decode",
            "packing": "low_nibble_first",
        },
        "tensor": {
            "name": "modelq.reference",
            "source_dtype": "F32",
            "shape": [2, 16],
            "source_values_f32_bits": [0, 0x3F800000] + [0] * 30,
            "packed_u8": [0x10] * 16,
            "block_scale_u8": [0x38, 0x38],
            "global_scale_f32_bits": 0x3F800000,
            "expected_dequant_f32_bits": [0, 0x3F000000] * 16,
        },
    }


class Nvfp4ReferenceDocumentTests(unittest.TestCase):
    def test_capture_source_matches_declared_shape(self) -> None:
        self.assertEqual(len(reference.SOURCE_VALUES), 2 * 32)
        self.assertEqual(reference.SOURCE_SHAPE, [2, 32])

    def test_capture_source_uses_canonical_zero_signs(self) -> None:
        self.assertFalse(
            any(value == 0.0 and math.copysign(1.0, value) < 0 for value in reference.SOURCE_VALUES)
        )

    def test_valid_document_is_normalized(self) -> None:
        document = _valid_document()

        normalized = reference.validate_fixture_document(document)

        self.assertEqual(normalized["schema"], "modelq.nvfp4.reference.v1")
        self.assertEqual(normalized["tensor"]["shape"], [2, 16])

    def test_rejects_wrong_schema(self) -> None:
        document = _valid_document()
        document["schema"] = "modelq.nvfp4.reference.v0"

        with self.assertRaises(reference.FixtureError):
            reference.validate_fixture_document(document)

    def test_rejects_inconsistent_payload_lengths(self) -> None:
        document = _valid_document()
        document["tensor"]["packed_u8"] = [0]

        with self.assertRaisesRegex(reference.FixtureError, "packed_u8"):
            reference.validate_fixture_document(document)

    def test_rejects_non_deterministic_recipe(self) -> None:
        document = _valid_document()
        document["recipe"]["disable_stochastic_rounding"] = False

        with self.assertRaisesRegex(reference.FixtureError, "disable_stochastic_rounding"):
            reference.validate_fixture_document(document)

    def test_rejects_non_f32_source_schema(self) -> None:
        document = _valid_document()
        document["tensor"]["source_dtype"] = "BF16"

        with self.assertRaisesRegex(reference.FixtureError, "source_dtype"):
            reference.validate_fixture_document(document)


if __name__ == "__main__":
    unittest.main()
