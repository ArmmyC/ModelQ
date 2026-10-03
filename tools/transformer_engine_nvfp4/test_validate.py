import builtins
import copy
import json
import pathlib
import tempfile
import unittest
import unittest.mock

import numpy as np
from safetensors.numpy import save_file

import validate


VALID_MANIFEST = {
    "schema_version": 1,
    "profile_id": "transformer-engine.nvfp4.rowwise.1x16.v1",
    "runtime": {"name": "transformer_engine", "version": "2.19.0"},
    "tensor_name": "weight",
    "logical_shape": [2, 16],
    "fields": {
        "rowwise_data": "weight.rowwise_data",
        "rowwise_scale_inv": "weight.rowwise_scale_inv",
        "amax_rowwise": "weight.amax_rowwise",
    },
    "quantization": {
        "data_format": "E2M1",
        "block_scale_format": "E4M3",
        "block_size": 16,
        "scaling": "rowwise_1x16_tensor_global",
    },
    "scale_storage": {"padding": [128, 4], "gemm_swizzled": False},
    "global_scale_denominator": 2688.0,
}


def valid_tensors():
    return {
        "weight.rowwise_data": np.zeros((2, 8), dtype=np.uint8),
        "weight.rowwise_scale_inv": np.zeros((128, 4), dtype=np.uint8),
        "weight.amax_rowwise": np.zeros((1,), dtype=np.float32),
    }


def write_artifact(
    path, *, manifest=None, tensors=None,
    file_format="transformer-engine-nvfp4-safetensors-v1",
):
    actual_manifest = copy.deepcopy(VALID_MANIFEST if manifest is None else manifest)
    actual_tensors = valid_tensors() if tensors is None else tensors
    metadata = {
        "modelq.format": file_format,
        "modelq.manifest": json.dumps(actual_manifest, sort_keys=True, separators=(",", ":")),
    }
    save_file(actual_tensors, str(path), metadata=metadata)
    return pathlib.Path(path)


def write_reference(path, *, values=None, schema="transformer-engine-nvfp4-reference-v1"):
    if values is None:
        values = np.zeros((2, 16), dtype=np.float32)
    save_file(
        {"weight.dequantized_reference": values},
        str(path),
        metadata={"modelq.reference_schema": schema},
    )
    return pathlib.Path(path)


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.temp_directory = pathlib.Path(self.temp.name)

    def test_accepts_valid_artifact(self):
        path = write_artifact(self.temp_directory / "valid.safetensors")
        artifact = validate.validate_container(path)
        self.assertEqual(artifact.manifest, VALID_MANIFEST)
        self.assertEqual(artifact.tensors["weight.rowwise_data"].shape, (2, 8))

    def test_accepts_separate_reference(self):
        path = write_reference(self.temp_directory / "reference.safetensors")
        reference = validate.validate_reference(path, (2, 16))
        self.assertEqual(reference.shape, (2, 16))
        self.assertEqual(reference.dtype, np.float32)

    def test_rejects_missing_scale_field(self):
        tensors = valid_tensors()
        del tensors["weight.rowwise_scale_inv"]
        path = write_artifact(self.temp_directory / "missing-scale.safetensors", tensors=tensors)
        with self.assertRaisesRegex(validate.ValidationError, "rowwise_scale_inv"):
            validate.validate_container(path)

    def test_rejects_wrong_format(self):
        path = write_artifact(self.temp_directory / "format.safetensors", file_format="wrong")
        with self.assertRaisesRegex(validate.ValidationError, "modelq.format"):
            validate.validate_container(path)

    def test_rejects_wrong_te_version(self):
        manifest = copy.deepcopy(VALID_MANIFEST)
        manifest["runtime"]["version"] = "2.19.1"
        path = write_artifact(self.temp_directory / "version.safetensors", manifest=manifest)
        with self.assertRaisesRegex(validate.ValidationError, "2.19.0"):
            validate.validate_container(path)

    def test_rejects_wrong_data_dtype(self):
        tensors = valid_tensors()
        tensors["weight.rowwise_data"] = np.zeros((2, 8), dtype=np.float32)
        path = write_artifact(self.temp_directory / "dtype.safetensors", tensors=tensors)
        with self.assertRaisesRegex(validate.ValidationError, "rowwise_data.*uint8"):
            validate.validate_container(path)

    def test_rejects_wrong_data_shape(self):
        tensors = valid_tensors()
        tensors["weight.rowwise_data"] = np.zeros((2, 9), dtype=np.uint8)
        path = write_artifact(self.temp_directory / "shape.safetensors", tensors=tensors)
        with self.assertRaisesRegex(validate.ValidationError, "rowwise_data.*shape"):
            validate.validate_container(path)

    def test_rejects_nonzero_scale_padding(self):
        tensors = valid_tensors()
        tensors["weight.rowwise_scale_inv"][0, 1] = 1
        path = write_artifact(self.temp_directory / "padding.safetensors", tensors=tensors)
        with self.assertRaisesRegex(validate.ValidationError, "padding"):
            validate.validate_container(path)

    def test_rejects_nonzero_scale_padding_row(self):
        tensors = valid_tensors()
        tensors["weight.rowwise_scale_inv"][2, 0] = 1
        path = write_artifact(self.temp_directory / "padding-row.safetensors", tensors=tensors)
        with self.assertRaisesRegex(validate.ValidationError, "padding"):
            validate.validate_container(path)

    def test_rejects_extra_columnwise_tensor(self):
        tensors = valid_tensors()
        tensors["weight.columnwise_data"] = np.zeros((1,), dtype=np.uint8)
        path = write_artifact(self.temp_directory / "extra.safetensors", tensors=tensors)
        with self.assertRaisesRegex(validate.ValidationError, "unexpected"):
            validate.validate_container(path)

    def test_rejects_nonfinite_amax(self):
        tensors = valid_tensors()
        tensors["weight.amax_rowwise"][0] = np.inf
        path = write_artifact(self.temp_directory / "amax.safetensors", tensors=tensors)
        with self.assertRaisesRegex(validate.ValidationError, "amax_rowwise"):
            validate.validate_container(path)

    def test_rejects_negative_amax(self):
        tensors = valid_tensors()
        tensors["weight.amax_rowwise"][0] = -1
        path = write_artifact(self.temp_directory / "negative-amax.safetensors", tensors=tensors)
        with self.assertRaisesRegex(validate.ValidationError, "amax_rowwise"):
            validate.validate_container(path)

    def test_rejects_inconsistent_manifest_field_names(self):
        manifest = copy.deepcopy(VALID_MANIFEST)
        manifest["fields"]["rowwise_data"] = "other.rowwise_data"
        path = write_artifact(self.temp_directory / "fields.safetensors", manifest=manifest)
        with self.assertRaisesRegex(validate.ValidationError, "fields"):
            validate.validate_container(path)

    def test_accepts_cpu_fixture_pair(self):
        artifact_path = write_artifact(self.temp_directory / "runtime.safetensors")
        reference_path = write_reference(self.temp_directory / "reference.safetensors")
        artifact, reference = validate.validate_cpu_fixture(artifact_path, reference_path)
        self.assertEqual(artifact.manifest["logical_shape"], [2, 16])
        self.assertEqual(reference.shape, (2, 16))

    def test_rejects_cpu_fixture_wrong_reference_shape(self):
        artifact_path = write_artifact(self.temp_directory / "runtime.safetensors")
        reference_path = write_reference(
            self.temp_directory / "reference.safetensors",
            values=np.zeros((1, 16), dtype=np.float32),
        )
        with self.assertRaisesRegex(validate.ValidationError, "shape"):
            validate.validate_cpu_fixture(artifact_path, reference_path)

    def test_rejects_reference_wrong_schema(self):
        path = write_reference(self.temp_directory / "reference.safetensors", schema="wrong")
        with self.assertRaisesRegex(validate.ValidationError, "reference_schema"):
            validate.validate_reference(path, (2, 16))

    def test_rejects_reference_nonfinite_value(self):
        values = np.zeros((2, 16), dtype=np.float32)
        values[0, 0] = np.nan
        path = write_reference(self.temp_directory / "reference.safetensors", values=values)
        with self.assertRaisesRegex(validate.ValidationError, "finite"):
            validate.validate_reference(path, (2, 16))

    def test_rejects_reference_wrong_dtype(self):
        path = write_reference(
            self.temp_directory / "reference.safetensors",
            values=np.zeros((2, 16), dtype=np.float64),
        )
        with self.assertRaisesRegex(validate.ValidationError, "float32"):
            validate.validate_reference(path, (2, 16))

    def test_rejects_reference_extra_tensor(self):
        path = self.temp_directory / "reference.safetensors"
        save_file(
            {
                "weight.dequantized_reference": np.zeros((2, 16), dtype=np.float32),
                "weight.extra": np.zeros((1,), dtype=np.float32),
            },
            str(path),
            metadata={"modelq.reference_schema": "transformer-engine-nvfp4-reference-v1"},
        )
        with self.assertRaisesRegex(validate.ValidationError, "tensor"):
            validate.validate_reference(path, (2, 16))


if __name__ == "__main__":
    unittest.main()
