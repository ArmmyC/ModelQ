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


def write_bf16_file(path, *, reference):
    """Write valid SafeTensors bytes containing a NumPy-unsupported BF16 entry."""
    if reference:
        metadata = {"modelq.reference_schema": "transformer-engine-nvfp4-reference-v1"}
        entries = [("weight.dequantized_reference", "BF16", [2, 16], bytes(64))]
    else:
        metadata = {
            "modelq.format": "transformer-engine-nvfp4-safetensors-v1",
            "modelq.manifest": json.dumps(VALID_MANIFEST, sort_keys=True, separators=(",", ":")),
        }
        entries = [
            ("weight.rowwise_data", "BF16", [2, 8], bytes(32)),
            ("weight.rowwise_scale_inv", "U8", [128, 4], bytes(512)),
            ("weight.amax_rowwise", "F32", [1], bytes(4)),
        ]
    header = {"__metadata__": metadata}
    payload = bytearray()
    for name, dtype, shape, data in entries:
        start = len(payload)
        payload.extend(data)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(payload)]}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * (-len(header_bytes) % 8)
    path.write_bytes(len(header_bytes).to_bytes(8, "little") + header_bytes + payload)
    return path


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

    def test_container_bf16_decode_error_is_validation_error(self):
        path = write_bf16_file(self.temp_directory / "bf16-container.safetensors", reference=False)
        with self.assertRaisesRegex(
            validate.ValidationError, "bf16-container.safetensors.*cannot read SafeTensors artifact"
        ):
            validate.validate_container(path)

    def test_reference_bf16_decode_error_is_validation_error(self):
        path = write_bf16_file(self.temp_directory / "bf16-reference.safetensors", reference=True)
        with self.assertRaisesRegex(
            validate.ValidationError, "bf16-reference.safetensors.*cannot read reference SafeTensors"
        ):
            validate.validate_reference(path, (2, 16))

    def test_runtime_preflight_rejects_wrong_te_version(self):
        with self.assertRaisesRegex(validate.ValidationError, "2.19.0"):
            validate.validate_runtime_preflight("Linux", "x86_64", "2.19.1", True, "12.8", 90300, (10, 0))

    def test_runtime_preflight_rejects_missing_cuda(self):
        with self.assertRaisesRegex(validate.ValidationError, "CUDA"):
            validate.validate_runtime_preflight("Linux", "x86_64", "2.19.0", False, None, None, None)

    def test_runtime_preflight_rejects_old_cuda_and_cudnn(self):
        with self.assertRaisesRegex(validate.ValidationError, "CUDA 12.8"):
            validate.validate_runtime_preflight("Linux", "x86_64", "2.19.0", True, "12.1", 90300, (10, 0))
        with self.assertRaisesRegex(validate.ValidationError, "cuDNN 9.3"):
            validate.validate_runtime_preflight("Linux", "x86_64", "2.19.0", True, "12.8", 90200, (10, 0))

    def test_runtime_preflight_rejects_pre_blackwell_gpu(self):
        with self.assertRaisesRegex(validate.ValidationError, "compute capability"):
            validate.validate_runtime_preflight("Linux", "x86_64", "2.19.0", True, "12.8", 90300, (9, 0))

    def test_runtime_preflight_rejects_unsupported_platform(self):
        with self.assertRaisesRegex(validate.ValidationError, "Linux"):
            validate.validate_runtime_preflight("Windows", "x86_64", "2.19.0", True, "12.8", 90300, (10, 0))
        with self.assertRaisesRegex(validate.ValidationError, "x86_64"):
            validate.validate_runtime_preflight("Linux", "aarch64", "2.19.0", True, "12.8", 90300, (10, 0))

    def test_runtime_preflight_accepts_supported_environment(self):
        self.assertIsNone(validate.validate_runtime_preflight(
            "Linux", "AMD64", "2.19.0", True, "12.8", 90300, (10, 0)
        ))

    def test_tn_reference_output_uses_rhs_times_weight_transpose(self):
        weight = np.array([[1, 2], [3, 4]], dtype=np.float32)
        rhs = np.array([[5, 6], [7, 8]], dtype=np.float32)
        self.assertTrue(callable(getattr(validate, "tn_reference_output", None)))
        actual = validate.tn_reference_output(rhs, weight)
        np.testing.assert_array_equal(
            actual, np.array([[17, 39], [23, 53]], dtype=np.float32)
        )

    def test_cpu_fixture_does_not_import_gpu_dependencies(self):
        artifact_path = write_artifact(self.temp_directory / "runtime.safetensors")
        reference_path = write_reference(self.temp_directory / "reference.safetensors")
        real_import = builtins.__import__

        def reject_gpu_imports(name, *args, **kwargs):
            if name == "torch" or name.startswith("transformer_engine"):
                raise AssertionError(f"CPU validation imported {name}")
            return real_import(name, *args, **kwargs)

        with unittest.mock.patch("builtins.__import__", side_effect=reject_gpu_imports):
            artifact, reference = validate.validate_cpu_fixture(artifact_path, reference_path)
        self.assertEqual(artifact.manifest["logical_shape"], [2, 16])
        self.assertEqual(reference.shape, (2, 16))

    def test_runtime_validates_artifacts_before_platform_or_gpu_imports(self):
        artifact_path = write_artifact(self.temp_directory / "runtime.safetensors")
        reference_path = write_reference(
            self.temp_directory / "reference.safetensors",
            values=np.zeros((1, 16), dtype=np.float32),
        )
        with self.assertRaisesRegex(validate.ValidationError, "shape"):
            validate.run_blackwell_gemm(artifact_path, reference_path)

    def test_runtime_rejects_windows_before_gpu_imports(self):
        manifest = copy.deepcopy(VALID_MANIFEST)
        manifest["logical_shape"] = [64, 64]
        tensors = {
            "weight.rowwise_data": np.zeros((64, 32), dtype=np.uint8),
            "weight.rowwise_scale_inv": np.zeros((128, 4), dtype=np.uint8),
            "weight.amax_rowwise": np.zeros((1,), dtype=np.float32),
        }
        artifact_path = write_artifact(
            self.temp_directory / "runtime.safetensors", manifest=manifest, tensors=tensors
        )
        reference_path = write_reference(
            self.temp_directory / "reference.safetensors",
            values=np.zeros((64, 64), dtype=np.float32),
        )
        real_import = builtins.__import__

        def reject_gpu_imports(name, *args, **kwargs):
            if name == "torch" or name.startswith("transformer_engine"):
                raise AssertionError(f"runtime imported {name} before platform preflight")
            return real_import(name, *args, **kwargs)

        with unittest.mock.patch("builtins.__import__", side_effect=reject_gpu_imports):
            with unittest.mock.patch.object(validate.platform, "system", return_value="Windows"):
                with self.assertRaisesRegex(validate.ValidationError, "Linux"):
                    validate.run_blackwell_gemm(artifact_path, reference_path)


if __name__ == "__main__":
    unittest.main()
