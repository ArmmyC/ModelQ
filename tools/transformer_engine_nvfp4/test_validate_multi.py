import copy
import json
import pathlib
import tempfile
import unittest

import numpy as np
from safetensors.numpy import save_file

import validate_multi as vm

BASE_MANIFEST = {
    "schema_version": 2,
    "profile_id": "transformer-engine.nvfp4.rowwise.1x16.v1",
    "runtime": {"name": "transformer_engine", "version": "2.19.0"},
    "quantization": {
        "data_format": "E2M1",
        "block_scale_format": "E4M3",
        "block_size": 16,
        "scaling": "rowwise_1x16_tensor_global",
    },
    "scale_storage": {"padding": [128, 4], "gemm_swizzled": False},
    "global_scale_denominator": 2688.0,
    "tensors": {},
}


def padded_shape(rows, columns):
    return (-(-rows // 128) * 128, -(-(columns // 16) // 4) * 4)


def matrix_tensors(name, rows, columns, *, seed=0):
    """A valid matrix: nonzero scales (E4M3 1.0 = 0x38), random codes, zero padding."""
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, size=(rows, columns // 2), dtype=np.uint8)
    scale = np.zeros(padded_shape(rows, columns), dtype=np.uint8)
    scale[:rows, : columns // 16] = 0x38
    return {
        f"{name}.rowwise_data": data,
        f"{name}.rowwise_scale_inv": scale,
        f"{name}.amax_rowwise": np.array([2.0 * 2688.0 / 8.0], dtype=np.float32),
    }


def matrix_entry(name, rows, columns):
    return {
        "action": "quantized",
        "logical_shape": [rows, columns],
        "original_dtype": "F32",
        "fields": {
            "rowwise_data": f"{name}.rowwise_data",
            "rowwise_scale_inv": f"{name}.rowwise_scale_inv",
            "amax_rowwise": f"{name}.amax_rowwise",
        },
    }


def write_container(path, *, matrices=(("w", 48, 96),), preserved=None, manifest=None,
                    tensors=None, file_format=vm.FORMAT, drop=()):
    actual_manifest = copy.deepcopy(BASE_MANIFEST if manifest is None else manifest)
    actual_tensors = {} if tensors is None else dict(tensors)
    if tensors is None:
        for index, (name, rows, columns) in enumerate(matrices):
            actual_tensors.update(matrix_tensors(name, rows, columns, seed=index))
            actual_manifest["tensors"][name] = matrix_entry(name, rows, columns)
        for name, array in (preserved or {}).items():
            actual_tensors[name] = array
            actual_manifest["tensors"][name] = {
                "action": "preserved",
                "dtype": "F32" if array.dtype == np.float32 else "U8",
                "shape": list(array.shape),
            }
    for name in drop:
        actual_tensors.pop(name)
    save_file(
        actual_tensors,
        str(path),
        metadata={"modelq.format": file_format, "modelq.manifest": json.dumps(actual_manifest)},
    )


class Base(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._dir.name)
        self.path = self.dir / "model.safetensors"

    def tearDown(self):
        self._dir.cleanup()

    def assertRejected(self, text):
        with self.assertRaisesRegex(vm.ValidationError, text):
            vm.validate_multi_container(self.path)

    def reference_file(self, name, array, *, schema=vm.REFERENCE_SCHEMA):
        path = self.dir / "reference.safetensors"
        save_file({name + vm.REFERENCE_SUFFIX: array}, str(path), metadata={"modelq.reference_schema": schema})
        return path


class MultiContainerTests(Base):

    def test_valid_container_with_preserved_tensors_passes(self):
        write_container(
            self.path,
            matrices=(("a", 48, 96), ("b", 16, 16), ("c", 144, 80)),
            preserved={"norm": np.ones(4, dtype=np.float32), "ids": np.array([1, 2, 3], dtype=np.uint8)},
        )
        result = vm.validate_multi_container(self.path)
        self.assertEqual(sorted(result.matrices), ["a", "b", "c"])
        self.assertEqual(sorted(result.preserved), ["ids", "norm"])
        self.assertEqual(result.matrices["c"].logical_shape, (144, 80))

    def test_foreign_or_malformed_files_are_rejected(self):
        write_container(self.path, file_format="transformer-engine-nvfp4-safetensors-v1")
        self.assertRejected("modelq.format")
        save_file({"x": np.zeros(1, dtype=np.float32)}, str(self.path))
        self.assertRejected("modelq.format")
        save_file({"x": np.zeros(1, dtype=np.float32)}, str(self.path),
                  metadata={"modelq.format": vm.FORMAT, "modelq.manifest": "{ not json"})
        self.assertRejected("not valid JSON")

    def test_manifest_header_values_are_enforced(self):
        for path, value in [
            (("schema_version",), 1),
            (("profile_id",), "other"),
            (("runtime", "version"), "2.18.0"),
            (("quantization", "block_size"), 32),
            (("quantization", "scaling"), "2d"),
            (("scale_storage", "gemm_swizzled"), True),
            (("scale_storage", "padding"), [64, 4]),
            (("global_scale_denominator",), 2000.0),
        ]:
            manifest = copy.deepcopy(BASE_MANIFEST)
            target = manifest
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            write_container(self.path, manifest=manifest)
            self.assertRejected(".")

    def test_field_shape_dtype_and_extra_tensor_checks(self):
        good = matrix_tensors("w", 48, 96)
        manifest = copy.deepcopy(BASE_MANIFEST)
        manifest["tensors"]["w"] = matrix_entry("w", 48, 96)
        cases = {
            "wrong scale shape": ({**good, "w.rowwise_scale_inv": np.zeros((128, 4), dtype=np.uint8)}, "rowwise_scale_inv"),
            "wrong data shape": ({**good, "w.rowwise_data": np.zeros((48, 47), dtype=np.uint8)}, "rowwise_data"),
            "wrong amax dtype": ({**good, "w.amax_rowwise": np.array([1.0], dtype=np.float64)}, "amax_rowwise"),
            "extra tensor": ({**good, "stray": np.zeros(1, dtype=np.float32)}, "not in the manifest"),
        }
        for label, (tensors, message) in cases.items():
            write_container(self.path, manifest=manifest, tensors=tensors)
            with self.subTest(label):
                self.assertRejected(message)
        missing = {k: v for k, v in good.items() if k != "w.amax_rowwise"}
        write_container(self.path, manifest=manifest, tensors=missing)
        self.assertRejected("missing")

    def test_padding_scales_and_amax_are_validated(self):
        manifest = copy.deepcopy(BASE_MANIFEST)
        manifest["tensors"]["w"] = matrix_entry("w", 48, 96)

        def corrupt(edit):
            tensors = matrix_tensors("w", 48, 96)
            edit(tensors)
            write_container(self.path, manifest=manifest, tensors=tensors)

        corrupt(lambda t: t["w.rowwise_scale_inv"].__setitem__((100, 0), 0x38))
        self.assertRejected("padding rows")
        corrupt(lambda t: t["w.rowwise_scale_inv"].__setitem__((0, 7), 0x38))
        self.assertRejected("padding columns")
        corrupt(lambda t: t["w.rowwise_scale_inv"].__setitem__((0, 0), 0xB8))
        self.assertRejected("positive finite E4M3")
        corrupt(lambda t: t["w.rowwise_scale_inv"].__setitem__((0, 0), 0x7F))
        self.assertRejected("positive finite E4M3")
        corrupt(lambda t: t.__setitem__("w.amax_rowwise", np.array([np.nan], dtype=np.float32)))
        self.assertRejected("amax")
        corrupt(lambda t: t.__setitem__("w.amax_rowwise", np.array([0.0], dtype=np.float32)))
        self.assertRejected("amax is zero")
        # A zero scale block must not hold nonzero codes (random data has some).
        corrupt(lambda t: t["w.rowwise_scale_inv"].__setitem__((0, 0), 0))
        self.assertRejected("zero-scale block")

    def test_an_all_zero_matrix_with_zero_amax_is_valid(self):
        tensors = {
            "z.rowwise_data": np.zeros((16, 8), dtype=np.uint8),
            "z.rowwise_scale_inv": np.zeros((128, 4), dtype=np.uint8),
            "z.amax_rowwise": np.zeros(1, dtype=np.float32),
        }
        manifest = copy.deepcopy(BASE_MANIFEST)
        manifest["tensors"]["z"] = matrix_entry("z", 16, 16)
        write_container(self.path, manifest=manifest, tensors=tensors)
        result = vm.validate_multi_container(self.path)
        data, scales, amax = vm.load_matrix(result.matrices["z"])
        self.assertFalse(vm.decode_matrix(16, 16, data, scales, amax).any())

    def test_decode_matches_hand_computed_values(self):
        # Codes 0..15 in one row: E2M1 magnitudes with sign in bit 3.
        codes = np.arange(16, dtype=np.uint8)
        data = (codes[0::2] | (codes[1::2] << 4)).reshape(1, 8)
        scales = np.zeros((128, 4), dtype=np.uint8)
        scales[0, 0] = 0x40  # E4M3 2.0
        amax = np.float32(2688.0 * 0.5)  # global decode scale 0.5
        decoded = vm.decode_matrix(1, 16, data, scales, float(amax))
        magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        expected = np.array(magnitudes + [-m for m in magnitudes], dtype=np.float32) * 2.0 * 0.5
        np.testing.assert_array_equal(decoded[0], expected)

    def test_sharded_containers_need_a_safe_index_and_unique_tensors(self):
        for name, spec in [("one.safetensors", ("a", 16, 16)), ("two.safetensors", ("b", 32, 32))]:
            write_container(self.dir / name, matrices=(spec,))
        index = {"weight_map": {
            "a.rowwise_data": "one.safetensors", "b.rowwise_data": "two.safetensors"}}
        (self.dir / vm.INDEX_NAME).write_text(json.dumps(index))
        result = vm.validate_multi_container(self.dir)
        self.assertEqual(sorted(result.matrices), ["a", "b"])
        self.assertEqual(len(result.files), 2)

        for bad in ["../one.safetensors", "sub/one.safetensors", "absent.safetensors"]:
            (self.dir / vm.INDEX_NAME).write_text(json.dumps({"weight_map": {"a": bad}}))
            with self.assertRaises(vm.ValidationError):
                vm.validate_multi_container(self.dir)
        (self.dir / vm.INDEX_NAME).write_text("{ nope")
        with self.assertRaises(vm.ValidationError):
            vm.validate_multi_container(self.dir)
        # The same matrix in two shards is rejected.
        write_container(self.dir / "two.safetensors", matrices=(("a", 16, 16),))
        (self.dir / vm.INDEX_NAME).write_text(json.dumps(index))
        with self.assertRaisesRegex(vm.ValidationError, "more than one shard"):
            vm.validate_multi_container(self.dir)

    def test_reference_comparison_accepts_the_decode_and_rejects_differences(self):
        write_container(self.path, matrices=(("w", 48, 96),))
        container = vm.validate_multi_container(self.path)
        data, scales, amax = vm.load_matrix(container.matrices["w"])
        decoded = vm.decode_matrix(48, 96, data, scales, amax)
        self.assertEqual(vm.validate_reference_multi(container, self.reference_file("w", decoded)), 0.0)

        wrong = decoded.copy()
        wrong[3, 3] += 1.0
        with self.assertRaisesRegex(vm.ValidationError, "does not decode"):
            vm.validate_reference_multi(container, self.reference_file("w", wrong))
        with self.assertRaisesRegex(vm.ValidationError, "reference_schema"):
            vm.validate_reference_multi(container, self.reference_file("w", decoded, schema="other"))
        with self.assertRaisesRegex(vm.ValidationError, "must be F32"):
            vm.validate_reference_multi(container, self.reference_file("w", decoded[:10]))
        with self.assertRaisesRegex(vm.ValidationError, "do not match"):
            vm.validate_reference_multi(container, self.reference_file("other", decoded))

    def test_command_line_exit_codes(self):
        write_container(self.path)
        self.assertEqual(vm.main(["cpu", "--container", str(self.path)]), 0)
        self.assertEqual(vm.main(["cpu", "--container", str(self.dir / "missing")]), 1)

    def test_runtime_mode_fails_cleanly_without_the_pinned_environment(self):
        # Neither Windows/macOS hosts nor Linux hosts without TE/CUDA can pass.
        write_container(self.path)
        reference = self.write_reference("w", 48, 96)
        self.assertEqual(
            vm.main(["runtime", "--container", str(self.path), "--reference", str(reference)]), 1
        )

    def write_reference(self, name, rows, columns):
        container = vm.validate_multi_container(self.path)
        data, scales, amax = vm.load_matrix(container.matrices[name])
        return self.reference_file(name, vm.decode_matrix(rows, columns, data, scales, amax))


class FakeBackend:
    """Stands in for Transformer Engine so the proof's control flow is testable."""

    def __init__(self, *, dequantize_offset=None, gemm_failures=(), environment_error=None):
        self.dequantize_offset = dequantize_offset or {}
        self.gemm_failures = set(gemm_failures)
        self.environment_error = environment_error
        self.loaded = []

    def environment(self):
        if self.environment_error:
            raise vm.ValidationError(self.environment_error)
        return {"transformer_engine": "2.19.0", "gpu": "fake"}

    def load_weight(self, entry, data, scale_inv, amax):
        self.loaded.append(entry.name)
        return (entry, vm.decode_matrix(*entry.logical_shape, data, scale_inv, amax))

    def dequantize(self, weight):
        entry, values = weight
        return values + self.dequantize_offset.get(entry.name, 0.0)

    def gemm_error(self, weight, reference):
        entry, _ = weight
        if entry.name in self.gemm_failures:
            raise vm.ValidationError("TN GEMM output is outside the tolerance")
        return 0.001


class RuntimeProofTests(Base):
    def prepare(self, matrices):
        write_container(self.path, matrices=matrices)
        container = vm.validate_multi_container(self.path)
        arrays = {}
        for name, entry in container.matrices.items():
            data, scales, amax = vm.load_matrix(entry)
            arrays[name + vm.REFERENCE_SUFFIX] = vm.decode_matrix(
                *entry.logical_shape, data, scales, amax
            )
        reference = self.dir / "reference.safetensors"
        save_file(arrays, str(reference), metadata={"modelq.reference_schema": vm.REFERENCE_SCHEMA})
        return reference

    def test_every_matrix_is_loaded_and_reported(self):
        reference = self.prepare((("a", 16, 16), ("b", 48, 96), ("c", 144, 80)))
        backend = FakeBackend()
        outcome = vm.run_runtime_proof(self.path, reference, backend)
        self.assertTrue(outcome["passed"])
        self.assertEqual(backend.loaded, ["a", "b", "c"])
        self.assertEqual([m["status"] for m in outcome["matrices"]], ["pass"] * 3)
        self.assertEqual(outcome["environment"]["gpu"], "fake")
        self.assertEqual(outcome["matrices"][1]["shape"], [48, 96])

    def test_a_failing_matrix_does_not_hide_the_others(self):
        reference = self.prepare((("a", 16, 16), ("b", 48, 96), ("c", 144, 80)))
        backend = FakeBackend(gemm_failures={"b"}, dequantize_offset={"c": 0.5})
        outcome = vm.run_runtime_proof(self.path, reference, backend)
        statuses = {m["name"]: m for m in outcome["matrices"]}
        self.assertFalse(outcome["passed"])
        self.assertEqual(statuses["a"]["status"], "pass")
        self.assertEqual(statuses["b"]["status"], "fail")
        self.assertIn("tolerance", statuses["b"]["detail"])
        self.assertEqual(statuses["c"]["status"], "fail")
        self.assertIn("dequantization does not match", statuses["c"]["detail"])
        # All three were attempted in one run.
        self.assertEqual(backend.loaded, ["a", "b", "c"])

    def test_an_unusable_environment_aborts_before_any_matrix(self):
        reference = self.prepare((("a", 16, 16),))
        backend = FakeBackend(environment_error="runtime proof requires Linux")
        with self.assertRaisesRegex(vm.ValidationError, "requires Linux"):
            vm.run_runtime_proof(self.path, reference, backend)
        self.assertEqual(backend.loaded, [])

    def test_a_bad_container_or_reference_is_rejected_before_the_runtime(self):
        reference = self.prepare((("a", 16, 16),))
        wrong = self.reference_file("other", np.zeros((16, 16), dtype=np.float32))
        backend = FakeBackend()
        with self.assertRaises(vm.ValidationError):
            vm.run_runtime_proof(self.path, wrong, backend)
        self.assertEqual(backend.loaded, [])
        self.assertTrue(reference.exists())

    def test_the_second_operand_recipe_matches_task_28(self):
        values = vm.second_operand_values(64)
        self.assertEqual(values.shape, (64, 64))
        indices = np.arange(4096, dtype=np.int32)
        expected = (((indices * 37) % 251) - 125).astype(np.float32) / np.float32(32.0)
        np.testing.assert_array_equal(values.reshape(-1), expected)
        self.assertEqual(vm.second_operand_values(80).shape, (64, 80))


if __name__ == "__main__":
    unittest.main()
