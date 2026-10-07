import math
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

import quality_eval as qe

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "transformer_engine_nvfp4"))
import test_validate_multi as tvm  # noqa: E402  (container-building helpers)
import validate_multi as vm  # noqa: E402


class Output:
    def __init__(self, logits):
        self.logits = logits


class ToyLM(nn.Module):
    """Embedding, one 32x32 linear projection, and an output head."""

    def __init__(self, vocab=40, width=32, seed=0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.embed = nn.Embedding(vocab, width)
        self.proj = nn.Linear(width, width, bias=False)
        self.head = nn.Linear(width, vocab, bias=False)
        for parameter in self.parameters():
            parameter.data = torch.randn(parameter.shape, generator=generator)

    def forward(self, ids):
        return Output(self.head(torch.tanh(self.proj(self.embed(ids)))))


class WindowTests(unittest.TestCase):
    def test_windows_are_non_overlapping_and_drop_the_tail(self):
        self.assertEqual(qe.make_windows(range(10), 4), [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(qe.make_windows(range(8), 4), [[0, 1, 2, 3], [4, 5, 6, 7]])
        self.assertEqual(qe.make_windows(range(3), 4), [])
        with self.assertRaises(ValueError):
            qe.make_windows(range(10), 1)


class MetricTests(unittest.TestCase):
    def windows(self, count=3, length=12, vocab=40):
        rng = np.random.default_rng(1)
        return [rng.integers(0, vocab, size=length).tolist() for _ in range(count)]

    def test_identical_models_have_zero_kl_and_full_agreement(self):
        model = ToyLM()
        result = qe.evaluate_pair(model, ToyLM(), self.windows(), torch, torch.device("cpu"))
        self.assertAlmostEqual(result["mean_kl_original_to_quantized"], 0.0, places=9)
        self.assertEqual(result["top1_agreement"], 1.0)
        self.assertAlmostEqual(result["perplexity_original"], result["perplexity_quantized"], places=9)
        self.assertEqual(result["positions"], 3 * 11)

    def test_a_perturbed_model_is_worse_and_measurably_different(self):
        base, other = ToyLM(), ToyLM()
        with torch.no_grad():
            other.head.weight.add_(torch.randn_like(other.head.weight) * 0.8)
        result = qe.evaluate_pair(base, other, self.windows(), torch, torch.device("cpu"))
        self.assertGreater(result["mean_kl_original_to_quantized"], 0.01)
        self.assertLess(result["top1_agreement"], 1.0)
        self.assertNotEqual(result["perplexity_original"], result["perplexity_quantized"])

    def test_perplexity_matches_a_direct_computation(self):
        model = ToyLM()
        windows = self.windows(count=2, length=9)
        result = qe.evaluate_pair(model, ToyLM(), windows, torch, torch.device("cpu"))
        nll = 0.0
        for window in windows:
            ids = torch.tensor([window])
            logits = model(ids).logits[0, :-1]
            nll += float(nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum"))
        self.assertAlmostEqual(result["perplexity_original"], math.exp(nll / (2 * 8)), places=6)

    def test_kl_is_asymmetric_non_negative_and_matches_the_definition(self):
        base, other = ToyLM(), ToyLM()
        with torch.no_grad():
            other.head.weight.mul_(1.5)
        windows = self.windows(count=1, length=10)
        ids = torch.tensor([windows[0]])
        p = torch.log_softmax(base(ids).logits[0, :-1], -1)
        q = torch.log_softmax(other(ids).logits[0, :-1], -1)
        expected = float((p.exp() * (p - q)).sum()) / 9
        result = qe.evaluate_pair(base, other, windows, torch, torch.device("cpu"))
        self.assertAlmostEqual(result["mean_kl_original_to_quantized"], expected, places=6)
        self.assertGreaterEqual(result["mean_kl_original_to_quantized"], 0.0)

    def test_text_shorter_than_a_window_is_an_error(self):
        with self.assertRaises(ValueError):
            qe.evaluate_pair(ToyLM(), ToyLM(), [], torch, torch.device("cpu"))

    def test_relative_perplexity_increase(self):
        self.assertAlmostEqual(
            qe.relative_perplexity_increase({"perplexity_original": 10.0, "perplexity_quantized": 11.0}), 0.1
        )


class SubstitutionTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._dir.name)

    def tearDown(self):
        self._dir.cleanup()

    def container(self, matrices, preserved=None):
        path = self.dir / "te.safetensors"
        tvm.write_container(path, matrices=matrices, preserved=preserved)
        return path

    def test_matching_matrices_are_replaced_with_the_decoded_values(self):
        model = ToyLM()
        path = self.container((("proj.weight", 32, 32),))
        before = model.proj.weight.detach().clone()
        head_before = model.head.weight.detach().clone()
        report = qe.substitute_nvfp4_weights(model, path, torch)

        container = vm.validate_multi_container(path)
        data, scales, amax = vm.load_matrix(container.matrices["proj.weight"])
        expected = vm.decode_matrix(32, 32, data, scales, amax)
        np.testing.assert_array_equal(model.proj.weight.detach().numpy(), expected)
        self.assertFalse(torch.equal(model.proj.weight, before))
        self.assertTrue(torch.equal(model.head.weight, head_before))
        self.assertEqual(report.replaced, ["proj.weight"])
        self.assertGreater(report.relative_errors["proj.weight"], 0.0)
        self.assertEqual(report.summary()["matrices_replaced"], 1)

    def test_unknown_names_and_shape_mismatches_are_errors(self):
        model = ToyLM()
        with self.assertRaisesRegex(ValueError, "no matching model parameter"):
            qe.substitute_nvfp4_weights(model, self.container((("missing.weight", 32, 32),)), torch)
        with self.assertRaisesRegex(ValueError, "parameter shape"):
            qe.substitute_nvfp4_weights(model, self.container((("proj.weight", 64, 64),)), torch)


if __name__ == "__main__":
    unittest.main()
