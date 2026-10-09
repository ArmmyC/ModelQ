"""Tests for the local evaluation CLI that do not download anything."""

import json
import pathlib

import pytest

import modelq_eval as me

REPO = pathlib.Path(__file__).resolve().parents[2]


def test_pinned_revisions_match_the_recorded_evidence():
    recorded = json.loads(
        (REPO / "docs" / "validation" / "nvfp4-quality-qwen2.5-0.5b-wikitext2.json").read_text(encoding="utf-8")
    )
    assert me.PINNED_MODEL[recorded["model"]["id"]] == recorded["model"]["revision"]
    assert me.DATASET_REVISION == recorded["dataset"]["revision"]
    assert me.DATASET_SHA256 == recorded["dataset"]["sha256"]


def test_download_is_refused_for_unpinned_models(tmp_path):
    container = tmp_path / "te.safetensors"
    container.write_bytes(b"")
    with pytest.raises(SystemExit, match="pinned models"):
        me.main(
            [
                "--model", "someone/other-model", "--download",
                "--container", str(container), "--report", str(tmp_path / "r.json"),
            ]
        )


def test_local_model_directory_must_contain_weights(tmp_path):
    container = tmp_path / "te.safetensors"
    container.write_bytes(b"")
    with pytest.raises(SystemExit, match="not a local model directory"):
        me.main(
            [
                "--model", str(tmp_path), "--dataset", str(tmp_path / "d.parquet"),
                "--container", str(container), "--report", str(tmp_path / "r.json"),
            ]
        )


def test_local_run_needs_an_explicit_dataset(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    container = tmp_path / "te.safetensors"
    container.write_bytes(b"")
    with pytest.raises(SystemExit, match="--dataset"):
        me.main(
            [
                "--model", str(tmp_path),
                "--container", str(container), "--report", str(tmp_path / "r.json"),
            ]
        )


def test_missing_container_is_reported_before_any_model_is_loaded(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(SystemExit, match="container not found"):
        me.main(
            [
                "--model", str(tmp_path), "--dataset", str(tmp_path / "d.parquet"),
                "--container", str(tmp_path / "missing.safetensors"),
                "--report", str(tmp_path / "r.json"),
            ]
        )


def test_device_request_for_cuda_fails_clearly_without_cuda(tmp_path):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available here")
    with pytest.raises(SystemExit, match="CUDA is not available"):
        me._resolve_device(torch, "cuda")
