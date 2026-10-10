"""Tests for the calibration CLI behind `modelq quantize --calibration awq` (ADR 0037). Nothing is downloaded."""

import pytest

import modelq_awq as cli

# ADR 0031's recorded window starts for the 2,518,423-token train split (the first, second and last).
ADR_0031_STARTS = {0: 0, 1: 81223, 31: 2517911}
TRAIN_TOKENS = 2518423


def test_offsets_reproduce_the_recorded_calibration_windows():
    offsets = cli.calibration_offsets(TRAIN_TOKENS, cli.CALIBRATION_WINDOWS, cli.CALIBRATION_LENGTH)
    assert len(offsets) == cli.CALIBRATION_WINDOWS == 32
    for index, start in ADR_0031_STARTS.items():
        assert offsets[index] == start
    assert offsets == sorted(offsets) and len(set(offsets)) == len(offsets)


def test_offsets_refuse_too_little_text_or_too_few_windows():
    with pytest.raises(ValueError, match="fewer than one"):
        cli.calibration_offsets(100, 32, 512)
    with pytest.raises(ValueError, match="at least two"):
        cli.calibration_offsets(10_000, 1, 512)


def test_without_data_or_download_nothing_is_read(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(SystemExit, match="--calibration-data"):
        cli.main(["--model", str(tmp_path), "--output", str(tmp_path / "o.safetensors"), "--report", str(tmp_path / "r.json")])


def test_a_missing_calibration_file_is_reported(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    with pytest.raises(SystemExit, match="calibration data not found"):
        cli.main(
            [
                "--model", str(tmp_path),
                "--calibration-data", str(tmp_path / "missing.parquet"),
                "--output", str(tmp_path / "o.safetensors"),
                "--report", str(tmp_path / "r.json"),
            ]
        )


def test_a_model_directory_needs_one_checkpoint_file(tmp_path):
    data = tmp_path / "train.parquet"
    data.write_bytes(b"")
    with pytest.raises(SystemExit, match="model.safetensors"):
        cli.main(
            [
                "--model", str(tmp_path),
                "--calibration-data", str(data),
                "--output", str(tmp_path / "o.safetensors"),
                "--report", str(tmp_path / "r.json"),
            ]
        )
