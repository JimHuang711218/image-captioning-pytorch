# tests/test_benchmark_metadata.py

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from unittest.mock import patch
import subprocess

import pytest

from evaluation.benchmark_metadata import (
    BenchmarkMetadata,
    collect_benchmark_metadata,
    get_git_commit,
)


def test_collect_benchmark_metadata():
    metadata = collect_benchmark_metadata(
        device="cpu",
        warmup_runs=10,
        benchmark_runs=100,
    )

    assert isinstance(metadata, BenchmarkMetadata)

    assert metadata.device == "cpu"
    assert metadata.warmup_runs == 10
    assert metadata.benchmark_runs == 100

    assert metadata.python_version
    assert metadata.pytorch_version
    assert metadata.platform
    assert metadata.machine


def test_metadata_timestamp_is_utc():
    metadata = collect_benchmark_metadata(
        device="cpu",
        warmup_runs=10,
        benchmark_runs=100,
    )

    timestamp = datetime.fromisoformat(
        metadata.timestamp_utc
    )

    assert timestamp.tzinfo is not None
    assert timestamp.utcoffset() == timedelta(0)


def test_get_git_commit_success():
    fake_result = subprocess.CompletedProcess(
        args=["git", "rev-parse", "HEAD"],
        returncode=0,
        stdout="fac95d8\n",
        stderr="",
    )

    with patch(
        "evaluation.benchmark_metadata.subprocess.run",
        return_value=fake_result,
    ) as mock_run:

        git_commit = get_git_commit()

    assert git_commit == "fac95d8"

    mock_run.assert_called_once_with(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )


def test_get_git_commit_empty_output():
    fake_result = subprocess.CompletedProcess(
        args=["git", "rev-parse", "HEAD"],
        returncode=0,
        stdout=" \n",
        stderr="",
    )

    with patch(
        "evaluation.benchmark_metadata.subprocess.run",
        return_value=fake_result,
    ):
        git_commit = get_git_commit()

    assert git_commit is None


@pytest.mark.parametrize(
    "exception",
    [
        FileNotFoundError(),
        subprocess.CalledProcessError(
            returncode=128,
            cmd=["git", "rev-parse", "HEAD"],
        ),
        subprocess.TimeoutExpired(
            cmd=["git", "rev-parse", "HEAD"],
            timeout=5,
        ),
    ],
    ids=[
        "git-not-installed",
        "git-command-failed",
        "git-timeout",
    ],
)
def test_get_git_commit_failure(exception):
    with patch(
        "evaluation.benchmark_metadata.subprocess.run",
        side_effect=exception,
    ):
        git_commit = get_git_commit()

    assert git_commit is None


@pytest.mark.parametrize(
    "warmup_runs",
    [-1, -5, True, False, 1.5, "10", None],
)
def test_metadata_invalid_warmup_runs(warmup_runs):
    with pytest.raises(ValueError):
        collect_benchmark_metadata(
            device="cpu",
            warmup_runs=warmup_runs,
            benchmark_runs=100,
        )


@pytest.mark.parametrize(
    "benchmark_runs",
    [0, -1, -5, True, False, 1.5, "100", None],
)
def test_metadata_invalid_benchmark_runs(benchmark_runs):
    with pytest.raises(ValueError):
        collect_benchmark_metadata(
            device="cpu",
            warmup_runs=10,
            benchmark_runs=benchmark_runs,
        )


def test_metadata_zero_warmup_runs():
    metadata = collect_benchmark_metadata(
        device="cpu",
        warmup_runs=0,
        benchmark_runs=100,
    )

    assert metadata.warmup_runs == 0


def test_metadata_minimum_benchmark_runs():
    metadata = collect_benchmark_metadata(
        device="cpu",
        warmup_runs=10,
        benchmark_runs=1,
    )

    assert metadata.benchmark_runs == 1


def test_metadata_immutable():
    metadata = collect_benchmark_metadata(
        device="cpu",
        warmup_runs=10,
        benchmark_runs=100,
    )

    with pytest.raises(FrozenInstanceError):
        metadata.device = "mps"