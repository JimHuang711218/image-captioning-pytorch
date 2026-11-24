from dataclasses import dataclass
from datetime import datetime, timezone

import platform
import torch
import subprocess

@dataclass(frozen=True)
class BenchmarkMetadata:
    timestamp_utc: str

    python_version: str
    pytorch_version: str
    platform: str
    machine: str
    processor: str

    device: str

    warmup_runs: int
    benchmark_runs: int

    git_commit: str| None

def get_git_commit() -> str | None:
    """
    Return the current Git commit hash.

    Returns None when Git information is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )

        git_commit = result.stdout.strip() or None

    except(
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):

        git_commit = None

    return git_commit


def collect_benchmark_metadata(
    device: str,
    warmup_runs: int,
    benchmark_runs: int,
) -> BenchmarkMetadata:
    """
    Collect execution environment metadata.

    Args:
        device:
            selected inference device.

        warmup_runs:
            Number of warm-up iterations.

        benchmark_runs:
            Number of measured iterations.

    Returns:
        BenchmarkMetadata:
            Immutabel environment metadata.
    """
    # Validate benchmark configuration
    if not isinstance(warmup_runs, int) or isinstance(warmup_runs, bool):
        raise ValueError(
            "warmup_runs must be an integer."
        )

    if not isinstance(benchmark_runs, int) or isinstance(benchmark_runs, bool):
        raise ValueError(
            "benchmark_runs must be an integer."
        )

    if warmup_runs < 0:
        raise ValueError(
            "warmup_runs must be non-negative."
        )

    if benchmark_runs <= 0:
        raise ValueError(
            "benchmark_runs must be positive."
        )

    timestamp_utc = datetime.now(timezone.utc).isoformat()

    python_version = platform.python_version()
    pytorch_version = torch.__version__
    os_platform = platform.system()
    machine = platform.machine()
    processor = platform.processor()

    git_commit = get_git_commit()

    metadata = BenchmarkMetadata(
        timestamp_utc=timestamp_utc,
        python_version=python_version,
        pytorch_version=pytorch_version,
        platform=os_platform,
        machine=machine,
        processor=processor,
        device=device,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        git_commit=git_commit,
    )

    return metadata
