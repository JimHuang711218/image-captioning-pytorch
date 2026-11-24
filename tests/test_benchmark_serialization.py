import json
import os

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from evaluation.benchmark import BenchmarkResult
from evaluation.benchmark_metadata import BenchmarkMetadata
from evaluation.benchmark_serialization import (
    save_benchmark_result,
    serialize_benchmark_result,
)

@pytest.fixture
def benchmark_metadata() -> BenchmarkMetadata:
    return BenchmarkMetadata(
        timestamp_utc="2026-11-10T08:00:00+00:00",
        python_version="3.12.8",
        pytorch_version="2.x",
        platform="Darwin",
        machine="arm64",
        processor="Apple Silicon",
        device="cpu",
        warmup_runs=10,
        benchmark_runs=100,
        git_commit="abc123",
    )

def test_serialize_benchmark_result(
    benchmark_metadata: BenchmarkMetadata,
):

    # Arrange
    result = BenchmarkResult(
        latencies_ms=(20.1, 21.3, 19.8),
        total_seconds=0.0612,
    )

    # Act
    serialized = serialize_benchmark_result(
        result=result,
        metadata=benchmark_metadata,
    )

    # Assert: dictionary structure
    assert "metadata" in serialized
    assert "raw_measurements" in serialized

    # Assert: latency serialization
    assert serialized["raw_measurements"]["latencies_ms"] == [
        20.1,
        21.3,
        19.8,
    ]

    # Assert: elapsed time
    assert serialized["raw_measurements"]["total_seconds"] == 0.0612

    # Assert: metadata preservation
    assert serialized["metadata"] == asdict(
        benchmark_metadata
    )

    # Assert: strict JSON compatibility
    json_string = json.dumps(
        serialized,
        allow_nan=False,
    )

    assert isinstance(json_string, str)

    # Assert: input immutability
    assert result.latencies_ms == (20.1, 21.3, 19.8)

# Test Empty Latencies
def test_serialize_empty_latencies(
    benchmark_metadata: BenchmarkMetadata,
):
    result = BenchmarkResult(
        latencies_ms=(),
        total_seconds=0.0,
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        serialize_benchmark_result(
            result=result,
            metadata=benchmark_metadata,
        )

@pytest.mark.parametrize(
    "invalid_latency",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_serialize_non_finite_latencies(
    benchmark_metadata: BenchmarkMetadata,
    invalid_latency: float,
):
    result = BenchmarkResult(
        latencies_ms=(20.1, invalid_latency),
        total_seconds=0.04,
    )

    with pytest.raises(ValueError):
        serialize_benchmark_result(
            result=result,
            metadata=benchmark_metadata,
        )

@pytest.mark.parametrize(
    "invalid_total_seconds",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        -1.0,
        0.0,
    ],
)
def test_serialize_invalid_total_seconds(
    benchmark_metadata: BenchmarkMetadata,
    invalid_total_seconds: float,
):
    result = BenchmarkResult(
        latencies_ms=(20.1, 21.3),
        total_seconds=invalid_total_seconds,
    )

    with pytest.raises(ValueError):
        serialize_benchmark_result(
            result=result,
            metadata=benchmark_metadata,
        )

# Test git commit
def test_serialize_git_commit_none(
    benchmark_metadata: BenchmarkMetadata,
):
    metadata = replace(
        benchmark_metadata,
        git_commit=None,
    )

    result = BenchmarkResult(
        latencies_ms=(20.1, 21.3),
        total_seconds=0.0414,
    )

    serialized = serialize_benchmark_result(
        result=result,
        metadata=metadata,
    )

    assert serialized["metadata"]["git_commit"] is None

    json_string = json.dumps(
        serialized,
        allow_nan=False,
    )

    restored = json.loads(json_string)

    assert restored["metadata"]["git_commit"] is None

# Immutability Test
def test_serialize_input_unchanged(
    benchmark_metadata: BenchmarkMetadata,
):
    result = BenchmarkResult(
        latencies_ms=(20.1, 21.3, 19.8),
        total_seconds=0.0612,
    )

    original_result = asdict(result)
    original_metadata = asdict(benchmark_metadata)

    serialize_benchmark_result(
        result=result,
        metadata=benchmark_metadata,
    )

    assert asdict(result) == original_result
    assert asdict(benchmark_metadata) == original_metadata

@pytest.mark.parametrize(
    "invalid_latency",
    [
        -1.0,
        -0.001,
    ],
)
def test_serialize_negative_latencies(
    benchmark_metadata: BenchmarkMetadata,
    invalid_latency: float,
):
    result = BenchmarkResult(
        latencies_ms=(20.1, invalid_latency),
        total_seconds=0.04,
    )

    with pytest.raises(ValueError):
        serialize_benchmark_result(
            result=result,
            metadata=benchmark_metadata,
        )

# Successful JSON persistence
def test_save_benchmark_result(
    tmp_path: Path,
) -> None:

    serialized = {
        "raw_measurements": {
            "latencies_ms": [20.1, 21.3, 19.8],
            "total_seconds": 0.0612,
        },
        "metadata": {
            "device": "cpu",
        },
    }

    output_path = tmp_path / "benchmark.json"

    save_benchmark_result(
        serialized=serialized,
        output_path=output_path,
    )

    assert output_path.exists()

    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved == serialized

# Reject non-finite values
@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_save_benchmark_result_non_finite(
    tmp_path: Path,
    invalid_value: float,
) -> None:

    serialized = {
        "latency_ms": invalid_value,
    }

    output_path = tmp_path / "benchmark.json"

    with pytest.raises(ValueError):
        save_benchmark_result(
            serialized,
            output_path,
        )

    assert not output_path.exists()

# Atomatic replacement failiure
def test_save_benchmark_result_atomic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    output_path = tmp_path / "benchmark.json"

    original_content = '{"status": "original"}'

    output_path.write_text(
        original_content,
        encoding="utf-8",
    )

    serialized = {
        "status": "updated",
    }

    def mock_replace(
        src: str | os.PathLike[str],
        dst: str | os.PathLike[str],
    ) -> None:

        source_path = Path(src)
        destination_path = Path(dst)

        assert source_path.parent == tmp_path
        assert destination_path == output_path
        assert source_path.exists()

        raise OSError("Simulated replacement failure")

    # Apply the patch outside mock_replace().
    monkeypatch.setattr(
        os,
        "replace",
        mock_replace,
    )

    with pytest.raises(
        OSError,
        match="Simulated replacement failure",
    ):
        save_benchmark_result(
            serialized,
            output_path,
        )

    # Verify the original destination remains unchanged.
    assert output_path.read_text(
        encoding="utf-8"
    ) == original_content

    # Verify temporary-file cleanup.
    assert list(tmp_path.glob("*.tmp")) == []

# Overwrite an existing benchmark result
def test_save_benchmark_result_overwrite(
    tmp_path: Path,
) -> None:

    # Arrange
    output_path = tmp_path / "benchmark.json"

    original = {
        "device": "cpu",
        "latency_ms": 20.1,
    }

    updated = {
        "device": "mps",
        "latency_ms": 15.8,
    }

    # Act 1: Save original benchmark data.
    save_benchmark_result(
        original,
        output_path,
    )

    # Act 2: Overwrite with updated benchmark data.
    save_benchmark_result(
        updated,
        output_path,
    )

    # Assert: Read the updated JSON file.
    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved == updated

    # Assert: No temporary files remain.
    assert list(tmp_path.glob("*.tmp")) == []

# Create missing parent directories
def test_save_benchmark_result_nested_directory(
    tmp_path: Path,
) -> None:

    output_path = (
        tmp_path
        / "results"
        / "benchmarks"
        / "cpu"
        / "benchmark.json"
    )

    serialized = {
        "device": "cpu",
        "latency_ms": 20.1,
    }

    save_benchmark_result(
        serialized,
        output_path,
    )

    assert output_path.exists()

    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved == serialized
