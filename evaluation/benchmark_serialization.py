from dataclasses import asdict
from pathlib import Path

import json
import os
import tempfile
import numpy as np

from evaluation.benchmark import BenchmarkResult
from evaluation.benchmark_metadata import BenchmarkMetadata


def serialize_benchmark_result(
    result: BenchmarkResult,
    metadata: BenchmarkMetadata,
) -> dict[str, object]:
    """
    Serialize raw benchmark measurements and metadata.

    Args:
        result:
            Raw inference benchmark measurements.

        metadata:
            Benchmark execution environment information.

    Returns:
        dict[str, object]:
            JSON-compatible benchmark representation.
    """

    if not result.latencies_ms:
        raise ValueError(
            "Latency measurements must not be empty."
        )

    values = np.asarray(result.latencies_ms, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Latency measurements must be finite."
        )

    if np.any(values < 0) :
        raise ValueError(
            "Latency measurements must be non-negative."
        )

    if not result.total_seconds:
        raise ValueError(
            "Total seconds must not empty."
        )

    if not np.isfinite(result.total_seconds):
        raise ValueError(
            "Total benchmark duration must be finite."
        )

    if result.total_seconds <= 0:
        raise ValueError(
            "Total benchmark duration must be non-negative."
        )

    # Step 1: Convert BenchmarkResult to dictionary.
    raw_measurements = asdict(result)

    # Step 2: Convert BenchmarkMetadata to dictionary.
    metadata_dict = asdict(metadata)

    # Step 3: Convert latency_ms from tuple to list
    raw_measurements["latencies_ms"] = list(raw_measurements["latencies_ms"])

    return {
        "raw_measurements": raw_measurements,
        "metadata": metadata_dict
    }

def save_benchmark_result(
    serialized: dict[str, object],
    output_path: str | Path,
) -> None:
    """
    Persist serialized benchmark results as JSON.

    Args:
        serialized:
            JSON-compatible benchmark data.

        output_path:
            Destination file path.

    Raises:
        TypeError:
            If the data contains unsupported JSON type.

        ValueError:
            If the data contains non-finite numerical values.

        OSError:
            If writing the file fails.
    """

    # Step 1: Normalize the output path
    path = Path(output_path)

    # Step 2: Serialize data and reject non-finite values.
    json_text = json.dumps(
        serialized,
        indent=2,
        allow_nan=False,
    )

    # Step 3: Ensure the destination directories exists.
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = None

    try:
        # Step 4: Create a temporary file in the destination directory.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f"{path.name}",
            suffix=".tmp",
            delete=False,
        ) as temp_file:

            temp_path = Path(temp_file.name)

            # Step 5: Write JSON into the temporary file.
            temp_file.write(json_text)
            temp_file.write("\n")

        # Step 6: Atomically replace the destination file.
        os.replace(
            temp_path,
            path,
        )

    finally:
          # Step 7: Clean up the temporary file if it still exists
          if temp_path is not None:
              temp_path.unlink(missing_ok=True)
