from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyMetrics:
    """
    Stores inference latency statistics in milliseconds.
    """

    mean: float
    median: float
    minimum: float
    maximum: float
    p50: float
    p90: float
    p95: float
    p99: float


def calculate_latency_metrics(
    latencies_ms: list[float],
) -> LatencyMetrics:
    """
    Calculate inference latency statistics.

    Args:
        latencies_ms:
            Measured inference latencies in milliseconds.

    Returns:
        LatencyMetrics:
            An immutable collection of latency statistics.

    Raises:
        ValueError:
            If the input is empty or contains invalid latency values.
    """

    if not latencies_ms:
        raise ValueError(
            "Latency measurements must not be empty."
        )

    values = np.asarray(latencies_ms, dtype=np.float64)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Latency measurements must be finite."
        )

    if np.any(values < 0):
        raise ValueError(
            "Latency measurements must be non-negative."
        )

    return LatencyMetrics(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        p50=float(np.percentile(values, 50)),
        p90=float(np.percentile(values, 90)),
        p95=float(np.percentile(values, 95)),
        p99=float(np.percentile(values, 99)),
    )