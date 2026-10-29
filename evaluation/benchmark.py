from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import time
import torch


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

# Implement device synchronization
def synchronize_device(
    device: str | torch.device
) -> None:
    """
    Synchronize the execution device before or after 
    measuring inference latency.

    CPU operations are generally synchronous.

    CUDA and MPS operations may be asynchronous,
    requiring explicit synchronization to ensure
    accurate latency measurements.

    Args:
        device:
            Execution device: cpu, cuda, or mps.
    
    Raises:
        ValueError:
            If the device type is unsupported.
    """

    device = torch.device(device)

    if device.type == "cpu":
        return
    
    if device.type == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA synchronization requested, "
                "but CUDA is not available."
            )

        torch.cuda.synchronize(device)
        return

    if device.type == "mps":

        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS synchronization requested, "
                "but MPS is not available."
            )

        torch.mps.synchronize()
        return
    
    raise ValueError(
        f"Unsupported benchmarking device: {device}"
    )

def benchmark_inference(
    inference_fn: Callable[[], object],
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    device: str | torch.device = "cpu",
) -> list[float]:
    """
    Benchmark inference latency using repeated execution.
    
    The function performs configurable warm-up runs
    before collecting latency measurements.

    Each benchmark run measures the execution time of
    one inference call.
    
    Args:
        inference_fn:
            A callable that performs one inference operations.
        
        warmup_runs:
            Number of unmeasured warm-up runs.

        benchmark_runs:
            Number of measured inference execution.

        device:
            Execution device: cpu, cuda, or mps.
    
    Returns:
        list[float]:
            individual inference latencies in milliseconds.
        
    Raises:
        ValueError:
            If warmup_runs is negative or benchmark_runs 
            is not positive
    """

    # Srep 1: Validate benchmark configuration
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
    
    device = torch.device(device)

    # Step 2: Perform warm-up executions
    for _ in range(warmup_runs):
        inference_fn()
    
    # Ensure all warm-up operations have completed
    synchronize_device(device)

    # Step 3: Initialize latency measurements
    latencies_ms = []

    # Step 4: Execute measured inference runs
    for _ in range(benchmark_runs):

        # Ensure previous deivce operations are complete
        synchronize_device(device)

        # Start high-resolution timer
        start_time = time.perf_counter()

        # Execute one inference operation
        inference_fn()

        # Stop timer
        end_time = time.perf_counter()

        # Convert seconds to milliseconds
        latency_ms = (end_time - start_time) * 1000.0

        # Record the measurement
        latencies_ms.append(latency_ms)
    
    return latencies_ms