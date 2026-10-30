from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
import time
import torch
import math

@dataclass(frozen=True)
class BenchmarkResult:
    """
    Stores raw measurements from an inference benchmark.

    Attributes:
        latencies_ms:
            Individual inference latencies in milliseconds.

        total_seconds:
            Total elapsed time of the measured benchmark phase.
    """

    latencies_ms: tuple[float, ...]
    total_seconds: float

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

# Calculate latency statistics
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

# Execute repeated inference and measure latency
def benchmark_inference(
    inference_fn: Callable[[], object],
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
    device: str | torch.device = "cpu",
) -> BenchmarkResult:
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
    latencies_ms: list[float] = []

    # Start overall benchmark measurement
    benchmark_start = time.perf_counter()

    # Step 4: Execute measured inference runs
    for _ in range(benchmark_runs):

        # Ensure previous deivce operations are complete
        synchronize_device(device)

        # Start high-resolution timer
        start_time = time.perf_counter()

        # Execute one inference operation
        inference_fn()

        # Ensure inference execution has completed
        synchronize_device(device)

        # Stop timer
        end_time = time.perf_counter()

        # Convert seconds to milliseconds
        latency_ms = (end_time - start_time) * 1000.0

        # Record the measurement
        latencies_ms.append(latency_ms)
    
    # Stop over all benchmark measurement
    benchmark_end = time.perf_counter()

    total_seconds = benchmark_end - benchmark_start
    
    return BenchmarkResult(
        latencies_ms=tuple(latencies_ms),
        total_seconds=total_seconds,
    )

# 
def calculate_throughput(
    num_inferences: int,
    total_seconds: float,
) -> float:
    """
    Calculate inference throughput in captions per second.

    Args:
        num_inferences:
            Number of completed inference operations.
        
        total_seconds:
            Total elapsed inference time in seconds.

    Returns:
        float:
            Throughput in captions per second.

    Raises:
        ValueError:
            If the input arguments are invalid.
    """

    # Step 1: Validate inference count
    if not isinstance(num_inferences, int) or isinstance(num_inferences, bool):
        raise ValueError(
            "num_inferences must be an integer."
        )
    
    if num_inferences <= 0:
        raise ValueError(
            "num_inferences must be positive."
        )
    
    # Step 2: Validate elapsed time
    if not isinstance(total_seconds, (int, float)) or isinstance(total_seconds, bool):
        raise ValueError(
            "total_seconds must be a real number."
        )
    
    if not math.isfinite(total_seconds):
        raise ValueError("total_seconds must be finite.")
    
    if total_seconds <= 0:
        raise ValueError(
            "total_seconds must be positive."
        )
    
    # Step 3: Calculate throughput
    throughput = num_inferences / total_seconds

    return throughput
