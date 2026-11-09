from pathlib import Path

import pytest
import torch
import math

from evaluation.benchmark import(
    benchmark_inference,
    calculate_latency_metrics,
    calculate_throughput,
)
from evaluation.benchmark_cli import (
    create_benchmark_inference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.model
def test_create_benchmark_inference_real_model():

    # Step 1: Select execution device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Step 2: Initialize inference service
    inference_service = create_benchmark_inference(
        device=device
    )

    # Step 3: Verify model device
    assert (
        next(inference_service.encoder.parameters()).device.type
        == device
    )

    assert (
        next(inference_service.decoder.parameters()).device.type
        == device
    )

    # Step 4: Verify evaluation mode
    assert inference_service.encoder.training is False

    assert inference_service.decoder.training is False

    # Step 5: Run real inference
    image_path = (
        PROJECT_ROOT / "images" / "coco-examples.jpg"
    )

    caption = inference_service.predict(
        image_path
    )

    # Step 6: Validate output
    assert isinstance(caption, str)

    assert len(caption) > 0

@pytest.mark.model
def test_real_pytorch_inference_benchmark():

    # Step 1: Select execution device
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Step 2: Initialize the model
    inference_service = create_benchmark_inference(
        device=device
    )

    # Step 3: Prepare the image path
    image_path = (
        PROJECT_ROOT
        / "images"
        / "coco-examples.jpg"
    )

    # Step 4: Define the inference callable
    def inference_fn() -> str:
        return inference_service.predict(image_path)
    
    # Step 5: Execute the benchmark
    result = benchmark_inference(
        inference_fn=inference_fn,
        warmup_runs=2,
        benchmark_runs=5,
        device=device
    )

    # Step 6: Validate benchmark results
    assert len(result.latencies_ms) == 5
    assert all(
        latency > 0
        for latency in result.latencies_ms
    )
    assert result.total_seconds > 0

    # Step 7: Calculate performance metrics
    latency_metrics = calculate_latency_metrics(
        list(result.latencies_ms)
    )

    throughput = calculate_throughput(
        num_inferences=len(result.latencies_ms),
        total_seconds=result.total_seconds,
    )

    # Step 8: Display performance results
    print(f"\nDevice: {device}")

    print(f"Mean latency: {latency_metrics.mean: .2f} ms")
    print(f"P50 latency: {latency_metrics.p50: .2f} ms")
    print(f"P90 latency: {latency_metrics.p90: .2f} ms")
    print(f"P95 latency: {latency_metrics.p95: .2f} ms")
    print(f"P99 latency: {latency_metrics.p99: .2f} ms")

    print(f"Throughput: {throughput: .2f} captions/sec")

@pytest.mark.model
@pytest.mark.parametrize(
    "device",
    ["mps", "cpu"],
)
def test_real_inference_cpu_mps(device):

    # Step 1: Check device availability
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    
    # Step 2: Create a real inference callable
    inference_service = create_benchmark_inference(
        device=device
    )

    image_path = (
        PROJECT_ROOT
        / "images"
        / "coco-examples.jpg"
    )

    def inference_fn():
        return inference_service.predict(image_path)

    # Step 3: Execute benchmark_inference()
    result = benchmark_inference(
        inference_fn=inference_fn,
        warmup_runs=10,
        benchmark_runs=100,
        device=device,
    )

    # Step 4: Calculate latency metrics
    latency_metrics = calculate_latency_metrics(
        result.latencies_ms
    )

    # Step 5: Calculate throughput
    throughput = calculate_throughput(
        num_inferences=len(result.latencies_ms),
        total_seconds=result.total_seconds,
    )

    # Step 6: Validate performance results

    # Validate the number of benchmark iterations.
    assert len(result.latencies_ms) == 100

    # Validate individual latency measurements.
    assert all(
        math.isfinite(latency) and latency >= 0
        for latency in result.latencies_ms
    )

    # Validate total execution time.
    assert math.isfinite(result.total_seconds)
    assert result.total_seconds > 0

    # Validate latency metrics.
    assert all(
        math.isfinite(value) and value >= 0
        for value in (
            latency_metrics.mean,
            latency_metrics.p50,
            latency_metrics.p90,
            latency_metrics.p95,
            latency_metrics.p99,
        )
    )

    # Validate percentile ordering.
    assert (
        latency_metrics.p50
        <= latency_metrics.p90
        <= latency_metrics.p95
        <= latency_metrics.p99
    )

    # Validate throughput.
    assert math.isfinite(throughput)
    assert throughput > 0

    print(f"\n{'=' * 40}")
    print(f"Device: {device.upper()}")
    print(f"{'=' * 40}")

    print(
        f"Mean latency: {latency_metrics.mean:.2f} ms"
    )
    print(
        f"P50 latency: {latency_metrics.p50:.2f} ms"
    )
    print(
        f"P90 latency: {latency_metrics.p90:.2f} ms"
    )
    print(
        f"P95 latency: {latency_metrics.p95:.2f} ms"
    )
    print(
        f"P99 latency: {latency_metrics.p99:.2f} ms"
    )

    print(
        f"Throughput: {throughput:.2f} captions/sec"
    )

@pytest.mark.model
@pytest.mark.parametrize(
    "device_order",
    [
        ("cpu", "mps"),
        ("mps", "cpu"),
        ("mps", "cpu"),
        ("cpu", "mps"),
    ],
    ids=[
        "session1-cpu-first",
        "session2-mps-first",
        "session3-mps-first",
        "session4-cpu-first",
    ]
)
def test_benchmark_execution_order(
    device_order: tuple[str, str]
) -> None:

    print(
        f"\nExecution order: "
        f"{device_order[0].upper()} -> {device_order[1].upper()}"
    )

    if not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")

    image_path = (
            PROJECT_ROOT
            / "images"
            / "coco-examples.jpg"
    )

    assert image_path.is_file()

    results = {}

    for device in device_order:

        # Step 1: Create real inference function
        inference_service = create_benchmark_inference(
            device=device
        )

        def inference_fn():
            return inference_service.predict(image_path)
        
        # Step 2: Execute benchmark inference
        result = benchmark_inference(
            inference_fn=inference_fn,
            warmup_runs=10,
            benchmark_runs=100,
            device=device,
        )

        results[device] = result

    # Validate results

    # Both devices produced results
    assert set(results) == {
        "cpu",
        "mps",
    }

    for device, result in results.items():

        assert len(result.latencies_ms) == 100

        # Calcuate latency
        latency_metrics = calculate_latency_metrics(
            result.latencies_ms
        )

        # Calculate throughput
        throughput = calculate_throughput(
            num_inferences=len(result.latencies_ms),
            total_seconds=result.total_seconds,
        )

        # Validate latency
        assert math.isfinite(latency_metrics.mean)
        assert latency_metrics.mean > 0

        # Validate throughput
        assert math.isfinite(throughput)
        assert throughput > 0

        # Display results
        print(f"\nDevice: {device.upper()}")
        print(f"Mean latency: {latency_metrics.mean: .2f} ms")
        print(f"P99 latency: {latency_metrics.p99: .2f} ms")
        print(f"Throughput: {throughput: .2f} captions/sec")
