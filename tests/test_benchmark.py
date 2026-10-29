import pytest

from evaluation.benchmark import calculate_latency_metrics, benchmark_inference, synchronize_device


def test_calculate_latency_metrics():

    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]

    metrics = calculate_latency_metrics(latencies)

    assert metrics.mean == pytest.approx(30.0)

    assert metrics.median == pytest.approx(30.0)

    assert metrics.minimum == pytest.approx(10.0)

    assert metrics.maximum == pytest.approx(50.0)

    assert metrics.p50 == pytest.approx(30.0)

    assert metrics.p90 == pytest.approx(46.0)

    assert metrics.p95 == pytest.approx(48.0)

    assert metrics.p99 == pytest.approx(49.6)


def test_calculate_latency_metrics_empty():

    with pytest.raises(ValueError):
        calculate_latency_metrics([])


def test_calculate_latency_metrics_negative():

    with pytest.raises(ValueError):
        calculate_latency_metrics([10.0, -5.0, 20.0])


def test_calculate_latency_metrics_non_finite():

    with pytest.raises(ValueError):
        calculate_latency_metrics([10.0, float("nan")])


def test_calculate_latency_metrics_single_measurement():

    metrics = calculate_latency_metrics([25.0])

    assert metrics.mean == pytest.approx(25.0)

    assert metrics.p95 == pytest.approx(25.0)

    assert metrics.p99 == pytest.approx(25.0)

'''--------- Benchmark Inference Test ----------'''

# Test 1: Verify inference execution counts
def test_benchmark_execution_counts():

    call_count = 0

    def mocked_inference():
        nonlocal call_count
        call_count += 1
        return "a dog running in the park"

    latencies = benchmark_inference(
        inference_fn=mocked_inference,
        warmup_runs=3,
        benchmark_runs=5,
        device="cpu",
    )

    assert call_count == 8
    assert len(latencies) == 5
    assert all(latency >= 0 for latency in latencies)

# Test 2: Verify zero warm-up runs 
def test_benchmark_inference_zero_warmup():

    call_count = 0

    def mocked_inference():
        nonlocal call_count
        call_count += 1

    latencies = benchmark_inference(
        inference_fn=mocked_inference,
        warmup_runs=0,
        benchmark_runs=5,
        device="cpu",
    )

    assert call_count == 5
    assert len(latencies) == 5

# Test 3: Verify invalid warm-up configuration
@pytest.mark.parametrize(
    "warmup_runs",
    [-1, -5, 1.5, True],
)
def test_benchmark_inference_invalid_warmup(warmup_runs):

    with pytest.raises(ValueError):

        benchmark_inference(
            inference_fn=lambda: None,
            warmup_runs=warmup_runs,
            benchmark_runs=10,
        )

# Test 4: Verify invalid benchmark runs
@pytest.mark.parametrize(
    "benchmark_runs",
    [0, -1, -10, 1.5, True],
)
def test_benchmark_inference_invalid_runs(benchmark_runs):

    with pytest.raises(ValueError):

        benchmark_inference(
            inference_fn=lambda: None,
            warmup_runs=5,
            benchmark_runs=benchmark_runs,
        )

# Test 5: Verify mocked latency measurement
def test_benchmark_inference_mocked_timer(monkeypatch):

    timestamps = iter([
        1.000,
        1.010,
        2.000,
        2.020,
        3.000,
        3.030,
    ])

    monkeypatch.setattr(
        "evaluation.benchmark.time.perf_counter",
        lambda: next(timestamps),
    )

    latencies = benchmark_inference(
        inference_fn=lambda: None,
        warmup_runs=0,
        benchmark_runs=3,
        device="cpu",
    )

    assert latencies == pytest.approx([
        10.0,
        20.0,
        30.0,
    ])

# Test 6: Verify CPU synchronization
def test_synchronize_device_cpu():

    synchronize_device("cpu")