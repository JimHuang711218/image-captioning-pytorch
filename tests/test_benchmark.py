import pytest

from evaluation.benchmark import calculate_latency_metrics, benchmark_inference, synchronize_device, calculate_throughput


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

    result = benchmark_inference(
        inference_fn=mocked_inference,
        warmup_runs=3,
        benchmark_runs=5,
        device="cpu",
    )

    assert call_count == 8
    assert len(result.latencies_ms) == 5
    assert all(latency >= 0 for latency in result.latencies_ms)
    assert result.total_seconds >= 0

# Test 2: Verify zero warm-up runs 
def test_benchmark_inference_zero_warmup():

    call_count = 0

    def mocked_inference():
        nonlocal call_count
        call_count += 1

    result = benchmark_inference(
        inference_fn=mocked_inference,
        warmup_runs=0,
        benchmark_runs=5,
        device="cpu",
    )

    assert call_count == 5
    assert len(result.latencies_ms) == 5
    assert result.total_seconds >= 0

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

# Test 5: Verify mocked latency and total benchmark time
def test_benchmark_inference_mocked_timer(monkeypatch):

    timestamps = iter([
        0.000,  # Benchmark start
        1.000,  # Inference 1 start
        1.010,  # Inference 1 end
        2.000,  # Inference 2 start
        2.020,  # Inference 2 end
        3.000,  # Inference 3 start
        3.030,  # Inference 3 end
        4.000,  # Benchmark end
    ])

    monkeypatch.setattr(
        "evaluation.benchmark.time.perf_counter",
        lambda: next(timestamps),
    )

    result = benchmark_inference(
        inference_fn=lambda: None,
        warmup_runs=0,
        benchmark_runs=3,
        device="cpu",
    )

    # Verify individual inference latencies
    assert result.latencies_ms == pytest.approx((
        10.0,
        20.0,
        30.0,
    ))

    # Verify total benchmark execution time
    assert result.total_seconds == pytest.approx(4.0)

    # Verify throughput calculation
    throughput = calculate_throughput(
        num_inferences=len(result.latencies_ms),
        total_seconds=result.total_seconds,
    )

    assert throughput == pytest.approx(0.75)

# Test 6: Assumes the benchmarking engine calls perf_counter() only after warm-up completes.
def test_benchmark_excludes_warmup_from_timing(monkeypatch):

    call_count = 0

    def mocked_inference():
        nonlocal call_count
        call_count += 1

    timestamps = iter([
        10.0,  # Benchmark start
        10.1,  # Inference 1 start
        10.2,  # Inference 1 end
        10.3,  # Benchmark end
    ])

    monkeypatch.setattr(
        "evaluation.benchmark.time.perf_counter",
        lambda: next(timestamps),
    )

    result = benchmark_inference(
        inference_fn=mocked_inference,
        warmup_runs=3,
        benchmark_runs=1,
        device="cpu",
    )

    assert call_count == 4
    assert len(result.latencies_ms) == 1
    assert result.latencies_ms[0] == pytest.approx(100.0)
    assert result.total_seconds == pytest.approx(0.3)

# Test 7: Verify CPU synchronization
def test_synchronize_device_cpu():

    synchronize_device("cpu")

def test_calculate_throughput():

    throughput = calculate_throughput(
        num_inferences=100,
        total_seconds=20.0,
    )

    assert throughput == pytest.approx(5.0)

# Test 8: Verify that synchronization occurs before and after measured inference.
def test_benchmark_synchronization_order(monkeypatch):

    events = []

    def mocked_synchronize(device):
        events.append("synchronize")

    def mocked_inference():
        events.append("inference")

    monkeypatch.setattr(
        "evaluation.benchmark.synchronize_device",
        mocked_synchronize,
    )

    benchmark_inference(
        inference_fn=mocked_inference,
        warmup_runs=0,
        benchmark_runs=1,
        device="cpu",
    )

    assert events == [
        "synchronize",  # After warm-up
        "synchronize",  # Before measured inference
        "inference",
        "synchronize",  # After measured inference
    ]

    
'''------ Test Throughput ------'''
# Test 1: Normal throughput calculation
def test_calculate_throughput():

    throughput = calculate_throughput(
        num_inferences=100,
        total_seconds=20.0,
    )

    assert throughput == pytest.approx(5.0)


# Test 2: Invalid inference counts
@pytest.mark.parametrize(
    "num_inferences",
    [0, -1, -10, 1.5, True],
)
def test_calculate_throughput_invalid_count(num_inferences):

    with pytest.raises(ValueError):
        calculate_throughput(
            num_inferences=num_inferences,
            total_seconds=10.0,
        )


# Test 3: Invalid elapsed time
@pytest.mark.parametrize(
    "total_seconds",
    [0.0, -1.0, float("nan"), float("inf"), True],
)
def test_calculate_throughput_invalid_time(total_seconds):

    with pytest.raises(ValueError):
        calculate_throughput(
            num_inferences=100,
            total_seconds=total_seconds,
        )