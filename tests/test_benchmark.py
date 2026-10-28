import pytest

from evaluation.benchmark import calculate_latency_metrics


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