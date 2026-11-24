from pathlib import Path
from unittest.mock import patch
import json
import pytest

from evaluation.benchmark_cli import (
    parse_args,
    main,
    run_benchmark,
    validate_artifacts,
)
from evaluation.benchmark import (
    BenchmarkResult,
    validate_device,
)
from evaluation.benchmark_serialization import save_benchmark_result

def test_parse_args_defaults():

    args = parse_args([])

    assert args.device == "cpu"
    assert args.warmup_runs == 10
    assert args.benchmark_runs == 100
    assert args.output is None

@pytest.mark.parametrize(
    "warmup_runs",
    [-1, -5, -10],
)
def test_parse_args_invalid_warmup(warmup_runs):

    with pytest.raises(SystemExit) as exc_info:
        parse_args([
            "--warmup-runs",
            str(warmup_runs),
        ])

    assert exc_info.value.code == 2

@pytest.mark.parametrize(
    "benchmark_runs",
    [1, 10, 50, 100],
)
def test_parse_args_valid_benchmark_runs(benchmark_runs):

    args = parse_args([
        "--benchmark-runs",
        str(benchmark_runs)
    ])

    assert args.benchmark_runs == benchmark_runs

@pytest.mark.parametrize(
    "benchmark_runs",
    [0, -1, -5, -10],
)
def test_parse_args_invalid_benchmark_runs(benchmark_runs):

    with pytest.raises(SystemExit) as exc_info:
        parse_args([
            "--benchmark-runs",
            str(benchmark_runs),
        ])

    assert exc_info.value.code == 2

@pytest.mark.parametrize(
    "device",
    ["cpu", "cuda", "mps"],
)
def test_parse_args_valid_device(device):

    args = parse_args([
        "--device",
        device
    ])

    assert args.device == device

@pytest.mark.parametrize(
    "device",
    ["invalid", "gpu", "tpu"],
)
def test_parse_args_invalid_device(device, capsys):

    with pytest.raises(SystemExit) as exc_info:
        parse_args([
            "--device",
            device,
        ])

    assert exc_info.value.code == 2

    captured = capsys.readouterr()

    assert "invalid choice" in captured.err
    assert device in captured.err

'''__Test Output__'''
@pytest.mark.parametrize(
    "output_path",
    [
        "results/benchmark.json",
        "evaluation/results/cpu_benchmark.json",
        "custom_output.json",
    ],
)
def test_parse_args_valid_output(output_path):

    args = parse_args([
        "--output",
        output_path,
    ])

    assert args.output == Path(output_path)

def test_parse_args_missing_output():

    with pytest.raises(SystemExit) as exc_info:
        parse_args([
            "--output",
        ])

    assert exc_info.value.code == 2

def test_parse_args_help(capsys):

    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--help"])

    assert exc_info.value.code == 0

    captured = capsys.readouterr()

    assert "--device" in captured.out
    assert "--warmup-runs" in captured.out
    assert "--benchmark-runs" in captured.out
    assert "--output" in captured.out
def test_main_valid_configuration(tmp_path, capsys):

    # Step 1: Create a temporary output path
    output_path = tmp_path / "results" / "test.json"

    # Step 2: Define a lightweight inference function
    def fake_inference():
        return "A dog running in a park."

    # Step 3: Execute the CLI
    exit_code = main(
        [
            "--device",
            "cpu",
            "--warmup-runs",
            "5",
            "--benchmark-runs",
            "20",
            "--output",
            str(output_path),
        ],
        inference_fn=fake_inference,
    )

    # Step 4: Verify successful execution
    assert exit_code == 0

    # Step 5: Verify output file creation
    assert output_path.exists()

    # Step 6: Verify CLI output
    captured = capsys.readouterr()

    assert captured.out.strip() == (
    f"Benchmark results saved to: {output_path}"
    )


def test_run_benchmark_configuration():

    def fake_inference():
        return "A dog running in a park."

    fake_result = BenchmarkResult(
        latencies_ms=(10.0, 20.0, 30.0),
        total_seconds=0.09,
    )

    with patch(
        "evaluation.benchmark_cli.benchmark_inference"
    ) as mock_benchmark:

        mock_benchmark.return_value = fake_result

        results = run_benchmark(
            inference_fn=fake_inference,
            warmup_runs=5,
            benchmark_runs=3,
            device="cpu",
        )

        mock_benchmark.assert_called_once_with(
            inference_fn=fake_inference,
            warmup_runs=5,
            benchmark_runs=3,
            device="cpu",
        )

    assert results["device"] == "cpu"
    assert results["warmup_runs"] == 5
    assert results["benchmark_runs"] == 3

    assert results["total_seconds"] == pytest.approx(0.09)

    assert results["latencies_ms"]["mean"] == pytest.approx(20.0)

    assert results["throughput_per_second"] == pytest.approx(
        3 / 0.09
    )

def test_run_benchmark_real_execution():

    call_count = 0

    def fake_inference():
        nonlocal call_count
        call_count += 1
        return "A dog running in a park."

    results = run_benchmark(
        inference_fn=fake_inference,
        warmup_runs=2,
        benchmark_runs=5,
        device="cpu",
    )

    assert call_count == 7
    assert results["benchmark_runs"] == 5
    assert results["latencies_ms"]["mean"] >= 0
    assert results["throughput_per_second"] > 0

def test_save_benchmark_results(tmp_path):

    output_path = tmp_path / "results" / "benchmark.json"

    results = {
        "device": "cpu",
        "warmup_runs": 5,
        "benchmark_runs": 3,
        "total_seconds": 0.09,
        "latencies_ms": {
            "mean": 20.0,
            "median": 20.0
        },
        "throughput_per_second": 33.333,
    }

    save_benchmark_result(
        serialized=results,
        output_path=str(output_path)
    )

    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        save_results = json.load(file)

    assert save_results == results

def test_save_benchmark_results_invalid_nan(tmp_path):

    output_path = tmp_path / "benchmark.json"

    results = {
        "device": "cpu",
        "throughput_per_second": float("nan"),
    }

    with pytest.raises(ValueError):
        save_benchmark_result(
            serialized=results,
            output_path=str(output_path),
        )

def test_main_full_workflow(tmp_path):

    # Step 1: Define the output path.
    output_path = tmp_path / "benchmark.json"

    # Step 2: Track inference execution count.
    call_count = 0

    def fake_inference():
        nonlocal call_count
        call_count += 1
        return "A dog running in a park."

    # Step 3: Execute the complete CLI workflow.
    exit_code = main(
        [
            "--device",
            "cpu",
            "--warmup-runs",
            "2",
            "--benchmark-runs",
            "5",
            "--output",
            str(output_path),
        ],
        inference_fn=fake_inference,
    )

    # Step 4: Validate successful execution.
    assert exit_code == 0

    # 2 warmup runs + 5 measured runs.
    assert call_count == 7

    # Step 5: Verify that the JSON file exists.
    assert output_path.is_file()

    # Step 6: Load the persisted JSON.
    with output_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    # Step 7: Validate the serialized structure.
    assert "metadata" in results
    assert "raw_measurements" in results

    # Step 8: Validate benchmark metadata.
    metadata = results["metadata"]

    assert metadata["device"] == "cpu"
    assert metadata["warmup_runs"] == 2
    assert metadata["benchmark_runs"] == 5

    # Step 9: Validate raw benchmark measurements.
    raw_measurements = results["raw_measurements"]

    latencies = raw_measurements["latencies_ms"]

    assert isinstance(latencies, list)
    assert len(latencies) == 5

    assert all(
        isinstance(latency, (int, float)) and latency >= 0
        for latency in latencies
    )

    assert raw_measurements["total_seconds"] > 0

'''
def test_main_missing_inference():

    with pytest.raises(ValueError, match="inference function"):
        main([
            "--device",
            "cpu",
            "--benchmark-runs",
            "5",
        ])
'''

def test_validate_artifacts_success(tmp_path):
    """
    Verify that artifact validation succeeds
    when all required files exist.
    """

    vocab_path = tmp_path / "vocab.pkl"
    encoder_checkpoint = tmp_path / "encoder.pt"
    decoder_checkpoint = tmp_path / "decoder.pt"

    for path in (
        vocab_path,
        encoder_checkpoint,
        decoder_checkpoint,
    ):
        # Create an empty file in the given path
        path.touch()

    validate_artifacts(
        vocab_path=vocab_path,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
    )

def test_validate_artifacts_missing_vocab(tmp_path):
    """
    Verify that missing vocabulary raises FileNotFoundError
    """

    vocab_path = tmp_path / "vocab.pkl"
    encoder_checkpoint = tmp_path / "encoder.pt"
    decoder_checkpoint = tmp_path / "decoder.pt"

    encoder_checkpoint.touch()
    decoder_checkpoint.touch()

    with pytest.raises(
        FileNotFoundError,
        match="Required inference artifact not found."
    ):
        validate_artifacts(
            vocab_path=vocab_path,
            encoder_checkpoint=encoder_checkpoint,
            decoder_checkpoint=decoder_checkpoint,
        )


import pytest

@pytest.mark.parametrize(
    "missing_artifact",
    [
        "vocab",
        "encoder",
        "decoder",
    ],
)
def test_validate_artifacts_missing_file(
    tmp_path,
    missing_artifact,
):
    """
    Verify that validate_artifacts raises FileNotFoundError
    when any required inference artifact is missing.

    Test cases:
        1. Missing vocabulary file.
        2. Missing encoder checkpoint.
        3. Missing decoder checkpoint.
    """

    # Step 1: Define paths for all required artifacts.
    artifacts = {
        "vocab": tmp_path / "vocab.pkl",
        "encoder": tmp_path / "encoder.pth",
        "decoder": tmp_path / "decoder.pth",
    }

    # Step 2: Create the selected artifact as a directory.
    # Create the remaining artifacts as regular files.
    for name, path in artifacts.items():
        if name == missing_artifact:
            path.mkdir()
        else:
            path.touch()

    # Step 3: Verify the missing artifact raises FileNotFoundError.
    with pytest.raises(
        FileNotFoundError,
        match="Required inference artifact not found",
    ) as exc_info:
        validate_artifacts(
            vocab_path=artifacts["vocab"],
            encoder_checkpoint=artifacts["encoder"],
            decoder_checkpoint=artifacts["decoder"],
        )

    # Step 4: Verify the error identifies the correct missing file.
    assert str(artifacts[missing_artifact]) in str(
        exc_info.value
    )

def test_main_initializes_real_inference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """
    Verify that main() initializes the inference service
    when no inference callable is injected.
    """

    import evaluation.benchmark_cli as cli

    calls = []

    class FakeInferenceService:
        def predict(self, image_path: Path) -> str:
            calls.append(image_path)
            return "A test caption."

    def fake_create_benchmark_inference(
        device: str,
    ) -> FakeInferenceService:
        assert device == "cpu"
        return FakeInferenceService()

    monkeypatch.setattr(
        cli,
        "create_benchmark_inference",
        fake_create_benchmark_inference,
    )

    monkeypatch.setattr(
        cli,
        "PROJECT_ROOT",
        tmp_path,
    )

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    (image_dir / "coco-examples.jpg").touch()

    exit_code = cli.main(
        argv=[
            "--device",
            "cpu",
            "--warmup-runs",
            "1",
            "--benchmark-runs",
            "3",
        ],
    )

    assert exit_code == 0

    assert len(calls) == 4

    assert all(
        path == image_dir / "coco-examples.jpg"
        for path in calls
    )
