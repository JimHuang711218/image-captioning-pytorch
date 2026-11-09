import argparse
import json
import pickle
import torch

from pathlib import Path
from dataclasses import asdict
from collections.abc import Callable
from evaluation.benchmark import benchmark_inference, calculate_latency_metrics, calculate_throughput
from src.config import ModelConfig
from src.inference import ImageCaptioningInference

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser
    for inference benchmarking.
    """

    parser = argparse.ArgumentParser(
        description="Image Captioning Inference Benchmark"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device used for inference benchmarking."
    )

    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=10,
        help="Number of unmeasured warm-up executions."
    )

    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=100,
        help="Number of measured inference executions."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/results/benchmark_results.json",
        help="Path of the benchmark results JSON file."
    )

    return parser


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """
    Parse and validate CLI arguments.

    Args:
        argv:
            Optional command-line arguments.
            None means using sys.argv.

    Returns:
        argparse.Namespace:
            Validated benchmark configuration.
    """

    # Step 1: Create the argument parser
    parser = build_parser()

    # Step 2: Parse command-line arguments
    args = parser.parse_args(argv)

    # Step 3: Validate warm-up runs
    if args.warmup_runs < 0:
        parser.error(
            "--warmup-runs must be non-negative."
        )

    # Step 4: Validate benchmark runs
    if args.benchmark_runs <= 0:
        parser.error(
            "--benchmark-runs must be positive."
        )

    # Step 5: Return validated arguments
    return args

def run_benchmark(
    inference_fn: Callable[[], object],
    warmup_runs: int,
    benchmark_runs: int,
    device: str,
) -> dict[str, object]:
    """
    Run inference benchmarking and calculate performance metrics.

    Args:
        inference_fn:
            Callable that executes one inference operation.
        warmup_runs:
            Number of unmeasured warmup executions.
        benchmark_runs:
            Number of measured inference executions.
        device:
            Target inference device.

    Returns:
        Dictionary containing benchmark configuration and performance metrics.
    """

    # Step 1: Execute inference benchmarking
    benchmark_result = benchmark_inference(
        inference_fn=inference_fn,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        device=device,
    )

    # Step 2: Calculate latency statistics
    latency_metrics = calculate_latency_metrics(
        list(benchmark_result.latencies_ms)
        )

    # Step 3: Calculate inference throughput
    throughput = calculate_throughput(
        num_inferences=len(benchmark_result.latencies_ms),
        total_seconds=benchmark_result.total_seconds,
        )

    # Step 4: Build structured result
    results = {
        "device": device,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "total_seconds": benchmark_result.total_seconds,
        "latency_ms": asdict(latency_metrics),
        "throughput_per_second": throughput,
    }

    return results

def save_benchmark_results(
    results: dict[str, object],
    output_path: str,
) -> None:
    """
    Serialize benchmark results into a JSON file.

    Args:
        results:
            Benchmark configuration and performance metrics

        output_path:
            Destination path for the JSON file.

    Raises:
        OSError:
            If the output directory or file cannot be created.

        TypeError:
            If results contain objects that cannot be serialized
    """

    # Step 1: Convert output path into a Path object
    path = Path(output_path)

    # Step 2: Create parent directories if necessary
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 3: Serialize benchmark results
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
            allow_nan=False,
        )

def create_benchmark_inference(
        device: str,
) -> ImageCaptioningInference:
    """
    Initialize the real PyTorch image-captioning
    inference service for benchmarking.

    Args:
        device:
            Target execution device.

    Returns:
        ImageCaptioningInference:
            Initialized inference service.
    """

    torch_device = torch.device(device)

    vocab_path = (
        PROJECT_ROOT / "artifacts" / "vocab.pkl"
    )

    encoder_checkpoint = (
        PROJECT_ROOT
        / "artifacts"
        / "models"
        / "encoder-3.pt"
    )

    decoder_checkpoint =(
        PROJECT_ROOT
        / "artifacts"
        / "models"
        / "decoder-3.pt"
    )

    with vocab_path.open("rb") as file:
        vocab = pickle.load(file)

    inference_service = ImageCaptioningInference(
        config=ModelConfig,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
        vocab=vocab,
        device=torch_device,
    )

    return inference_service

def main(
    argv: list[str] | None = None,
    *,
    inference_fn: Callable[[], object] | None = None,
) -> int:
    """
    Execute the inference benchmark CLI.

    Args:
        argv:
            Optional command-line arguments.

        inference_fn:
            Optional inference callable supplied by the caller.

    Returns:
        int:
            Process exit code.
            0 indicates sucessful execution.
    """

    # Step 1: Parse CLI arguments
    args = parse_args(argv)

    # Step 2: Validate inference dependency
    if inference_fn is None:
        raise ValueError(
            "An inference function must be provided."
        )

    # Step 3: Execute benchmarking
    results = run_benchmark(
        inference_fn=inference_fn,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        device=args.device,
    )

    # Step 4: Save benchmark results
    save_benchmark_results(
        results=results,
        output_path=args.output,
    )

    # Step 5: Report Completion
    print(f"Benchmark results saved to: {args.output}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())