# Standard library imports
import argparse
import json
import pickle
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

# Thrird-party imports
import torch

# Local apllication imports
from evaluation.benchmark import(
    calculate_latency_metrics,
    calculate_throughput,
    validate_device,
    benchmark_inference,
)
from evaluation.benchmark_serialization import (
    save_benchmark_result,
    serialize_benchmark_result,
)
from evaluation.benchmark_metadata import collect_benchmark_metadata
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
        type=Path,
        default=None,
        help="Optional path to save benchmark results as JSON."
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
        "latencies_ms": asdict(latency_metrics),
        "throughput_per_second": throughput,
    }

    return results

def validate_artifacts(
    vocab_path: Path,
    encoder_checkpoint: Path,
    decoder_checkpoint: Path,
) -> None:
    """
    Validate the existence of required inference artifacts.

    Raises:
        FileNotFoundError:
            If any required artifacts is not found.
    """

    required_artifacts = (
        vocab_path,
        encoder_checkpoint,
        decoder_checkpoint,
    )

    for artifact in required_artifacts:
        if not artifact.is_file():
            raise FileNotFoundError(
                f"Required inference artifact not found: {artifact}"
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

    torch_device = validate_device(device)

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

    validate_artifacts(
        vocab_path=vocab_path,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
    )

    with vocab_path.open("rb") as file:
        vocab = pickle.load(file)

    inference_service = ImageCaptioningInference(
        config=ModelConfig(),
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

    # Step 2: Initialize real inference when no callable is injected.
    if inference_fn is None:

       inference_service = create_benchmark_inference(
           device=args.device,
       )

       # Resolve benchmark image path
       image_path = (
           PROJECT_ROOT
           / "images"
           / "coco-examples.jpg"
       )

       # Validate benchmark input
       if not image_path.is_file():
           raise FileNotFoundError(
               f"Benchmark image not found: {image_path}"
           )

       # Bind the initialized service to a zero-argument callable
       def inference_fn () -> str:
           return inference_service.predict(image_path)

    # Step 3: Execute benchmarking
    results = benchmark_inference(
        inference_fn=inference_fn,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        device=args.device,
    )

    # Step 4: Collect benchmark metadata
    metadata = collect_benchmark_metadata(
        device=args.device,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs
    )

    # Step 5. Optional persist benchmark results
    if args.output is not None:
        serialized = serialize_benchmark_result(
            result=results,
            metadata=metadata,
        )

        # Step 6: Save benchmark results
        save_benchmark_result(
            serialized=serialized,
            output_path=args.output,
        )

        # Step 7: Report Completion
        print(f"Benchmark results saved to: {args.output}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
