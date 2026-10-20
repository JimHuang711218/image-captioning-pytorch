from typing import Any

from evaluation.dataset import EvaluationDataset
from evaluation.metrics import calculate_corpus_bleu
from src.inference import ImageCaptioningInference

import argparse
from pathlib import Path

import json

import torch

import pickle

from src.config import ModelConfig

def resolve_device(device_name: str) -> torch.device:
    """
    Validate the requested inference device.

    Args:
        device_name:
            Requested device: cpu, cuda, or mps

    Returns:
        Validated torch.device instance.

    Reaise:
        ValueError:
            If the requested device is unsupported or unavailable.
    """

    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "CUDA was requested but is not available."
            )

        return torch.device("cuda")

    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "MPS was requested but is not availabe."
            )

        return torch.device("mps")

    raise ValueError(
        f"Unsupported device: {device_name}"
    )

def save_evaluation_results(
    results: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        results:
            Evaluation results containing sample count
            and BLEU metrics.

        output_path:
            Destination JSON file.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

def parse_args(
    args: list[str] | None = None,
) -> argparse.Namespace:
    """
    Parse command-line arguments for model evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate image captioning model."
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Number of evaluation samples.",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device used for inference.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/bleu_results.json"),
        help="Path to save evaluation results.",
    )

    parser.add_argument(
    "--image-dir",
    type=Path,
    required=True,
    help="Directory containing COCO evaluation images.",
    )

    parser.add_argument(
        "--annotations-file",
        type=Path,
        required=True,
        help="COCO evaluation annotations JSON file.",
    )

    parser.add_argument(
        "--vocab-file",
        type=Path,
        required=True,
        help="Path to the serialized vocabulary file.",
    )

    parser.add_argument(
        "--encoder-checkpoint",
        type=Path,
        required=True,
        help="Path to the trained encoder checkpoint.",
    )

    parser.add_argument(
        "--decoder-checkpoint",
        type=Path,
        required=True,
        help="Path to the trained decoder checkpoint.",
    )



    return parser.parse_args(args)


def evaluate_model(
    inference: ImageCaptioningInference,
    dataset: EvaluationDataset,
    num_samples: int,
) -> dict[str, Any]:
    """
    Evaluate an Image-Captioning model on a fixed dataset subset.

    Args:
        inference:
            Initialized production inference pipeline.

        dataset:
            COCO evaluation dataset.

        num_samples:
            Number of images to evaluate.

    Returns:
        Dictionary containing the evaluated sample count
        and corpus-level BLEU-1 through BLEU-4
    """

    # Validate evaluation configuration
    if num_samples <= 0:
        raise ValueError(
            "num_samples must be greater than zero."
        )

    if num_samples > len(dataset):
        raise ValueError(
            f"Requested {num_samples} samples,"
            f"but dataset only contains {len(dataset)}"
        )

    predictions: list[str] = []
    references: list[list[str]] = []

    # Evaluate a deterministic subset
    for index in range(num_samples):

        sample = dataset[index]

        # Generate a caption using production inference
        prediction = inference.predict(
            sample.image_path
        )

        predictions.append(prediction)
        references.append(sample.references)

    # Calculate corpus-level BLEU metrics
    bleu_scores = calculate_corpus_bleu(
        predictions=predictions,
        references=references
    )

    return {
        "num_samples": num_samples,
        "metrics": bleu_scores,
    }

def main(
    args: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the complete image-captioning evaluation pipeline.

    Workflow:
        1. Parse CLI arguments.
        2. Validate execution device.
        3. Load vocabulary.
        4. Initialize evaluation dataset.
        5. Initialize production inference pipeline.
        6. Evaluate model quality.
        7. Export evaluation results to JSON
    """

    # 1. Parse Command-line arguments
    parsed_args = parse_args(args)

    # 2. Resolve inference device
    device = resolve_device(parsed_args.device)

    # 3. Load vocabulary
    with parsed_args.vocab_file.open("rb") as file:
        vocab = pickle.load(file)

    # 4. Initialize model configuration
    config = ModelConfig()

    # 5. Initialize COCO evaluation dataset
    dataset = EvaluationDataset(
        image_dir=parsed_args.image_dir,
        annotations_file=parsed_args.annotations_file,
    )

    # 6. Initialize production inference pipeline
    inference = ImageCaptioningInference(
        config=config,
        encoder_checkpoint=parsed_args.encoder_checkpoint,
        decoder_checkpoint=parsed_args.decoder_checkpoint,
        vocab=vocab,
        device=device,
    )

    # 7. Evaluate model
    results = evaluate_model(
        inference=inference,
        dataset=dataset,
        num_samples=parsed_args.max_samples,
    )

    # 8. Attatch evaluation metadata
    results["evaluation_config"] = {
        "device": str(device),
        "max_samples": parsed_args.max_samples,
        "image_dir": str(parsed_args.image_dir),
        "annotations_file": str(parsed_args.annotations_file),
        "vocab_file": str(parsed_args.vocab_file),
        "encoder_checkpoint": str(
            parsed_args.encoder_checkpoint
        ),
        "decoder_checkpoint": str(
            parsed_args.decoder_checkpoint
        ),
        "model_config": {
            "embed_size": config.embed_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "max_caption_length": config.max_caption_length,
        },
    }

    # 9. Save evaluation results
    save_evaluation_results(
        results=results,
        output_path=parsed_args.output,
    )

    print(
        f"Evaluation completed successfully. "
        f"Results saved to: {parsed_args.output}"
    )

    return results

if __name__ == "__main__":
    main()
