import pytest
import json
import pickle

from pathlib import Path
from unittest.mock import patch

import torch

from evaluation.evaluate import main
from evaluation.evaluate import resolve_device

def test_main_success(tmp_path):
    """
    Verify that main() correctly orchestrates the
    complete evaluation pipeline without loading
    real models or COCO images.
    """

    # 1. Create a temporary vocabulary file
    vocab_file = tmp_path / "vocab.pkl"

    fake_vocab = {
        "a": 0,
        "dog": 1,
        "runs": 2,
    }

    with vocab_file.open("wb") as file:
        pickle.dump(fake_vocab, file)

    # 2. Define CLI arguments
    output_path = tmp_path / "results" / "bleu_results.json"

    args = [
        "--image-dir", str(tmp_path / "images"),
        "--annotations-file", str(tmp_path / "annotations.json"),
        "--vocab-file", str(vocab_file),
        "--encoder-checkpoint", str(tmp_path / "encoder.pth"),
        "--decoder-checkpoint", str(tmp_path / "decoder.pth"),
        "--max-samples", "2",
        "--device", "cpu",
        "--output", str(output_path),
    ]

    # 3. Mock expensive dependencies
    with (
        patch("evaluation.evaluate.EvaluationDataset") as mock_dataset,
        patch("evaluation.evaluate.ImageCaptioningInference") as mock_inference,
        patch("evaluation.evaluate.evaluate_model") as mock_evaluate,
    ):

        mock_evaluate.return_value = {
            "num_samples": 2,
            "metrics": {
                "BLEU-1": 0.5,
                "BLEU-2": 0.4,
                "BLEU-3": 0.3,
                "BLEU-4": 0.2,
            },
        }

        # 4. Execute the application entry point
        results = main(args)

        # 5. Verify dependencies were initialized
        mock_dataset.assert_called_once_with(
            image_dir=Path(tmp_path / "images"),
            annotations_file=Path(tmp_path / "annotations.json"),
        )

        mock_inference.assert_called_once()

        inference_kwargs = mock_inference.call_args.kwargs

        assert inference_kwargs["device"] == torch.device("cpu")
        assert inference_kwargs["encoder_checkpoint"] == (
            tmp_path / "encoder.pth"
        )
        assert inference_kwargs["decoder_checkpoint"] == (
            tmp_path / "decoder.pth"
        )
        assert inference_kwargs["vocab"] == fake_vocab

        # 6. Verify evaluation was executed
        mock_evaluate.assert_called_once_with(
            inference=mock_inference.return_value,
            dataset=mock_dataset.return_value,
            num_samples=2,
        )

    # 7. Verify JSON output
    assert output_path.exists()

    with output_path.open("r", encoding="utf-8") as file:
        saved_results = json.load(file)

    assert saved_results == results
    assert results["num_samples"] == 2
    assert results["metrics"]["BLEU-4"] == 0.2

    # 8. Verify evaluation metadata
    assert results["evaluation_config"]["device"] == "cpu"
    assert results["evaluation_config"]["max_samples"] == 2
    assert results["evaluation_config"]["model_config"]["embed_size"] == 256


def test_main_missing_vocab(tmp_path):

    args = [
        "--image-dir", str(tmp_path / "images"),
        "--annotations-file", str(tmp_path / "annotations.json"),
        "--vocab-file", str(tmp_path / "missing_vocab.pkl"),
        "--encoder-checkpoint", str(tmp_path / "encoder.pth"),
        "--decoder-checkpoint", str(tmp_path / "decoder.pth"),
    ]

    with pytest.raises(FileNotFoundError):
        main(args)


def test_resolve_device_cuda_unavailable():

    with patch(
        "evaluation.evaluate.torch.cuda.is_available",
        return_value=False,
    ):
        with pytest.raises(
            ValueError,
            match="CUDA was requested but is not available",
        ):
            resolve_device("cuda")