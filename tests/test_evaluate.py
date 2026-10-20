import pytest

from pathlib import Path
from unittest.mock import Mock, patch

from evaluation.dataset import EvaluationSample
from evaluation.evaluate import evaluate_model


@pytest.fixture
def mock_dataset():
    """
    Create a small evaluation dataset without COCO dependencies.
    """

    samples = [
        EvaluationSample(
            image_path=Path("image_001.jpg"),
            references=[
                "a dog running in the park",
                "a dog playing outside",
            ],
        ),
        EvaluationSample(
            image_path=Path("image_002.jpg"),
            references=[
                "a man riding a bicycle",
                "a person cycling outside",
            ],
        ),
        EvaluationSample(
            image_path=Path("image_003.jpg"),
            references=[
                "two children playing soccer",
                "kids playing football",
            ],
        ),
    ]

    dataset = Mock()

    dataset.__len__ = Mock(return_value=len(samples))
    dataset.__getitem__ = Mock(
        side_effect=lambda index: samples[index]
    )

    return dataset


@pytest.fixture
def mock_inference():
    """
    Simulate the production inference pipeline.
    """

    inference = Mock()

    predictions = [
        "a dog running in the park",
        "a man riding a bicycle",
        "two children playing soccer",
    ]

    inference.predict.side_effect = predictions

    return inference

def test_evaluate_model_success(
    mock_dataset,
    mock_inference,
):
    """
    Verify that evaluation completes successfully.
    """

    result = evaluate_model(
        inference=mock_inference,
        dataset=mock_dataset,
        num_samples=3,
    )

    assert result["num_samples"] == 3

    assert "metrics" in result

    assert set(result["metrics"].keys()) == {
        "BLEU-1",
        "BLEU-2",
        "BLEU-3",
        "BLEU-4",
    }

def test_evaluate_model_uses_correct_samples(
    mock_dataset,
    mock_inference,
):
    """
    Verify that evaluation uses the first N samples.
    """

    evaluate_model(
        inference=mock_inference,
        dataset=mock_dataset,
        num_samples=2,
    )

    assert mock_dataset.__getitem__.call_count == 2

    mock_dataset.__getitem__.assert_any_call(0)
    mock_dataset.__getitem__.assert_any_call(1)

    mock_inference.predict.assert_any_call(
        Path("image_001.jpg")
    )

    mock_inference.predict.assert_any_call(
        Path("image_002.jpg")
    )

def test_evaluate_model_prediction_alignment(
    mock_dataset,
    mock_inference,
):
    """
    Verify predictions and references remain aligned.
    """

    with patch(
        "evaluation.evaluate.calculate_corpus_bleu"
    ) as mock_bleu:

        mock_bleu.return_value = {
            "BLEU-1": 0.5,
            "BLEU-2": 0.4,
            "BLEU-3": 0.3,
            "BLEU-4": 0.2,
        }

        evaluate_model(
            inference=mock_inference,
            dataset=mock_dataset,
            num_samples=2,
        )

        mock_bleu.assert_called_once_with(
            predictions=[
                "a dog running in the park",
                "a man riding a bicycle",
            ],
            references=[
                [
                    "a dog running in the park",
                    "a dog playing outside",
                ],
                [
                    "a man riding a bicycle",
                    "a person cycling outside",
                ],
            ],
        )

@pytest.mark.parametrize(
    "num_samples",
    [0, -1, -10],
)
def test_evaluate_model_invalid_sample_count(
    mock_dataset,
    mock_inference,
    num_samples,
):
    """
    Reject zero or negative evaluation sample counts.
    """

    with pytest.raises(
        ValueError,
        match="num_samples must be greater than zero",
    ):
        evaluate_model(
            inference=mock_inference,
            dataset=mock_dataset,
            num_samples=num_samples,
        )

    mock_inference.predict.assert_not_called()

def test_evaluate_model_exceeds_dataset_size(
    mock_dataset,
    mock_inference,
):
    """
    Reject requests larger than the available dataset.
    """

    with pytest.raises(
        ValueError,
        match="Requested 10 samples",
    ):
        evaluate_model(
            inference=mock_inference,
            dataset=mock_dataset,
            num_samples=10,
        )

    mock_inference.predict.assert_not_called()

def test_evaluate_model_returns_metrics(
    mock_dataset,
    mock_inference,
):
    """
    Verify that evaluation returns the calculated BLEU scores.
    """

    expected_scores = {
        "BLEU-1": 0.8,
        "BLEU-2": 0.7,
        "BLEU-3": 0.6,
        "BLEU-4": 0.5,
    }

    with patch(
        "evaluation.evaluate.calculate_corpus_bleu",
        return_value=expected_scores,
    ):

        result = evaluate_model(
            inference=mock_inference,
            dataset=mock_dataset,
            num_samples=3,
        )

    assert result == {
        "num_samples": 3,
        "metrics": expected_scores,
    }

def test_evaluation_reproducibility(mock_dataset):
    """
    Verify that repeated evaluations with identical inputs
    produce identical BLEU metrics.

    This test uses deterministic mocked inference to avoid
    dependencies on real model checkpoints and COCO images.
    """

    # Fixed predictions for each evaluation image
    predictions = {
        Path("image_001.jpg"): "a dog running in the park",
        Path("image_002.jpg"): "a man riding a bicycle",
        Path("image_003.jpg"): "two children playing soccer",
    }

    # Create deterministic mocked inference
    mock_inference = Mock()

    mock_inference.predict.side_effect = (
        lambda image_path: predictions[image_path]
    )

    # First evaluation run
    result1 = evaluate_model(
        inference=mock_inference,
        dataset=mock_dataset,
        num_samples=3,
    )

    # Second evaluation run with identical inputs
    result2 = evaluate_model(
        inference=mock_inference,
        dataset=mock_dataset,
        num_samples=3,
    )

    # Verify evaluation sample counts
    assert result1["num_samples"] == result2["num_samples"]

    # Verify identical BLEU metrics
    assert result1["metrics"] == result2["metrics"]

    # Verify complete evaluation results
    assert result1 == result2

    # Each evaluation processes three images
    assert mock_inference.predict.call_count == 6
