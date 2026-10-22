from pathlib import Path

import pytest
import json

from evaluation.qualitative import QualitativeSample

from unittest.mock import Mock

from evaluation.dataset import EvaluationSample
from evaluation.qualitative import generate_qualitative_samples, serialize_qualitative_samples


def test_qualitative_sample_creation():

    sample = QualitativeSample(
        image_id=123,
        image_path=Path("images/test.jpg"),
        predicted_caption="a dog running",
        reference_captions=(
            "a dog is running",
            "a dog running outside"
        ),
    )

    assert sample.image_id == 123

    assert sample.image_path == Path("images/test.jpg")

    assert sample.predicted_caption == "a dog running"

    assert len(sample.reference_captions) == 2


def test_qualitative_sample_is_immutable():
    
    sample = QualitativeSample(
        image_id=123,
        image_path=("images/test.jpg"),
        predicted_caption="a dog running",
        reference_captions=(
            "a dog is running",
            "a dog running outside"
        ),
    )

    with pytest.raises(AttributeError):

        sample.predicted_caption = "a cat sleeping"


'''---Test Stage 6.5.5----'''

'''Test 1: Successful Caption Generation'''
def test_generate_qualitative_samples_success():

    # Create mocked evaluation samples.
    samples = [
        EvaluationSample(
            image_id=101,
            image_path=Path("image_001.jpg"),
            references=[
                "a dog running",
                "a dog playing outside",
            ],
        ),
        EvaluationSample(
            image_id=102,
            image_path=Path("image_002.jpg"),
            references=[
                "a cat sleeping",
                "a cat resting on a sofa",
            ],
        ),
    ]

    # Mock the evaluation dataset.
    mock_dataset = Mock()

    mock_dataset.__len__ = Mock(
        return_value=len(samples)
    )

    mock_dataset.__getitem__ = Mock(
        side_effect=lambda index: samples[index]
    )

    # Mock production inference.
    mock_inference = Mock()

    predictions = {
        Path("image_001.jpg"): "a dog running",
        Path("image_002.jpg"): "a cat sleeping",
    }

    mock_inference.predict.side_effect = (
        lambda image_path: predictions[image_path]
    )

    # Execute qualitative evaluation.
    results = generate_qualitative_samples(
        inference=mock_inference,
        dataset=mock_dataset,
        num_samples=2,
    )

    # Verify generated records.
    assert len(results) == 2

    assert results[0].image_id == 101
    assert results[0].predicted_caption == "a dog running"

    assert results[0].reference_captions == (
        "a dog running",
        "a dog playing outside",
    )

    assert results[1].image_id == 102
    assert results[1].predicted_caption == "a cat sleeping"

    # Verify production inference was called correctly.
    assert mock_inference.predict.call_count == 2


'''Test 2: Invalid Sample Count'''
@pytest.mark.parametrize(
    "num_samples",
    [0, -1, 3],
)
def test_generate_qualitative_samples_invalid_count(
    num_samples,
):

    mock_dataset = Mock()

    mock_dataset.__len__ = Mock(return_value=2)

    mock_inference = Mock()

    with pytest.raises(ValueError):

        generate_qualitative_samples(
            inference=mock_inference,
            dataset=mock_dataset,
            num_samples=num_samples,
        )

    # Invalid inputs must not trigger inference.
    mock_inference.predict.assert_not_called()


'''Test 3: Missing image identifier'''
def test_generate_qualitative_samples_missing_image_id():

    sample = EvaluationSample(
        image_path=Path("image_001.jpg"),
        references=["a dog running"],
    )

    mock_dataset = Mock()

    mock_dataset.__len__ = Mock(return_value=1)

    mock_dataset.__getitem__ = Mock(
        return_value=sample
    )

    mock_inference = Mock()

    with pytest.raises(ValueError, match="Missing image_id"):

        generate_qualitative_samples(
            inference=mock_inference,
            dataset=mock_dataset,
            num_samples=1,
        )

    mock_inference.predict.assert_not_called()

'''Test 4: Serialization Success'''
def test_serialize_qualitative_samples():

    samples = [
        QualitativeSample(
            image_id=101,
            image_path=Path("image_001.jpg"),
            predicted_caption="a dog running",
            reference_captions=(
                "a dog is running",
                "a dog playing outside",
            ),
        ),
        QualitativeSample(
            image_id=102,
            image_path=Path("image_002.jpg"),
            predicted_caption="a cat sleeping",
            reference_captions=(
                "a cat is sleeping",
                "a cat resting",
            ),
        ),
    ]

    results = serialize_qualitative_samples(samples)

    assert results["num_samples"] == 2

    assert len(results["samples"]) == 2

    assert results["samples"][0]["image_id"] == 101

    assert results["samples"][0]["image_path"] == (
        "image_001.jpg"
    )

    assert results["samples"][0]["predicted_caption"] == (
        "a dog running"
    )

    assert results["samples"][0]["reference_captions"] == [
        "a dog is running",
        "a dog playing outside",
    ]

'''Test 5: Empty Samples'''
def test_serialize_empty_qualitative_samples():

    results = serialize_qualitative_samples([])

    assert results == {
        "num_samples": 0,
        "samples": [],
    }


'''Test 6: JSON Compatibility'''
def test_qualitative_samples_json_serializable():

    sample = QualitativeSample(
        image_id=101,
        image_path=Path("image_001.jpg"),
        predicted_caption="a dog running",
        reference_captions=(
            "a dog is running",
        ),
    )

    results = serialize_qualitative_samples([sample])

    # Verify the dictionary can be encoded as JSON.
    json_string = json.dumps(results)

    # Decode JSON to verify structural integrity.
    decoded_results = json.loads(json_string)

    assert decoded_results == results