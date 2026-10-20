import json
from pathlib import Path

import pytest

from evaluation.dataset import EvaluationDataset, EvaluationSample


@pytest.fixture
def coco_dataset(tmp_path: Path):
    """Create a small COCO-compatible evaluation dataset."""

    image_dir = tmp_path / "val2014"
    image_dir.mkdir()

    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()

    # Create placeholder images for path validation.
    (image_dir / "image_1.jpg").touch()
    (image_dir / "image_2.jpg").touch()

    annotations_file = annotations_dir / "captions_val2014.json"

    coco_annotations = {
        "images": [
            {"id": 2, "file_name": "image_2.jpg"},
            {"id": 1, "file_name": "image_1.jpg"},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "caption": "A dog is running.",
            },
            {
                "id": 2,
                "image_id": 1,
                "caption": "A brown dog is playing.",
            },
            {
                "id": 3,
                "image_id": 2,
                "caption": "A cat is sleeping.",
            },
        ],
        "type": "captions",
        "info": {},
        "licenses": [],
    }

    annotations_file.write_text(
        json.dumps(coco_annotations),
        encoding="utf-8",
    )

    return image_dir, annotations_file


def test_dataset_initialization(coco_dataset):
    image_dir, annotations_file = coco_dataset

    dataset = EvaluationDataset(
        image_dir=image_dir,
        annotations_file=annotations_file,
    )

    assert isinstance(dataset, EvaluationDataset)
    assert dataset.image_ids == [1, 2]


def test_dataset_length(coco_dataset):
    image_dir, annotations_file = coco_dataset

    dataset = EvaluationDataset(image_dir, annotations_file)

    assert len(dataset) == 2


def test_getitem_returns_evaluation_sample(coco_dataset):
    image_dir, annotations_file = coco_dataset

    dataset = EvaluationDataset(image_dir, annotations_file)

    sample = dataset[0]

    assert isinstance(sample, EvaluationSample)
    assert sample.image_path == image_dir / "image_1.jpg"


def test_multiple_reference_captions(coco_dataset):
    image_dir, annotations_file = coco_dataset

    dataset = EvaluationDataset(image_dir, annotations_file)

    sample = dataset[0]

    assert len(sample.references) == 2
    assert "A dog is running." in sample.references
    assert "A brown dog is playing." in sample.references


def test_missing_image_directory(tmp_path):
    annotations_file = tmp_path / "annotations.json"
    annotations_file.touch()

    with pytest.raises(FileNotFoundError):
        EvaluationDataset(
            image_dir=tmp_path / "missing_images",
            annotations_file=annotations_file,
        )


def test_missing_annotations_file(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        EvaluationDataset(
            image_dir=image_dir,
            annotations_file=tmp_path / "missing.json",
        )


def test_missing_image_file(coco_dataset):
    image_dir, annotations_file = coco_dataset

    dataset = EvaluationDataset(image_dir, annotations_file)

    (image_dir / "image_1.jpg").unlink()

    with pytest.raises(FileNotFoundError):
        dataset[0]


def test_invalid_index(coco_dataset):
    image_dir, annotations_file = coco_dataset

    dataset = EvaluationDataset(image_dir, annotations_file)

    with pytest.raises(IndexError):
        dataset[100]

def test_missing_image_directory(tmp_path):
    """
    Verify that a missing evaluation image directory
    raises FileNotFoundError.
    """

    image_dir = tmp_path / "missing_images"

    annotations_file = tmp_path / "captions.json"

    with pytest.raises(
        FileNotFoundError,
        match="Evaluation image directory not found",
    ):
        EvaluationDataset(
            image_dir=image_dir,
            annotations_file=annotations_file,
        )

def test_missing_annotations_file(tmp_path):
    """
    Verify that a missing annotations file
    raises FileNotFoundError.
    """

    image_dir = tmp_path / "images"

    image_dir.mkdir()

    annotations_file = tmp_path / "missing_captions.json"

    with pytest.raises(
        FileNotFoundError,
        match="Evaluation annotations not found",
    ):
        EvaluationDataset(
            image_dir=image_dir,
            annotations_file=annotations_file,
        )