import json
from pathlib import Path


def test_qualitative_analysis_matches_predictions():

    results_dir = Path("evaluation/results")

    with (
        results_dir / "bleu_results_qualitative.json"
    ).open(encoding="utf-8") as file:
        predictions = json.load(file)

    with (
        results_dir / "qualitative_analysis.json"
    ).open(encoding="utf-8") as file:
        analysis = json.load(file)

    prediction_ids = {
        sample["image_id"]
        for sample in predictions["samples"]
    }

    annotation_ids = {
        annotation["image_id"]
        for annotation in analysis["annotations"]
    }

    # All reviewed images must exist in the prediction report.
    assert annotation_ids <= prediction_ids

    # Each image must have exactly one annotation.
    assert len(annotation_ids) == len(
        analysis["annotations"]
    )

    # Recorded sample count must match actual annotations.
    assert analysis["reviewed_samples"] == len(
        analysis["annotations"]
    )