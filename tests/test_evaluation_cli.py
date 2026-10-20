import json
from pathlib import Path

import pytest
import torch

from evaluation.evaluate import (
    parse_args,
    resolve_device,
    save_evaluation_results,
)


# Reusable CLI arguments
REQUIRED_ARGS = [
    "--image-dir", "data/images",
    "--annotations-file", "data/annotations.json",
    "--vocab-file", "data/vocab.pkl",
    "--encoder-checkpoint", "checkpoints/encoder.pth",
    "--decoder-checkpoint", "checkpoints/decoder.pth",
]


def test_parse_args_defaults():
    """
    Verify default CLI argument values.
    """

    args = parse_args(REQUIRED_ARGS)

    assert args.max_samples == 100
    assert args.device == "cpu"
    assert args.output == Path(
        "evaluation/results/bleu_results.json"
    )


def test_parse_args_custom_values():
    """
    Verify custom CLI arguments override defaults.
    """

    args = parse_args(
        REQUIRED_ARGS + [
            "--max-samples", "500",
            "--device", "mps",
            "--output", "results/custom.json",
        ]
    )

    assert args.max_samples == 500
    assert args.device == "mps"
    assert args.output == Path("results/custom.json")


@pytest.mark.parametrize(
    "device",
    ["invalid", "gpu", "tpu"],
)
def test_parse_args_invalid_device(device):
    """
    Reject unsupported inference devices.
    """

    with pytest.raises(SystemExit) as exc_info:
        parse_args(
            REQUIRED_ARGS + [
                "--device", device,
            ]
        )

    assert exc_info.value.code == 2


def test_parse_args_missing_required_arguments():
    """
    Reject CLI execution when required arguments
    are missing.
    """

    with pytest.raises(SystemExit) as exc_info:
        parse_args([])

    assert exc_info.value.code == 2


def test_resolve_device_cpu():
    """
    Verify CPU device resolution.
    """

    device = resolve_device("cpu")

    assert device == torch.device("cpu")


def test_resolve_device_invalid():
    """
    Reject unsupported inference devices.
    """

    with pytest.raises(
        ValueError,
        match="Unsupported device",
    ):
        resolve_device("invalid")


def test_save_evaluation_results(tmp_path):
    """
    Verify evaluation results are serialized
    correctly into a JSON file.
    """

    results = {
        "num_samples": 2,
        "metrics": {
            "BLEU-1": 0.5,
            "BLEU-2": 0.4,
            "BLEU-3": 0.3,
            "BLEU-4": 0.2,
        },
    }

    output_path = (
        tmp_path / "results" / "evaluation.json"
    )

    save_evaluation_results(
        results=results,
        output_path=output_path,
    )

    assert output_path.is_file()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_results = json.load(file)

    assert saved_results == results