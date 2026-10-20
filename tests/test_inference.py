import pickle
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch

from src.config import ModelConfig
from src.inference import ImageCaptioningInference


PROJECT_ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.model
def test_inference_predict():
    config = ModelConfig()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load vocabulary
    vocab_path = PROJECT_ROOT / "artifacts" / "vocab.pkl"

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Load trained checkpoints
    encoder_checkpoint = (
        PROJECT_ROOT / "artifacts" / "models" / "encoder-3.pt"
    )

    decoder_checkpoint = (
        PROJECT_ROOT / "artifacts" / "models" / "decoder-3.pt"
    )

    inference = ImageCaptioningInference(
        config=config,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
        vocab=vocab,
        device=device,
    )

    # Replace this with one real test image in your project
    image_path = PROJECT_ROOT / "images" / "coco-examples.jpg"

    caption = inference.predict(image_path)

    print("Generated caption:", caption)

    assert isinstance(caption, str)
    assert len(caption) > 0

def test_missing_encoder_checkpoint(tmp_path):
    """
    Verify that the inference pipeline rejects
    a missing encoder checkpoint.
    """

    missing_encoder = tmp_path / "missing_encoder.pt"
    missing_decoder = tmp_path / "decoder.pt"

    vocab = Mock()
    vocab.__len__ = Mock(return_value=100)

    with (
        patch("src.inference.EncoderCNN"),
        patch("src.inference.DecoderRNN"),
    ):
        with pytest.raises(FileNotFoundError):
            ImageCaptioningInference(
                config=ModelConfig(),
                encoder_checkpoint=missing_encoder,
                decoder_checkpoint=missing_decoder,
                vocab=vocab,
                device=torch.device("cpu"),
            )

def test_missing_decoder_checkpoint(tmp_path):
    """
    Verify that the inference pipeline rejects
    a missing decoder checkpoint.
    """

    encoder_checkpoint = tmp_path / "encoder.pt"
    decoder_checkpoint = tmp_path / "missing_decoder.pt"

    vocab = Mock()
    vocab.__len__ = Mock(return_value=100)

    def mock_torch_load(path, map_location):
        if path == decoder_checkpoint:
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        return {}

    with (
        patch("src.inference.EncoderCNN"),
        patch("src.inference.DecoderRNN"),
        patch(
            "src.inference.torch.load",
            side_effect=mock_torch_load,
        ),
    ):
        with pytest.raises(
            FileNotFoundError,
            match="Checkpoint not found",
        ):
            ImageCaptioningInference(
                config=ModelConfig(),
                encoder_checkpoint=encoder_checkpoint,
                decoder_checkpoint=decoder_checkpoint,
                vocab=vocab,
                device=torch.device("cpu"),
            )