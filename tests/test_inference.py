from pathlib import Path
import pickle

import torch

from src.config import ModelConfig
from src.inference import ImageCaptioningInference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_inference_predict():
    config = ModelConfig()

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load vocabulary
    vocab_path = PROJECT_ROOT / "notebooks" / "vocab.pkl"

    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    # Load trained checkpoints
    encoder_checkpoint = (
        PROJECT_ROOT / "notebooks" / "models" / "encoder-3.pt"
    )

    decoder_checkpoint = (
        PROJECT_ROOT / "notebooks" / "models" / "decoder-3.pt"
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