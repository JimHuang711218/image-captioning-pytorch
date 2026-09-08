import pickle
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from src.config import ModelConfig
from src.model import EncoderCNN, DecoderRNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_v1_checkpoint_inference():
    config = ModelConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    vocab_path = PROJECT_ROOT / "notebooks" / "vocab.pkl"

    with vocab_path.open("rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab)

    encoder = EncoderCNN(
        embed_size=config.embed_size
    ).to(device)

    decoder = DecoderRNN(
        embed_size=config.embed_size,
        hidden_size=config.hidden_size,
        vocab_size=vocab_size,
        num_layers=config.num_layers
    ).to(device)

    encoder_path = (
        PROJECT_ROOT
        / "notebooks"
        / "models"
        / "encoder-3.pt"
    )

    decoder_path = (
        PROJECT_ROOT
        / "notebooks"
        / "models"
        / "decoder-3.pt"
    )

    encoder.load_state_dict(
        torch.load(
            encoder_path,
            map_location=device
        )
    )

    decoder.load_state_dict(
        torch.load(
            decoder_path,
            map_location=device
        )
    )

    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_path = PROJECT_ROOT / "images" / "coco-examples.jpg"

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(
            image_tensor
        )

        predicted_ids = decoder.sample(
            features,
            max_len=config.max_caption_length
        )

    assert len(
        predicted_ids
    ) > 0