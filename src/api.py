import pickle
import tempfile
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.config import ModelConfig
from src.inference import ImageCaptioningInference


PROJECT_ROOT = Path(__file__).resolve().parents[1]


app = FastAPI(
    title="Image Captioning API",
    version="1.0.0"
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

config = ModelConfig()

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

vocab_path = PROJECT_ROOT / "notebooks" / "vocab.pkl"

encoder_checkpoint = (
    PROJECT_ROOT
    / "notebooks"
    / "models"
    / "encoder-3.pt"
)

decoder_checkpoint = (
    PROJECT_ROOT
    / "notebooks"
    / "models"
    / "decoder-3.pt"
)


with open(vocab_path, "rb") as file:
    vocab = pickle.load(file)


inference_service = ImageCaptioningInference(
    config=config,
    encoder_checkpoint=encoder_checkpoint,
    decoder_checkpoint=decoder_checkpoint,
    vocab=vocab,
    device=device,
)

@app.post("/predict")
async def predict(
    image: UploadFile = File(...)
):
    if image.content_type is None:
        raise HTTPException(
            status_code=400,
            detail="Missing content type."
        )

    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image."
        )

    suffix = Path(image.filename or "image.jpg").suffix

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False
    ) as temp_file:
        contents = await image.read()
        temp_file.write(contents)
        temp_path = Path(temp_file.name)

    try:
        caption = inference_service.predict(temp_path)

        return {
            "caption": caption
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Image caption generation failed."
        ) from exc

    finally:
        temp_path.unlink(missing_ok=True)