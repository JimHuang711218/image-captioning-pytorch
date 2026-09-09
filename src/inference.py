from pathlib import Path
from typing import Union

import torch
from PIL import Image
from torchvision import transforms

from src.config import ModelConfig
from src.model import EncoderCNN, DecoderRNN

class ImageCaptioningInference:
    '''
    Production-oriented inference pipeline for the image captioning image

    Responsibilities:
    - Load vocabulary
    - Build encoder and decoder
    - Load trained checkpoints
    - Prepare models for inference
    - Preprogress input images
    - Generate token IDs
    - Decode tokens into a human-readable caption
    '''
    # Initialization and Model Loading
    def __init__(
        self,
        config: ModelConfig,
        encoder_checkpoint: Path,
        decoder_checkpoint: Path,
        vocab,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.vocab = vocab

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.encoder = EncoderCNN(
            embed_size=config.embed_size
        )

        self.decoder = DecoderRNN(
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
            vocab_size=len(vocab)
        )

        # Load the trained parameters
        self.encoder.load_state_dict(
            torch.load(
                encoder_checkpoint,
                map_location=device
            )
        )

        self.decoder.load_state_dict(
           torch.load(
                decoder_checkpoint,
                map_location=device
            )
        )

        # Move both models to the selected device
        self.encoder.to(device)
        self.decoder.to(device)

        # Switch into evaluation
        self.encoder.eval()
        self.decoder.eval()
    
    # Image Preprocessing
    def preprocess_image(
        self,
        image: Image.Image
    ) -> torch.Tensor:
        image_tensor = self.transform(image)

        # Add a batch dimension: [C, H, W] -> [1, C, H, W],
        # because the CNN expects input of shape [B, C, H, W]
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)
    
    # Generate Token IDs
    def generate_tokens(
        self,
        image_tensor: torch.Tensor
    ):
        with torch.no_grad():
            features = self.encoder(image_tensor)

            token_id = self.decoder.sample(
                features,
                max_len=self.config.max_caption_length
            )

        return token_id
    
    # Decode Token IDs
    def decode_tokens(
        self,
        token_ids
    ) -> str:
        words = []
        
        for token_id in token_ids:
            word = self.vocab.idx2word[token_id]

            if word == '<end>':
                break

            if word not in {"<start>", "<pad>"}:
                words.append(word)
        
        return " ".join(words)
    
    # Public predict() interface
    def predict(
        self,
        image_path: Union[str, Path]
    ) -> str:
        # Load the images and ensure it has exactly 3RGB channels
        image = Image.open(image_path).convert("RGB")

        image_tensor = self.preprocess_image(image)

        token_ids = self.generate_tokens(image_tensor)

        caption = self.decode_tokens(token_ids)

        return caption