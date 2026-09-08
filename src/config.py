from dataclasses import dataclass

@dataclass
class ModelConfig:
    """
    Configuration for the image captioning model.
    """

    embed_size: int = 256
    hidden_size: int = 256
    num_layers: int = 1
    max_caption_length: int = 20