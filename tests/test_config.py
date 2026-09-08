from src.config import ModelConfig


def test_model_config_defaults():
    config = ModelConfig()

    assert config.embed_size == 256
    assert config.hidden_size == 256
    assert config.num_layers == 1
    assert config.max_caption_length == 20

def test_model_config_custom_values():
    config = ModelConfig(
        embed_size=512,
        hidden_size=512,
        num_layers=2,
        max_caption_length=30
    )

    assert config.embed_size == 512
    assert config.hidden_size == 512
    assert config.num_layers == 2
    assert config.max_caption_length == 30