import torch

from src.model import EncoderCNN, DecoderRNN


def test_encoder_shape():
    batch_size = 2
    embed_size = 256

    encoder = EncoderCNN(embed_size=embed_size)
    encoder.eval()

    images = torch.randn(
        batch_size,
        3,
        224,
        224
    )

    with torch.no_grad():
        features = encoder(images)

    assert features.shape == (
        batch_size,
        embed_size
    )

    print("Encoder shape test passed.")


def test_decoder_forward_shape():
    batch_size = 2
    embed_size = 256
    hidden_size = 256
    vocab_size = 1000
    caption_length = 10

    decoder = DecoderRNN(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size
    )
    decoder.eval()

    features = torch.randn(
        batch_size,
        embed_size
    )

    captions = torch.randint(
        0,
        vocab_size,
        (batch_size, caption_length)
    )

    with torch.no_grad():
        outputs = decoder(
            features,
            captions
        )

    assert outputs.shape == (
        batch_size,
        caption_length,
        vocab_size
    )

    print("Decoder forward shape test passed.")


def test_decoder_sample():
    embed_size = 256
    hidden_size = 256
    vocab_size = 1000
    max_len = 20

    decoder = DecoderRNN(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab_size=vocab_size
    )
    decoder.eval()

    features = torch.randn(
        1,
        embed_size
    )

    with torch.no_grad():
        predicted_ids = decoder.sample(
            features,
            max_len=max_len
        )

    assert isinstance(
        predicted_ids,
        list
    )

    assert all(
        isinstance(token_id, int)
        for token_id in predicted_ids
    )

    assert len(predicted_ids) <= max_len

    assert all(
        0 <= token_id < vocab_size
        for token_id in predicted_ids
    )

    print("Decoder sample test passed.")
    print("Predicted IDs:", predicted_ids)


def test_sample_rejects_batch_size_greater_than_one():
    decoder = DecoderRNN(
        embed_size=256,
        hidden_size=256,
        vocab_size=1000
    )

    features = torch.randn(
        2,
        256
    )

    try:
        decoder.sample(features)

    except ValueError:
        print("Batch-size validation test passed.")
        return

    raise AssertionError(
        "Expected ValueError for batch_size > 1."
    )


if __name__ == "__main__":
    test_encoder_shape()
    test_decoder_forward_shape()
    test_decoder_sample()
    test_sample_rejects_batch_size_greater_than_one()

    print("\nAll model tests passed.")