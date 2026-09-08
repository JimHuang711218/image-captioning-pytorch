import torch
import torch.nn as nn
import torchvision.models as models


LSTMState = tuple[torch.Tensor, torch.Tensor]


class EncoderCNN(nn.Module):
    """
    CNN encoder that extracts image features using a pretrained ResNet-34
    and projects them into the decoder embedding space.
    """

    def __init__(self, embed_size: int) -> None:
        super().__init__()

        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
        )

        for param in resnet.parameters():
            param.requires_grad = False

        self.resnet = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.embed = nn.Linear(
            resnet.fc.in_features,
            embed_size
        )

    def forward(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            images:
                Tensor of shape
                [batch_size, channels, height, width]

        Returns:
            Tensor of shape
                [batch_size, embed_size]
        """

        features = self.resnet(images)

        # [B, 512, 1, 1] -> [B, 512]
        features = torch.flatten(features, 1)

        # [B, 512] -> [B, embed_size]
        features = self.embed(features)

        return features


class DecoderRNN(nn.Module):
    """
    LSTM decoder that generates captions from encoded image features.
    """

    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(
            vocab_size,
            embed_size
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.linear = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(
        self,
        features: torch.Tensor,
        captions: torch.Tensor
    ) -> torch.Tensor:
        """
        Training forward pass.

        Args:
            features:
                [batch_size, embed_size]

            captions:
                [batch_size, caption_length]

        Returns:
            Vocabulary logits:
                [batch_size, caption_length, vocab_size]
        """

        embeddings = self.embed(
            captions[:, :-1]
        )

        image_features = features.unsqueeze(1)

        inputs = torch.cat(
            (image_features, embeddings),
            dim=1
        )

        hiddens, _ = self.lstm(inputs)

        outputs = self.linear(hiddens)

        return outputs

    def sample(
        self,
        features: torch.Tensor,
        states: LSTMState | None = None,
        max_len: int = 20,
        end_token_id: int | None = None
    ) -> list[int]:
        """
        Generate a caption using greedy decoding.

        Args:
            features:
                Tensor of shape [1, embed_size].

            states:
                Optional initial LSTM hidden and cell states.

            max_len:
                Maximum number of generated tokens.

            end_token_id:
                Vocabulary ID corresponding to <end>.

        Returns:
            List of predicted token IDs.

        Note:
            Current implementation supports batch_size = 1.
        """

        if features.size(0) != 1:
            raise ValueError(
                "DecoderRNN.sample currently supports batch_size=1 only."
            )

        predicted_sentence: list[int] = []

        inputs = features.unsqueeze(1)

        for _ in range(max_len):
            hiddens, states = self.lstm(
                inputs,
                states
            )

            logits = self.linear(
                hiddens[:, -1, :]
            )

            predicted = logits.argmax(dim=1)

            token_id = predicted.item()
            predicted_sentence.append(token_id)

            if (
                end_token_id is not None
                and token_id == end_token_id
            ):
                break

            inputs = self.embed(
                predicted
            ).unsqueeze(1)

        return predicted_sentence

