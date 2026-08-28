import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """
    CNN encoder that extracts image features using a pretrained ResNet-34
    and projects them into the decoder embedding space.
    """

    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()

        # Load a pretrained ResNet-34 backbone.
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT
        )

        # Freeze the pretrained CNN parameters.
        for param in resnet.parameters():
            param.requires_grad = False

        # Remove the final classification layer.
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        # Project CNN features into the embedding space.
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        """
        Extract image features and project them into the embedding space.

        Args:
            images: Tensor of shape
                [batch_size, channels, height, width]

        Returns:
            Tensor of shape
                [batch_size, embed_size]
        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)

        return features


class DecoderRNN(nn.Module):
    """
    LSTM decoder that generates captions from encoded image features.
    """

    def __init__(
        self,
        embed_size,
        hidden_size,
        vocab_size,
        num_layers=1
    ):
        super(DecoderRNN, self).__init__()

        # Convert token IDs into embedding vectors.
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Process image and word embeddings sequentially.
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Map LSTM hidden states to vocabulary scores.
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        """
        Run the decoder during training.

        Args:
            features:
                Image features with shape
                [batch_size, embed_size]

            captions:
                Caption token IDs with shape
                [batch_size, caption_length]

        Returns:
            Vocabulary logits with shape
                [batch_size, caption_length, vocab_size]
        """

        # Remove the final <end> token because the model predicts
        # the next token at every time step.
        embeddings = self.embed(captions[:, :-1])

        # Add the sequence dimension to image features.
        features = features.unsqueeze(1)

        # Use the image embedding as the first LSTM input,
        # followed by the caption word embeddings.
        inputs = torch.cat(
            (features, embeddings),
            dim=1
        )

        # Process the complete sequence.
        hiddens, _ = self.lstm(inputs)

        # Convert hidden states into vocabulary logits.
        outputs = self.linear(hiddens)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """
        Generate a caption using greedy decoding.

        Args:
            inputs:
                Encoded image features with shape
                [batch_size, 1, embed_size]

            states:
                Optional initial LSTM hidden and cell states.

            max_len:
                Maximum number of generated tokens.

        Returns:
            List of predicted vocabulary token IDs.
        """

        predicted_sentence = []

        for _ in range(max_len):

            # Predict the next token.
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            _, predicted = outputs.max(1)

            predicted_sentence.append(predicted.item())

            # Feed the predicted token back into the LSTM.
            inputs = self.embed(predicted).unsqueeze(1)

        return predicted_sentence