# Image Captioning with PyTorch

An end-to-end image captioning system built with **PyTorch**, combining computer vision and natural language processing to generate natural-language descriptions from images.

This project was initially inspired by my Udacity Computer Vision coursework and has since been refactored and extended into a portfolio-level AI engineering project.

The current V1 implementation uses a **CNN-based image encoder** to extract visual features and an **LSTM-based decoder** to generate captions word by word.

## Project Overview

Image captioning is a multimodal deep learning task that connects **Computer Vision (CV)** and **Natural Language Processing (NLP)**.

Given an input image, the model:

1. Preprocesses the image.
2. Extracts visual features using a CNN encoder.
3. Projects the visual features into an embedding space.
4. Passes the image representation to an LSTM decoder.
5. Generates a natural-language caption token by token.

The project demonstrates:

- Computer vision feature extraction
- Sequence modeling for natural language generation
- Vocabulary construction and tokenization
- PyTorch model implementation
- End-to-end training and inference pipelines
- Model checkpointing
- Practical AI project organization

## Model Architecture

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
 CNN Encoder
     │
     ▼
Image Feature Vector
     │
     ▼
Embedding Layer
     │
     ▼
 LSTM Decoder
     │
     ▼
Vocabulary Projection
     │
     ▼
Generated Caption
```

The encoder extracts a compact visual representation from the input image, while the LSTM decoder models the caption as a sequence and predicts the next token based on the image features and previously generated words.

## Tech Stack

- Python
- PyTorch
- Torchvision
- CNN
- LSTM
- Natural Language Processing
- Computer Vision
- NumPy
- NLTK
- Matplotlib
- MS COCO Dataset

## Current Implementation

The V1 pipeline currently includes:

- MS COCO dataset loading and preprocessing
- Image augmentation and normalization
- Vocabulary construction
- Caption tokenization
- CNN-based image feature extraction
- LSTM-based caption generation
- Training pipeline
- Loss and perplexity logging
- Model checkpointing
- Model loading
- End-to-end inference
- Generated caption output

## Training

The model is trained using image-caption pairs from the **MS COCO dataset**.

During training:

```text
Image
   │
   ▼
CNN Encoder
   │
   ▼
Image Features
   │
   ▼
LSTM Decoder ◄── Ground-Truth Caption Tokens
   │
   ▼
Vocabulary Scores
   │
   ▼
Cross-Entropy Loss
```

Training progress is recorded in:

```text
notebooks/training_log.txt
```

Model checkpoints are generated during training but are excluded from Git tracking to keep the repository lightweight.

## Inference

The inference pipeline loads the trained encoder and decoder checkpoints and generates a caption for an unseen image.

The decoder predicts words sequentially until the end-of-sequence token is produced or the maximum sequence length is reached.

### Example Model Output

```text
Generated caption:
"a train is traveling down the tracks near a forest."
```

This example demonstrates that the complete inference pipeline is operational while also highlighting limitations of the current V1 model. The generated sentence is grammatically coherent but does not fully correspond to the visual content of the image.

## Current Limitations

The current implementation is intended as a V1 baseline and still has several limitations:

- Caption generation can contain semantic errors or hallucinated objects.
- Decoding currently uses a simple sequential generation strategy.
- Model performance depends strongly on training duration and dataset coverage.
- No quantitative caption-quality evaluation is currently included.
- The current decoder does not implement an attention mechanism.

These limitations provide clear directions for the next development stage.

## Planned Improvements

Future development will focus on improving both model quality and engineering maturity:

- BLEU and CIDEr evaluation
- Attention-based image captioning
- Attention visualization
- Beam-search decoding
- Additional qualitative inference examples
- Refactored standalone training script
- Refactored standalone inference script
- Configuration management
- Automated testing
- Simple demo interface
- Containerized deployment

## Repository Structure

```text
image-captioning-pytorch/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   └── vocabulary.py
│
├── notebooks/
│   ├── 0_Dataset.ipynb
│   ├── 1_Preliminaries.ipynb
│   ├── 2_Training.ipynb
│   ├── 3_Inference.ipynb
│   └── training_log.txt
│
├── images/
│   ├── coco-examples.jpg
│   ├── encoder.png
│   ├── decoder.png
│   └── encoder-decoder.png
│
└── .github/
    └── workflows/
```

Model checkpoints, datasets, and generated vocabulary files are intentionally excluded from Git tracking.

## Development Status

**Current milestone: V1 — End-to-End Training and Inference Pipeline**

The complete baseline workflow is operational:

```text
Dataset
   ↓
Preprocessing
   ↓
Vocabulary
   ↓
CNN Encoder
   ↓
LSTM Decoder
   ↓
Training
   ↓
Checkpoint
   ↓
Inference
   ↓
Generated Caption
```

The next stage will focus on quantitative evaluation, improved decoding, attention mechanisms, testing, and deployment-oriented engineering.