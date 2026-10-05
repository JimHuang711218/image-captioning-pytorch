# Image Captioning with PyTorch

A production-oriented image captioning system built with **PyTorch**, combining Computer Vision and Natural Language Processing to generate natural-language descriptions from images.

This project was initially developed as part of my Udacity Computer Vision coursework and has since been **refactored, productionized, containerized, and integrated with an automated CI/CD pipeline** as a portfolio-level AI engineering project.

The current V1 model uses a CNN-based image encoder and an LSTM-based decoder, while the surrounding system demonstrates the engineering required to move a machine learning model from experimentation toward a deployable AI service.

## Project Overview

Image captioning is a multimodal deep learning task that connects **Computer Vision (CV)** and **Natural Language Processing (NLP)**.

Given an input image, the model:

1. Preprocesses the image.
2. Extracts visual features using a CNN encoder.
3. Projects the visual features into an embedding space.
4. Passes the image representation to an LSTM decoder.
5. Generates a natural-language caption token by token.

Beyond model development, this project now includes a production-oriented inference and delivery pipeline with:

- Modular PyTorch inference
- FastAPI model serving
- Automated testing
- Docker containerization
- Container health checks
- GitHub Actions CI/CD
- Docker image publishing to GitHub Container Registry (GHCR)

The goal is not only to train an image-captioning model, but also to understand and implement the engineering lifecycle required to turn an ML model into a maintainable software service.

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
     |
     ▼
 FastAPI
     |
     ▼
REST API Response
```

The encoder extracts a compact visual representation from the input image, while the LSTM decoder models the caption as a sequence and predicts the next token based on the image features and previously generated words.

The inference pipeline encapsulates model loading, preprocessing, caption generation, and token decoding behind an API layer.

## Production Delivery Pipeline

The project includes an automated CI/CD workflow using **GitHub Actions**.

```text
Code Change
     │
     ▼
Git Push / Pull Request
     │
     ▼
GitHub Actions
     │
     ├──► Install Dependencies
     │
     ├──► Run CI-Safe Tests
     │
     ├──► Build Docker Image
     │
     ├──► Start Container
     │
     ├──► API Health / Smoke Test
     │
     └──► Publish Docker Image
                │
                ▼
      GitHub Container Registry
```

This pipeline automatically verifies that changes can pass the test suite, build successfully into a Docker image, launch as a running service, and respond correctly through the API before the container image is published.

## Tech Stack

### Machine Learning
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

### Software Engineering
- FastAPI
- Pytest
- Docker
- REST API
- Health checks
- Smoke testing

### CI/CD
- GitHub Actions
- GitHub Container Resgistry (GHCR)
- Automated testing
- Automated Docker builds
- Automated container verification
- Docker image publishing

## Current Implementation

The current system includes:

**Model Development**
- MS COCO dataset loading and preprocessing
- Image augmentation and normalization
- Vocabulary construction
- Caption tokenization
- CNN-based image feature extraction
- LSTM-based caption generation
- Training pipeline
- Loss and perplexity logging
- Model checkpointing

**Inference**
- Production-oriented inference pipeline
- Model checkpoint loading
- Image preprocessing
- Token generation
- Vocabulary decoding
- End-to-end caption generation

**API Serving**
- FastAPI inference service
- Image input handling
- Caption response generation
- Health endpoint

**Testing**
- Automated Pytest test suite
- CI-safe tests
- API verification
- Container smoke testing

**Containerization**
- Dockerized application
- Reproducible runtime environment
- Dependency installation through requirements.txt
- Production source and artifact packaging
- Exposed API serving on port 8000

**CI/CD**
- Automated GitHub Actions workflow
- Tests triggered by pushes and pull requests
- Docker image build inside CI
- Container startup verification
- API health checkpoint
- GitHub Container Registry authentication
- Docker image tagging
- Docker image publishing to GHCR

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

Training progress is recorded during model development, while model checkpoints are excluded from Gti tracking to keep the repository lightweight.


## Inference

The inference pipeline loads the trained encoder and decoder checkpoints and generates a caption for an unseen image.

The decoder predicts words sequentially until the end-of-sequence token is produced or the maximum sequence length is reached.

### Example Model Output

```text
Generated caption:
"a train is traveling down the tracks near a forest."
```

This example demonstrates that the complete inference pipeline is operational while also highlighting limitations of the current V1 model. The generated sentence is grammatically coherent but does not fully correspond to the visual content of the image.

## API Serving

The trained model is exposed through a FastAPI REST service.

At runtime:

```text
Client
  |
  | Image Request
  ▼
FastAPI
  |
  ▼
Inference Pipeline
  |
  ├⎯ Image Preprocessing
  ├⎯ CNN Encoding
  ├⎯ LSTM Decoding
  └── Vocabulary Decoding
  │
  ▼
Generated Caption
  │
  ▼
JSON Response
```
A health endpoint is also provided so that local environments, Docker containers, and CI workflows can verify that the service is running correctly.

## Docker

The application is packaged as a Docker image to provide a reproducible runtime environment.

The container includes the application source code, Python dependencies, and required production artifacts.

The containerized service runs the FastAPI application with Uvicorn and exposes the API through port 8000.

```text
Docker Image
     │
     ▼
Docker Container
     │
     ▼
Uvicorn
     │
     ▼
FastAPI
     │
     ▼
Image Captioning Inference
```

## CI/CD

GitHub Actions automatically validates changes pushed to the repository.

The CI/CD workflow performs the following sequence:
```text
Checkout Repository
        ↓
Set Up Python
        ↓
Install Dependencies
        ↓
Run CI-Safe Tests
        ↓
Build Docker Image
        ↓
Start Docker Container
        ↓
Test Health Endpoint
        ↓
Authenticate with GHCR
        ↓
Tag Docker Image
        ↓
Push Docker Image
```
This ensures that the application is tested not only as Python source code but also as a running containerized service.

## Current Limitations

The current model represents a V1 baseline and still has several ML limitations:

- Caption generation can contain semantic errors or hallucinated objects.
- Decoding currently uses a simple sequential generation strategy.
- Model performance depends strongly on training duration and dataset coverage.
- No quantitative caption-quality evaluation is currently included.
- The decoder does not yet implement an attention mechanism.

The production infrastructure is therefore more mature than the current baseline model architecture, leaving clear opportunities to improve both model quality and system capabilities.

## Planned Improvements

Future development will focus on both ML capability and production engineering.

**Model Improvements**
- BLEU and CIDEr evaluation
- Attention-based image captioning
- Attention visualization
- Beam-search decoding
- Additional qualitative inference examples

**Engineering Improvements**
- Refactor notebook-based training into standalone Python modules
- Configuration management improvements
- Additional unit and integration tests
- Improved model artifact management
- API observability and logging
- Deployment-oriented configuration
- Simple interactive demo interface
- C++ inference and performance-oriented experimentation

## Repository Structure

```text
image-captioning-pytorch/
│
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── data_loader.py
│   ├── inference.py
│   ├── model.py
│   └── vocabulary.py
│
├── tests/
│
├── artifacts/
│
├── notebooks/
│
├── images/
│
└── .github/
    └── workflows/
        └── ci.yml
```
Model checkpoints, datasets, caches, and other large development artifacts are excluded from Git tracking where appropriate.

## Development Status

**Current milestone: Current milestone: Productionization — Stage 5 Complete

Completed engineering stages include:

```text
Model Development
       ↓
Modular Inference
       ↓
FastAPI Serving
       ↓
Automated Testing
       ↓
Dockerization
       ↓
CI/CD
       ↓
GitHub Container Registry
```
The project has evolved from a notebook-centered deep learning project into a containerized AI inference service with automated testing and delivery infrastructure.

The next stages will continue improving production readiness, model quality, training architecture, and ML systems engineering capabilities.
