# Image Captioning with PyTorch

This project was initially inspired by my Udacity Computer Vision coursework and has been refactored and extended into a portfolio-level AI engineering project.

This project implements an image captioning system that generates natural-language descriptions from input images by combining computer vision and sequence modeling techniques.

The goal of this project is to build an end-to-end AI pipeline using a CNN-based image encoder and an attention-based decoder.

## Project Overview

Image captioning is a multimodal deep learning task that connects Computer Vision and Natural Language Processing. Given an input image, the model learns to extract visual features and generate a descriptive sentence word by word.

This project is designed as a portfolio-level AI engineering project to demonstrate:

- Computer vision feature extraction
- Sequence modeling for natural language generation
- Attention mechanism
- PyTorch model implementation
- Training and inference pipeline design
- Practical AI project organization

## Tech Stack

- Python
- PyTorch
- CNN Encoder
- LSTM / Attention Decoder
- NLP preprocessing
- Computer Vision
- NumPy
- pandas
- Matplotlib

## Features / Development Plan

### Current Implementation

- Dataset loading and preprocessing
- Vocabulary construction
- CNN encoder and LSTM-based decoder structure
- Training notebook
- Inference notebook


## Planned Improvement

- BLEU score evaluation
- Attention visualization
- Sample generated captions
- Refactored training and inference scripts
- Simple demo interface

## Repository Structure

```text
image-captioning-pytorch/
│
├── README.md
├── requirements.txt
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── vocabulary.py
│   ├── train.py              # planned
│   ├── inference.py          # planned
│   └── utils.py              # planned
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_training_experiment.ipynb
│   └── 03_inference_demo.ipynb
│
├── configs/
│   └── config.yaml           # planned
│
├── assets/
│   └── sample_results/       # planned
│
└── .gitignore
