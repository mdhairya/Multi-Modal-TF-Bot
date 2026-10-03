# Multi-Modal-TF-Bot 🤖👁️🗣️

A complete deep learning pipeline built from scratch in TensorFlow 2.x for Visual Question Answering (VQA). This architecture combines a Vision Transformer (ViT) to process uploaded images and a Transformer Encoder to process text queries, fusing them via Cross-Attention to output accurate answers.

## Features
- **Custom Vision Transformer (ViT)**: Implements custom patch extraction and positional embeddings.
- **Text Encoder**: Multi-layer Transformer block for capturing sequence context.
- **Cross-Modal Fusion**: Fuses vision and text pathways logically before global pooling.
- **Custom Training Loop**: Highly configurable training via `tf.GradientTape`.

## Installation

```bash
git clone [https://github.com/yourusername/Multi-Modal-TF-Bot.git](https://github.com/yourusername/Multi-Modal-TF-Bot.git)
cd Multi-Modal-TF-Bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
