---
title: Image2Wiki
emoji: 🖼️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Image2Wiki

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/letitbE/image2wiki)

Image2Wiki is a service that generates Wikipedia-style articles based on an uploaded image. It uses a fine-tuned VisionEncoderDecoder model (`tuman/vit-rugpt2-image-captioning` with a LoRA adapter) to generate structured text (title, lead, sections, paragraphs) from images.

## Features
- FastAPI based web service
- Wikipedia-like UI for generated articles
- Fine-tuned model for structured article generation

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the service:
```bash
uvicorn app.main:app --port 8013 --reload
```

## Project Structure
- `app/` - FastAPI application and UI templates
- `adapted_best_embed2/` - Fine-tuned LoRA adapter weights
- `collect_data.py` & `collect_data_async.py` - Scripts for collecting training data
- `finetune.ipynb` - Notebook used for fine-tuning the model
