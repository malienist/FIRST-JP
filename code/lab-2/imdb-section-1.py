# IMDB Attack
# Section 1

#START
!pip install textattack torch torchvision

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textattack.attack_recipes import BAEGarg2019
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.datasets import HuggingFaceDataset, Dataset
from textattack import Attacker

# Load a pre-trained sentiment analysis model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a HuggingFace pipeline for sentiment analysis
pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Wrap the HuggingFace model for TextAttack
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Load the IMDB dataset
dataset = HuggingFaceDataset("imdb", split="test")
