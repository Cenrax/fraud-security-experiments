#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple application to classify a single text using the trained TinyBERT model
"""

import os
import argparse
import torch
from transformers import AutoTokenizer
import logging

from models.tinybert import TinyBertForSequenceClassification
from utils.data_utils import preprocess_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)

def classify_text(model_path, text, device=None):
    """
    Classify a single text using the trained TinyBERT model
    
    Args:
        model_path (str): Path to the trained model
        text (str): Text to classify
        device (str): Device to use for inference (cpu or cuda)
        
    Returns:
        dict: Classification results
    """
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    if os.path.isdir(model_path):
        # Load from local directory
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"No config.json found in {model_path}. Make sure this is a valid model directory.")
        
        model = TinyBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    else:
        # Try loading from Hugging Face Hub
        model = TinyBertForSequenceClassification.from_pretrained(model_path)
    
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Tokenize
    inputs = tokenizer(
        processed_text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    prediction = torch.argmax(logits, dim=1).item()
    
    # Convert to numpy
    fraud_probability = probabilities[0, 1].item()
    
    # Map prediction to label
    prediction_label = "Fraud" if prediction == 1 else "Non-Fraud"
    
    return {
        "text": text,
        "prediction": prediction,
        "prediction_label": prediction_label,
        "fraud_probability": fraud_probability
    }

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Classify text for financial fraud detection")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained TinyBERT model")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--file", type=str, help="File containing text to classify")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Check if text or file is provided
    if args.text is None and args.file is None:
        parser.error("Either --text or --file must be provided")
    
    # Get text from file if provided
    if args.file is not None:
        with open(args.file, "r") as f:
            text = f.read().strip()
    else:
        text = args.text
    
    # Classify text
    result = classify_text(args.model_path, text, args.device)
    
    # Print results
    print("\n" + "="*50)
    print("FINANCIAL FRAUD DETECTION RESULTS")
    print("="*50)
    print(f"Text: {result['text'][:100]}..." if len(result['text']) > 100 else f"Text: {result['text']}")
    print(f"Classification: {result['prediction_label']}")
    print(f"Fraud Probability: {result['fraud_probability']:.4f}")
    
    # Print risk assessment
    if result['fraud_probability'] >= 0.8:
        risk_level = "HIGH RISK"
    elif result['fraud_probability'] >= 0.4:
        risk_level = "MEDIUM RISK"
    else:
        risk_level = "LOW RISK"
    
    print(f"Risk Assessment: {risk_level}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
