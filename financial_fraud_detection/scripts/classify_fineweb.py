#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FineWeb Financial Fraud Classification Script
This script downloads financial data from FineWeb, preprocesses it, and classifies it using the trained TinyBERT model.
"""

import os
import argparse
import logging
import pandas as pd
import torch
import requests
import json
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add parent directory to path to import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tinybert import TinyBertForSequenceClassification
from utils.data_utils import preprocess_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S'
)

def download_fineweb_data(output_dir, category="finance", limit=1000):
    """
    Download financial data from FineWeb API
    
    Args:
        output_dir (str): Directory to save the downloaded data
        category (str): Category of data to download (default: finance)
        limit (int): Maximum number of samples to download
        
    Returns:
        str: Path to the downloaded data file
    """
    logging.info(f"Downloading FineWeb data for category: {category}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"fineweb_{category}_data.csv")
    
    # Check if file already exists
    if os.path.exists(output_file):
        logging.info(f"FineWeb data already exists at {output_file}")
        return output_file
    
    # FineWeb API endpoint (replace with actual endpoint if different)
    api_url = f"https://fineweb-api.example.com/v1/documents?category={category}&limit={limit}"
    
    try:
        # Download data from FineWeb API
        response = requests.get(api_url)
        response.raise_for_status()
        
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data["documents"])
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logging.info(f"Downloaded {len(df)} documents from FineWeb to {output_file}")
        
        return output_file
    
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading FineWeb data: {e}")
        
        # Create sample data for demonstration if API is unavailable
        logging.info("Creating sample data for demonstration")
        
        # Sample financial texts (replace with more realistic examples if needed)
        sample_texts = [
            "Quarterly financial report shows 15% increase in revenue",
            "Investment opportunity with guaranteed 50% returns in 3 months",
            "Please update your banking details by clicking this link",
            "Transfer $5000 to the specified account for verification purposes",
            "Our company achieved record profits this quarter",
            "Urgent: Your account has been compromised, send details immediately",
            "Dividend payment of $2.50 per share announced",
            "Exclusive investment opportunity, limited time offer",
            "Annual report shows stable growth across all business segments",
            "Your account requires verification, please send your credentials"
        ]
        
        # Create DataFrame with sample texts
        df = pd.DataFrame({
            "text": sample_texts,
            "source": ["sample"] * len(sample_texts),
            "date": ["2025-01-01"] * len(sample_texts)
        })
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logging.info(f"Created sample data with {len(df)} examples at {output_file}")
        
        return output_file

def preprocess_fineweb_data(data_file, tokenizer, max_length=128):
    """
    Preprocess FineWeb data for classification
    
    Args:
        data_file (str): Path to the FineWeb data file
        tokenizer: Tokenizer for the model
        max_length (int): Maximum sequence length
        
    Returns:
        datasets.Dataset: Preprocessed dataset
    """
    logging.info(f"Preprocessing FineWeb data from {data_file}")
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Ensure text column exists
    if "text" not in df.columns:
        if "content" in df.columns:
            df["text"] = df["content"]
        elif "description" in df.columns:
            df["text"] = df["description"]
        else:
            raise ValueError("No text column found in the data")
    
    # Preprocess text
    df["processed_text"] = df["text"].apply(preprocess_text)
    
    # Create dataset
    dataset = Dataset.from_pandas(df)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["processed_text"],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Format for model
    tokenized_dataset = tokenized_dataset.remove_columns(
        [col for col in tokenized_dataset.column_names if col not in ["input_ids", "attention_mask", "token_type_ids"]]
    )
    
    tokenized_dataset.set_format("torch")
    
    return tokenized_dataset, df

def classify_fineweb_data(model_path, data_file, output_dir, batch_size=16, device=None):
    """
    Classify FineWeb data using the trained TinyBERT model
    
    Args:
        model_path (str): Path to the trained model
        data_file (str): Path to the FineWeb data file
        output_dir (str): Directory to save the classification results
        batch_size (int): Batch size for inference
        device (str): Device to use for inference (cpu or cuda)
        
    Returns:
        str: Path to the classification results file
    """
    logging.info(f"Classifying FineWeb data using model from {model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and tokenizer
    model = TinyBertForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Preprocess data
    dataset, original_df = preprocess_fineweb_data(data_file, tokenizer)
    
    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    # Perform inference
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Classifying"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Get predictions
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            # Move to CPU and convert to numpy
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Add predictions to original data
    original_df["prediction"] = all_predictions
    original_df["fraud_probability"] = [prob[1] for prob in all_probabilities]
    
    # Map predictions to labels
    original_df["prediction_label"] = original_df["prediction"].map({0: "Non-Fraud", 1: "Fraud"})
    
    # Save results
    results_file = os.path.join(output_dir, "fineweb_classification_results.csv")
    original_df.to_csv(results_file, index=False)
    
    # Generate summary
    fraud_count = original_df["prediction"].sum()
    total_count = len(original_df)
    fraud_percentage = (fraud_count / total_count) * 100
    
    logging.info(f"Classification complete. Found {fraud_count} potential fraud cases out of {total_count} ({fraud_percentage:.2f}%)")
    
    # Generate visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(original_df["fraud_probability"], bins=20, kde=True)
    plt.title("Distribution of Fraud Probabilities")
    plt.xlabel("Fraud Probability")
    plt.ylabel("Count")
    plt.savefig(os.path.join(output_dir, "fraud_probability_distribution.png"))
    
    # Generate high-risk list
    high_risk_threshold = 0.7
    high_risk_df = original_df[original_df["fraud_probability"] >= high_risk_threshold]
    high_risk_file = os.path.join(output_dir, "high_risk_fraud_cases.csv")
    high_risk_df.to_csv(high_risk_file, index=False)
    
    logging.info(f"Identified {len(high_risk_df)} high-risk cases (probability >= {high_risk_threshold})")
    logging.info(f"Results saved to {results_file}")
    
    return results_file

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Classify FineWeb financial data for fraud detection")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained TinyBERT model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save/load FineWeb data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save classification results")
    parser.add_argument("--category", type=str, default="finance", help="FineWeb data category")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of samples to download")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use for inference")
    
    args = parser.parse_args()
    
    # Download FineWeb data
    data_file = download_fineweb_data(args.data_dir, args.category, args.limit)
    
    # Classify data
    classify_fineweb_data(args.model_path, data_file, args.output_dir, args.batch_size, args.device)

if __name__ == "__main__":
    main()
