import os
import random
import pandas as pd
import torch
import re
import string
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from nltk.corpus import wordnet
import nltk
from tqdm import tqdm

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class FinancialFraudDataset(Dataset):
    """
    Dataset for financial fraud detection
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """
        Args:
            texts (list): List of text samples
            labels (list): List of labels (1 for fraud, 0 for non-fraud)
            tokenizer: Tokenizer for encoding texts
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_financial_fraud_data(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Load financial fraud data from the data directory
    
    Args:
        data_dir (str): Path to data directory
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        test_ratio (float): Ratio of test data
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary containing train, val, and test data
    """
    random.seed(seed)
    
    # Load fraud data (positive samples)
    fraud_data = pd.read_csv(os.path.join(data_dir, 'fraud_data.csv'))
    fraud_texts = fraud_data['text'].tolist()
    fraud_labels = [1] * len(fraud_texts)
    
    # Load non-fraud data (negative samples)
    non_fraud_data = pd.read_csv(os.path.join(data_dir, 'non_fraud_data.csv'))
    non_fraud_texts = non_fraud_data['text'].tolist()
    non_fraud_labels = [0] * len(non_fraud_texts)
    
    # Combine data
    all_texts = fraud_texts + non_fraud_texts
    all_labels = fraud_labels + non_fraud_labels
    
    # Shuffle data
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    
    # Split data
    total_samples = len(all_texts)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    
    train_texts = all_texts[:train_size]
    train_labels = all_labels[:train_size]
    
    val_texts = all_texts[train_size:train_size + val_size]
    val_labels = all_labels[train_size:train_size + val_size]
    
    test_texts = all_texts[train_size + val_size:]
    test_labels = all_labels[train_size + val_size:]
    
    return {
        'train': (train_texts, train_labels),
        'val': (val_texts, val_labels),
        'test': (test_texts, test_labels)
    }

def create_dataloaders(data_splits, tokenizer, batch_size=16, max_length=512):
    """
    Create DataLoader objects for training, validation, and testing
    
    Args:
        data_splits (dict): Dictionary containing train, val, and test data
        tokenizer: Tokenizer for encoding texts
        batch_size (int): Batch size
        max_length (int): Maximum sequence length
        
    Returns:
        dict: Dictionary containing train, val, and test DataLoader objects
    """
    dataloaders = {}
    
    for split_name, (texts, labels) in data_splits.items():
        dataset = FinancialFraudDataset(texts, labels, tokenizer, max_length)
        shuffle = split_name == 'train'
        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2
        )
    
    return dataloaders

def augment_text(text, tokenizer, bert_model, glove_embeddings, augmentation_prob=0.15):
    """
    Augment text using the TinyBERT data augmentation technique
    
    Args:
        text (str): Input text
        tokenizer: Tokenizer for encoding text
        bert_model: BERT model for word prediction
        glove_embeddings: GloVe embeddings for word replacement
        augmentation_prob (float): Probability of augmenting a word
        
    Returns:
        str: Augmented text
    """
    words = text.split()
    augmented_words = []
    
    for word in words:
        if random.random() < augmentation_prob:
            # Case 1: Word is tokenized as a single token
            tokens = tokenizer.tokenize(word)
            if len(tokens) == 1:
                # Use BERT to predict a replacement
                inputs = tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = bert_model(**inputs)
                    predictions = outputs.logits
                
                # Get token index
                word_idx = text.find(word)
                if word_idx != -1:
                    token_idx = tokenizer.encode(text[:word_idx], add_special_tokens=False)
                    token_idx = len(token_idx)
                    
                    # Get top predictions
                    top_predictions = torch.topk(predictions[0, token_idx], 5).indices
                    replacement_token = tokenizer.decode(top_predictions[random.randint(0, 4)].item())
                    augmented_words.append(replacement_token)
                else:
                    augmented_words.append(word)
            
            # Case 2: Word is tokenized into multiple tokens
            else:
                # Find similar word using GloVe embeddings
                synonyms = []
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
                
                if synonyms:
                    augmented_words.append(random.choice(synonyms))
                else:
                    augmented_words.append(word)
        else:
            augmented_words.append(word)
    
    return ' '.join(augmented_words)

def preprocess_text(text):
    """
    Preprocess text for financial fraud detection
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    if not isinstance(text, str):
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '[EMAIL]', text)
    
    # Replace phone numbers with placeholder
    text = re.sub(r'\+?[0-9][\s-]?\(?[0-9]{3}\)?[\s-]?[0-9]{3}[\s-]?[0-9]{4}', '[PHONE]', text)
    
    # Replace account numbers and credit card numbers with placeholder
    text = re.sub(r'[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}', '[ACCOUNT]', text)
    
    # Replace dollar amounts with placeholder
    text = re.sub(r'\$\s*\d+(?:\.\d+)?', '[AMOUNT]', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def prepare_fineweb_data(fineweb_data_path, output_dir, bert_model, tokenizer, threshold=0.003):
    """
    Prepare FineWeb data for training
    
    Args:
        fineweb_data_path (str): Path to FineWeb data
        output_dir (str): Output directory for processed data
        bert_model: BERT model for classification
        tokenizer: Tokenizer for encoding texts
        threshold (float): Threshold for filtering data
        
    Returns:
        tuple: Paths to processed fraud and non-fraud data files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load FineWeb data
    fineweb_data = pd.read_csv(fineweb_data_path)
    
    # Classify each text
    fraud_texts = []
    non_fraud_texts = []
    
    for text in tqdm(fineweb_data['text']):
        # Preprocess text
        processed_text = preprocess_text(text)
        
        inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            logits = outputs.logits
            score = torch.softmax(logits, dim=1)[0, 1].item()  # Score for fraud class
        
        if score >= threshold:
            fraud_texts.append(text)
        else:
            non_fraud_texts.append(text)
    
    # Save processed data
    fraud_df = pd.DataFrame({'text': fraud_texts})
    non_fraud_df = pd.DataFrame({'text': non_fraud_texts})
    
    fraud_path = os.path.join(output_dir, 'fraud_data.csv')
    non_fraud_path = os.path.join(output_dir, 'non_fraud_data.csv')
    
    fraud_df.to_csv(fraud_path, index=False)
    non_fraud_df.to_csv(non_fraud_path, index=False)
    
    return fraud_path, non_fraud_path
