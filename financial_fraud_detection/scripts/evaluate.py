import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import BertTokenizer
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tinybert import TinyBertForSequenceClassification
from utils.data_utils import load_financial_fraud_data, create_dataloaders

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_model(model, dataloader, device):
    """
    Evaluate model performance on a dataset
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs.logits
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Add to lists
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    # Return metrics
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'auc': auc,
        'labels': all_labels,
        'predictions': all_preds,
        'probabilities': all_probs
    }

def plot_metrics(metrics, output_dir):
    """
    Plot evaluation metrics
    
    Args:
        metrics: Dictionary containing evaluation metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    conf_matrix = metrics['confusion_matrix']
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Non-Fraud', 'Fraud']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = sklearn.metrics.roc_curve(metrics['labels'], metrics['probabilities'])
    plt.plot(fpr, tpr, label=f'AUC = {metrics["auc"]:.3f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    precision, recall, _ = precision_recall_curve(metrics['labels'], metrics['probabilities'])
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'))

def compare_models(teacher_metrics, student_metrics, output_dir):
    """
    Compare teacher and student model performance
    
    Args:
        teacher_metrics: Dictionary containing teacher model metrics
        student_metrics: Dictionary containing student model metrics
        output_dir: Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison table
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    comparison = {
        'Metric': metrics,
        'Teacher': [teacher_metrics[m] for m in metrics],
        'TinyBERT': [student_metrics[m] for m in metrics]
    }
    
    df = pd.DataFrame(comparison)
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Plot comparison bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, [teacher_metrics[m] for m in metrics], width, label='Teacher')
    plt.bar(x + width/2, [student_metrics[m] for m in metrics], width, label='TinyBERT')
    
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    
    # Print comparison summary
    logger.info("\nModel Performance Comparison:")
    logger.info(f"{'Metric':<10} {'Teacher':<10} {'TinyBERT':<10} {'Difference':<10}")
    logger.info("-" * 40)
    
    for metric in metrics:
        teacher_val = teacher_metrics[metric]
        student_val = student_metrics[metric]
        diff = student_val - teacher_val
        diff_str = f"{diff:.4f} ({diff/teacher_val*100:.1f}%)"
        logger.info(f"{metric:<10} {teacher_val:.4f}    {student_val:.4f}    {diff_str}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyBERT for financial fraud detection")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--teacher_model_path", type=str, required=True, help="Path to teacher model")
    parser.add_argument("--student_model_path", type=str, required=True, help="Path to student model")
    
    # Evaluation arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.teacher_model_path)
    
    # Load data
    data_splits = load_financial_fraud_data(
        args.data_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )
    
    # Create test dataloader
    dataloaders = create_dataloaders(
        {'test': data_splits['test']},
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length
    )
    test_dataloader = dataloaders['test']
    
    # Load teacher model
    logger.info(f"Loading teacher model from {args.teacher_model_path}")
    teacher_model = BertForSequenceClassification.from_pretrained(args.teacher_model_path)
    teacher_model.to(device)
    
    # Load student model
    logger.info(f"Loading student model from {args.student_model_path}")
    student_model = TinyBertForSequenceClassification.from_pretrained(args.student_model_path)
    student_model.to(device)
    
    # Evaluate teacher model
    logger.info("Evaluating teacher model...")
    teacher_metrics = evaluate_model(teacher_model, test_dataloader, device)
    
    # Evaluate student model
    logger.info("Evaluating student model...")
    student_metrics = evaluate_model(student_model, test_dataloader, device)
    
    # Plot metrics
    logger.info("Plotting teacher model metrics...")
    plot_metrics(teacher_metrics, os.path.join(args.output_dir, 'teacher'))
    
    logger.info("Plotting student model metrics...")
    plot_metrics(student_metrics, os.path.join(args.output_dir, 'student'))
    
    # Compare models
    logger.info("Comparing models...")
    compare_models(teacher_metrics, student_metrics, args.output_dir)
    
    # Calculate model size reduction
    teacher_size = sum(p.numel() for p in teacher_model.parameters())
    student_size = sum(p.numel() for p in student_model.parameters())
    size_reduction = (teacher_size - student_size) / teacher_size * 100
    
    logger.info(f"\nModel Size Comparison:")
    logger.info(f"Teacher model parameters: {teacher_size:,}")
    logger.info(f"TinyBERT model parameters: {student_size:,}")
    logger.info(f"Size reduction: {size_reduction:.2f}%")
    
    # Save model size comparison
    with open(os.path.join(args.output_dir, 'model_size.txt'), 'w') as f:
        f.write(f"Teacher model parameters: {teacher_size:,}\n")
        f.write(f"TinyBERT model parameters: {student_size:,}\n")
        f.write(f"Size reduction: {size_reduction:.2f}%\n")
    
    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()
