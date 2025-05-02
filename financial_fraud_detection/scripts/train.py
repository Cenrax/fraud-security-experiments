import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import AdamW
from transformers import (
    BertForSequenceClassification, 
    BertTokenizer, 
    BertConfig,
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter
import sys
import logging
import wandb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tinybert import (
    TinyBertForSequenceClassification, 
    TinyBertDistillationLoss,
    create_tinybert_config
)
from utils.data_utils import (
    load_financial_fraud_data, 
    create_dataloaders, 
    augment_text
)

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

def train_teacher_model(args):
    """
    Train a BERT model for financial fraud detection
    
    Args:
        args: Command-line arguments
        
    Returns:
        str: Path to the saved teacher model
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
    
    # Load data
    data_splits = load_financial_fraud_data(
        args.data_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_splits,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length
    )
    
    # Initialize model
    config = BertConfig.from_pretrained(
        args.bert_model_name,
        num_labels=2
    )
    model = BertForSequenceClassification.from_pretrained(
        args.bert_model_name,
        config=config
    )
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.teacher_learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(dataloaders['train']) * args.teacher_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'teacher_logs'))
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project="financial-fraud-detection",
            name="teacher-training",
            config=vars(args)
        )
    
    # Training loop
    best_val_accuracy = 0.0
    best_model_path = os.path.join(args.output_dir, 'teacher_model')
    os.makedirs(best_model_path, exist_ok=True)
    
    for epoch in range(args.teacher_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{args.teacher_epochs} [Train]"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(logits, dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        # Calculate training metrics
        train_loss = train_loss / len(dataloaders['train'])
        train_accuracy = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{args.teacher_epochs} [Val]"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate validation metrics
        val_loss = val_loss / len(dataloaders['val'])
        val_accuracy = val_correct / val_total
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.teacher_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Write to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)
            logger.info(f"Saved best model with validation accuracy: {val_accuracy:.4f}")
    
    # Close tensorboard writer
    writer.close()
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    
    return best_model_path

def train_tinybert_general_distillation(args, teacher_model_path):
    """
    Train TinyBERT with general distillation
    
    Args:
        args: Command-line arguments
        teacher_model_path: Path to the saved teacher model
        
    Returns:
        str: Path to the saved TinyBERT model after general distillation
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load teacher model and tokenizer
    teacher_tokenizer = BertTokenizer.from_pretrained(teacher_model_path)
    teacher_config = BertConfig.from_pretrained(teacher_model_path)
    teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_path)
    teacher_model.to(device)
    teacher_model.eval()
    
    # Create TinyBERT config and model
    tinybert_config = create_tinybert_config(
        teacher_config,
        num_hidden_layers=args.tinybert_layers,
        hidden_size=args.tinybert_hidden_size,
        intermediate_size=args.tinybert_intermediate_size
    )
    student_model = TinyBertForSequenceClassification(tinybert_config)
    student_model.to(device)
    
    # Load data
    data_splits = load_financial_fraud_data(
        args.data_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_splits,
        teacher_tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length
    )
    
    # Initialize distillation loss
    distillation_loss_fn = TinyBertDistillationLoss(temperature=args.temperature)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        student_model.parameters(),
        lr=args.student_learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(dataloaders['train']) * args.general_distill_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'general_distill_logs'))
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project="financial-fraud-detection",
            name="tinybert-general-distillation",
            config=vars(args)
        )
    
    # Training loop
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'tinybert_general_distill')
    os.makedirs(best_model_path, exist_ok=True)
    
    for epoch in range(args.general_distill_epochs):
        # Training
        student_model.train()
        train_loss = 0.0
        
        for batch in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{args.general_distill_epochs} [Train]"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            # Get teacher outputs
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True,
                    output_hidden_states=True
                )
                
                teacher_logits = teacher_outputs.logits
                teacher_attentions = teacher_outputs.attentions
                teacher_hidden_states = teacher_outputs.hidden_states
                teacher_embedding_output = teacher_model.bert.embeddings(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )
            
            # Get student outputs
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True,
                output_hidden_states=True
            )
            
            student_logits = student_outputs.logits
            student_attentions = student_outputs.attentions
            student_hidden_states = student_outputs.hidden_states
            student_embedding_output = student_model.bert.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
            
            # Calculate distillation loss
            loss_dict = distillation_loss_fn(
                {
                    'embedding_output': student_embedding_output,
                    'attention_outputs': student_attentions,
                    'hidden_states': student_hidden_states,
                    'logits': student_logits
                },
                {
                    'embedding_output': teacher_embedding_output,
                    'attention_outputs': teacher_attentions,
                    'hidden_states': teacher_hidden_states,
                    'logits': teacher_logits
                },
                attention_mask
            )
            
            loss = loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
        
        # Calculate training metrics
        train_loss = train_loss / len(dataloaders['train'])
        
        # Validation
        student_model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{args.general_distill_epochs} [Val]"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                
                # Get teacher outputs
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True,
                    output_hidden_states=True
                )
                
                teacher_logits = teacher_outputs.logits
                teacher_attentions = teacher_outputs.attentions
                teacher_hidden_states = teacher_outputs.hidden_states
                teacher_embedding_output = teacher_model.bert.embeddings(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )
                
                # Get student outputs
                student_outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True,
                    output_hidden_states=True
                )
                
                student_logits = student_outputs.logits
                student_attentions = student_outputs.attentions
                student_hidden_states = student_outputs.hidden_states
                student_embedding_output = student_model.bert.embeddings(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )
                
                # Calculate distillation loss
                loss_dict = distillation_loss_fn(
                    {
                        'embedding_output': student_embedding_output,
                        'attention_outputs': student_attentions,
                        'hidden_states': student_hidden_states,
                        'logits': student_logits
                    },
                    {
                        'embedding_output': teacher_embedding_output,
                        'attention_outputs': teacher_attentions,
                        'hidden_states': teacher_hidden_states,
                        'logits': teacher_logits
                    },
                    attention_mask
                )
                
                loss = loss_dict['total_loss']
                
                # Update metrics
                val_loss += loss.item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(dataloaders['val'])
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.general_distill_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Write to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss
            })
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            student_model.save_pretrained(best_model_path)
            teacher_tokenizer.save_pretrained(best_model_path)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # Close tensorboard writer
    writer.close()
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    
    return best_model_path

def train_tinybert_task_specific_distillation(args, general_distill_model_path, teacher_model_path):
    """
    Train TinyBERT with task-specific distillation
    
    Args:
        args: Command-line arguments
        general_distill_model_path: Path to the TinyBERT model after general distillation
        teacher_model_path: Path to the saved teacher model
        
    Returns:
        str: Path to the saved TinyBERT model after task-specific distillation
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load teacher model and tokenizer
    teacher_tokenizer = BertTokenizer.from_pretrained(teacher_model_path)
    teacher_model = BertForSequenceClassification.from_pretrained(teacher_model_path)
    teacher_model.to(device)
    teacher_model.eval()
    
    # Load student model
    student_model = TinyBertForSequenceClassification.from_pretrained(general_distill_model_path)
    student_model.to(device)
    
    # Load data
    data_splits = load_financial_fraud_data(
        args.data_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=args.seed
    )
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        data_splits,
        teacher_tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_seq_length
    )
    
    # Initialize distillation loss
    distillation_loss_fn = TinyBertDistillationLoss(temperature=args.temperature)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        student_model.parameters(),
        lr=args.task_learning_rate,
        weight_decay=args.weight_decay
    )
    
    total_steps = len(dataloaders['train']) * args.task_distill_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )
    
    # Initialize tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'task_distill_logs'))
    
    # Initialize wandb if enabled
    if args.use_wandb:
        wandb.init(
            project="financial-fraud-detection",
            name="tinybert-task-distillation",
            config=vars(args)
        )
    
    # Training loop
    best_val_accuracy = 0.0
    best_model_path = os.path.join(args.output_dir, 'tinybert_task_distill')
    os.makedirs(best_model_path, exist_ok=True)
    
    for epoch in range(args.task_distill_epochs):
        # Training
        student_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(dataloaders['train'], desc=f"Epoch {epoch+1}/{args.task_distill_epochs} [Train]"):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Get teacher outputs
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True,
                    output_hidden_states=True
                )
                
                teacher_logits = teacher_outputs.logits
                teacher_attentions = teacher_outputs.attentions
                teacher_hidden_states = teacher_outputs.hidden_states
                teacher_embedding_output = teacher_model.bert.embeddings(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids
                )
            
            # Get student outputs
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                output_attentions=True,
                output_hidden_states=True
            )
            
            student_loss = student_outputs.loss
            student_logits = student_outputs.logits
            student_attentions = student_outputs.attentions
            student_hidden_states = student_outputs.hidden_states
            student_embedding_output = student_model.bert.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids
            )
            
            # Calculate distillation loss
            distill_loss_dict = distillation_loss_fn(
                {
                    'embedding_output': student_embedding_output,
                    'attention_outputs': student_attentions,
                    'hidden_states': student_hidden_states,
                    'logits': student_logits
                },
                {
                    'embedding_output': teacher_embedding_output,
                    'attention_outputs': teacher_attentions,
                    'hidden_states': teacher_hidden_states,
                    'logits': teacher_logits
                },
                attention_mask
            )
            
            # Combine task loss and distillation loss
            loss = student_loss + distill_loss_dict['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = torch.max(student_logits, dim=1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        
        # Calculate training metrics
        train_loss = train_loss / len(dataloaders['train'])
        train_accuracy = train_correct / train_total
        
        # Validation
        student_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloaders['val'], desc=f"Epoch {epoch+1}/{args.task_distill_epochs} [Val]"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                logits = outputs.logits
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = torch.max(logits, dim=1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate validation metrics
        val_loss = val_loss / len(dataloaders['val'])
        val_accuracy = val_correct / val_total
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.task_distill_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Write to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # Log to wandb if enabled
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            })
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            student_model.save_pretrained(best_model_path)
            teacher_tokenizer.save_pretrained(best_model_path)
            logger.info(f"Saved best model with validation accuracy: {val_accuracy:.4f}")
    
    # Close tensorboard writer
    writer.close()
    
    # Finish wandb run if enabled
    if args.use_wandb:
        wandb.finish()
    
    return best_model_path

def main():
    parser = argparse.ArgumentParser(description="Train TinyBERT for financial fraud detection")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    
    # Model arguments
    parser.add_argument("--bert_model_name", type=str, default="bert-base-uncased", help="BERT model name")
    parser.add_argument("--tinybert_layers", type=int, default=4, help="Number of layers in TinyBERT")
    parser.add_argument("--tinybert_hidden_size", type=int, default=312, help="Hidden size in TinyBERT")
    parser.add_argument("--tinybert_intermediate_size", type=int, default=1200, help="Intermediate size in TinyBERT")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--teacher_epochs", type=int, default=3, help="Number of epochs for teacher training")
    parser.add_argument("--general_distill_epochs", type=int, default=5, help="Number of epochs for general distillation")
    parser.add_argument("--task_distill_epochs", type=int, default=5, help="Number of epochs for task-specific distillation")
    parser.add_argument("--teacher_learning_rate", type=float, default=2e-5, help="Learning rate for teacher training")
    parser.add_argument("--student_learning_rate", type=float, default=1e-4, help="Learning rate for general distillation")
    parser.add_argument("--task_learning_rate", type=float, default=5e-5, help="Learning rate for task-specific distillation")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for distillation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train teacher model
    logger.info("Training teacher model...")
    teacher_model_path = train_teacher_model(args)
    logger.info(f"Teacher model saved to {teacher_model_path}")
    
    # Train TinyBERT with general distillation
    logger.info("Training TinyBERT with general distillation...")
    general_distill_model_path = train_tinybert_general_distillation(args, teacher_model_path)
    logger.info(f"TinyBERT general distillation model saved to {general_distill_model_path}")
    
    # Train TinyBERT with task-specific distillation
    logger.info("Training TinyBERT with task-specific distillation...")
    task_distill_model_path = train_tinybert_task_specific_distillation(args, general_distill_model_path, teacher_model_path)
    logger.info(f"TinyBERT task-specific distillation model saved to {task_distill_model_path}")
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
