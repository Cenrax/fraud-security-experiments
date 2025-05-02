#!/usr/bin/env python
import os
import argparse
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Financial Fraud Detection using TinyBERT")
    parser.add_argument("--action", type=str, required=True, choices=["prepare", "train", "evaluate", "all"],
                        help="Action to perform: prepare data, train model, evaluate model, or all")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory for storing data")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory for storing output models and results")
    parser.add_argument("--use_agno", action="store_true", 
                        help="Use Agno for data generation")
    parser.add_argument("--use_openai", action="store_true",
                        help="Use OpenAI GPT-4.1-mini for data generation")
    parser.add_argument("--fraud_samples", type=int, default=1000,
                        help="Number of fraud samples to generate")
    parser.add_argument("--non_fraud_samples", type=int, default=10000,
                        help="Number of non-fraud samples to generate")
    parser.add_argument("--teacher_epochs", type=int, default=3,
                        help="Number of epochs for teacher model training")
    parser.add_argument("--general_distill_epochs", type=int, default=5,
                        help="Number of epochs for general distillation")
    parser.add_argument("--task_distill_epochs", type=int, default=5,
                        help="Number of epochs for task-specific distillation")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Execute the requested action
    if args.action in ["prepare", "all"]:
        logger.info("Preparing data...")
        from scripts.prepare_data import download_financial_fraud_data
        
        download_financial_fraud_data(
            output_dir=args.data_dir,
            num_fraud_samples=args.fraud_samples,
            num_non_fraud_samples=args.non_fraud_samples,
            use_agno=args.use_agno,
            use_openai=args.use_openai
        )
    
    if args.action in ["train", "all"]:
        logger.info("Training models...")
        from scripts.train import train_teacher_model, train_tinybert_general_distillation, train_tinybert_task_specific_distillation
        
        # Create a namespace object with the required arguments for training
        train_args = argparse.Namespace(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            bert_model_name="bert-base-uncased",
            tinybert_layers=4,
            tinybert_hidden_size=312,
            tinybert_intermediate_size=1200,
            batch_size=args.batch_size,
            teacher_epochs=args.teacher_epochs,
            general_distill_epochs=args.general_distill_epochs,
            task_distill_epochs=args.task_distill_epochs,
            teacher_learning_rate=2e-5,
            student_learning_rate=1e-4,
            task_learning_rate=5e-5,
            weight_decay=0.01,
            max_grad_norm=1.0,
            temperature=1.0,
            seed=args.seed,
            max_seq_length=512,
            use_wandb=False
        )
        
        # Train teacher model
        logger.info("Training teacher model...")
        teacher_model_path = train_teacher_model(train_args)
        
        # Train TinyBERT with general distillation
        logger.info("Training TinyBERT with general distillation...")
        general_distill_model_path = train_tinybert_general_distillation(train_args, teacher_model_path)
        
        # Train TinyBERT with task-specific distillation
        logger.info("Training TinyBERT with task-specific distillation...")
        task_distill_model_path = train_tinybert_task_specific_distillation(train_args, general_distill_model_path, teacher_model_path)
    
    if args.action in ["evaluate", "all"]:
        logger.info("Evaluating models...")
        import subprocess
        
        # Find the model paths
        teacher_model_path = os.path.join(args.output_dir, 'teacher_model')
        student_model_path = os.path.join(args.output_dir, 'tinybert_task_distill')
        
        # Check if the model paths exist
        if not os.path.exists(teacher_model_path) or not os.path.exists(student_model_path):
            logger.error("Model paths not found. Please train the models first.")
            return
        
        # Run the evaluation script
        cmd = [
            "python", "scripts/evaluate.py",
            "--data_dir", args.data_dir,
            "--output_dir", os.path.join(args.output_dir, "evaluation"),
            "--teacher_model_path", teacher_model_path,
            "--student_model_path", student_model_path,
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed)
        ]
        
        subprocess.run(cmd)
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
