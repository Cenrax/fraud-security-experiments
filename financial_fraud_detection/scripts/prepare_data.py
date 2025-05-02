import os
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
import sys
import logging
import random
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def download_fineweb_sample(output_path, sample_size=10000):
    """
    Download a sample of FineWeb data for processing
    This is a placeholder function - in a real implementation,
    you would need to access the actual FineWeb dataset
    
    Args:
        output_path (str): Path to save the sample data
        sample_size (int): Number of samples to download
        
    Returns:
        str: Path to the downloaded sample
    """
    logger.info(f"Downloading {sample_size} samples from FineWeb...")
    
    # This is a placeholder - in a real implementation, you would
    # download actual FineWeb data or use a local copy
    
    # Generate synthetic data for demonstration
    texts = []
    for i in range(sample_size):
        # Generate random text with varying length
        length = random.randint(50, 500)
        text = ' '.join([f"word{random.randint(1, 5000)}" for _ in range(length)])
        texts.append(text)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({'text': texts})
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample data saved to {output_path}")
    return output_path

def download_financial_fraud_data(output_dir, num_fraud_samples=1000, num_non_fraud_samples=10000, use_agno=True, use_openai=True):
    """
    Generate financial fraud data for training using Agno and GPT-4.1-mini
    
    Args:
        output_dir (str): Directory to save the data
        num_fraud_samples (int): Number of fraud samples to generate
        num_non_fraud_samples (int): Number of non-fraud samples to generate
        use_agno (bool): Whether to use Agno for data generation
        use_openai (bool): Whether to use OpenAI GPT-4.1-mini for data generation
        
    Returns:
        tuple: Paths to fraud and non-fraud data files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define fraud categories and keywords for prompting
    fraud_categories = {
        'identity_theft': [
            'unauthorized access', 'identity theft', 'personal information stolen',
            'social security number compromised', 'account takeover', 'synthetic identity'
        ],
        'payment_fraud': [
            'unauthorized transaction', 'fraudulent charge', 'card-not-present fraud',
            'chargeback fraud', 'payment manipulation', 'wire fraud'
        ],
        'money_laundering': [
            'money laundering', 'layering transactions', 'shell companies',
            'smurfing', 'trade-based laundering', 'cryptocurrency laundering'
        ],
        'investment_fraud': [
            'ponzi scheme', 'pyramid scheme', 'pump and dump', 
            'advance fee fraud', 'boiler room', 'market manipulation'
        ],
        'phishing_scams': [
            'phishing attempt', 'spear phishing', 'business email compromise',
            'fake invoice', 'CEO fraud', 'invoice manipulation'
        ]
    }
    
    # Initialize lists for fraud and non-fraud texts
    fraud_texts = []
    non_fraud_texts = []
    
    try:
        if use_agno:
            logger.info("Initializing Agno for financial fraud data generation...")
            
            try:
                from agno.agent import Agent
                from agno.models.openai import OpenAIChat
                
                # Create an Agno agent for generating financial fraud data
                fraud_agent = Agent(
                    model=OpenAIChat(id="gpt-4.1-mini" if use_openai else "gpt-4o-mini"),
                    description="You are a financial fraud expert who can generate realistic examples of financial fraud text data.",
                    instructions=[
                        "Generate realistic examples of financial fraud text that might appear in financial documents, emails, or transaction descriptions.",
                        "Make the examples diverse, realistic, and subtle - not all fraud is obvious.",
                        "Include technical financial terminology where appropriate.",
                        "Each example should be 3-5 sentences long.",
                        "Do not include any disclaimers or explanations in your response, just the fraud text itself."
                    ],
                    markdown=False
                )
                
                # Create an Agno agent for generating non-fraud financial data
                non_fraud_agent = Agent(
                    model=OpenAIChat(id="gpt-4.1-mini" if use_openai else "gpt-4o-mini"),
                    description="You are a financial expert who can generate realistic examples of legitimate financial text data.",
                    instructions=[
                        "Generate realistic examples of legitimate financial text that might appear in financial documents, emails, or transaction descriptions.",
                        "Make the examples diverse and realistic, covering various aspects of normal financial operations.",
                        "Include technical financial terminology where appropriate.",
                        "Each example should be 3-5 sentences long.",
                        "Do not include any disclaimers or explanations in your response, just the legitimate financial text itself."
                    ],
                    markdown=False
                )
                
                # Generate fraud data samples
                logger.info(f"Generating {num_fraud_samples} fraud samples using Agno...")
                for i in tqdm(range(num_fraud_samples)):
                    # Select a random fraud category
                    category = random.choice(list(fraud_categories.keys()))
                    keywords = fraud_categories[category]
                    
                    # Create a prompt with specific fraud category and keywords
                    prompt = f"Generate a realistic example of {category.replace('_', ' ')} fraud. Include at least one of these terms if appropriate: {', '.join(random.sample(keywords, min(3, len(keywords))))}. Make it subtle and realistic."
                    
                    # Generate the fraud text
                    response = fraud_agent.run(prompt)
                    if response:
                        # Extract text from RunResponse object
                        if hasattr(response, 'content') and response.content is not None:
                            # New Agno API returns content attribute
                            text = response.content
                        elif hasattr(response, 'response') and response.response is not None:
                            # Older Agno API might return response attribute
                            text = response.response
                        elif isinstance(response, str):
                            # Direct string response
                            text = response
                        else:
                            # Try converting to string
                            text = str(response)
                            
                        # Clean the response
                        text = text.strip() if isinstance(text, str) else ""
                        if text:
                            fraud_texts.append(text)
                
                # Generate non-fraud data samples
                logger.info(f"Generating {num_non_fraud_samples} non-fraud samples using Agno...")
                for i in tqdm(range(num_non_fraud_samples)):
                    # Create a prompt for legitimate financial text
                    financial_topics = [
                        "quarterly financial report", "investment strategy", "transaction processing",
                        "account statement", "financial planning", "budget allocation",
                        "audit procedure", "compliance report", "risk assessment"
                    ]
                    topic = random.choice(financial_topics)
                    prompt = f"Generate a realistic example of legitimate financial text about {topic}. Make it detailed and professional."
                    
                    # Generate the non-fraud text
                    response = non_fraud_agent.run(prompt)
                    if response:
                        # Extract text from RunResponse object
                        if hasattr(response, 'content') and response.content is not None:
                            # New Agno API returns content attribute
                            text = response.content
                        elif hasattr(response, 'response') and response.response is not None:
                            # Older Agno API might return response attribute
                            text = response.response
                        elif isinstance(response, str):
                            # Direct string response
                            text = response
                        else:
                            # Try converting to string
                            text = str(response)
                            
                        # Clean the response
                        text = text.strip() if isinstance(text, str) else ""
                        if text:
                            non_fraud_texts.append(text)
                
            except ImportError:
                logger.warning("Agno not installed. Falling back to traditional generation method.")
                use_agno = False
            except Exception as e:
                logger.warning(f"Error using Agno: {str(e)}. Falling back to traditional generation method.")
                use_agno = False
        
        # If Agno is not used or failed, fall back to OpenAI direct API
        if not use_agno and use_openai:
            logger.info("Using OpenAI API directly for data generation...")
            
            try:
                import openai
                
                # Check if OPENAI_API_KEY is set
                if not os.environ.get('OPENAI_API_KEY'):
                    logger.warning("OPENAI_API_KEY environment variable not set. Falling back to synthetic data generation.")
                    raise ValueError("OPENAI_API_KEY not set")
                
                client = openai.OpenAI()
                
                # Generate fraud data samples
                logger.info(f"Generating {num_fraud_samples} fraud samples using OpenAI API...")
                for i in tqdm(range(num_fraud_samples)):
                    # Select a random fraud category
                    category = random.choice(list(fraud_categories.keys()))
                    keywords = fraud_categories[category]
                    
                    # Create a system message with specific fraud category and keywords
                    system_message = f"You are a financial fraud expert. Generate a realistic example of {category.replace('_', ' ')} fraud. Include at least one of these terms if appropriate: {', '.join(random.sample(keywords, min(3, len(keywords))))}. Make it subtle and realistic. The text should be 3-5 sentences long. Provide only the fraud text, no explanations or disclaimers."
                    
                    # Generate the fraud text
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": "Generate a realistic financial fraud text example."}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    
                    if response.choices and response.choices[0].message.content:
                        text = response.choices[0].message.content.strip()
                        if text:
                            fraud_texts.append(text)
                
                # Generate non-fraud data samples
                logger.info(f"Generating {num_non_fraud_samples} non-fraud samples using OpenAI API...")
                for i in tqdm(range(num_non_fraud_samples)):
                    # Create a system message for legitimate financial text
                    financial_topics = [
                        "quarterly financial report", "investment strategy", "transaction processing",
                        "account statement", "financial planning", "budget allocation",
                        "audit procedure", "compliance report", "risk assessment"
                    ]
                    topic = random.choice(financial_topics)
                    system_message = f"You are a financial expert. Generate a realistic example of legitimate financial text about {topic}. Make it detailed and professional. The text should be 3-5 sentences long. Provide only the financial text, no explanations or disclaimers."
                    
                    # Generate the non-fraud text
                    response = client.chat.completions.create(
                        model="gpt-4.1-mini",
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": "Generate a realistic legitimate financial text example."}
                        ],
                        temperature=0.7,
                        max_tokens=300
                    )
                    
                    if response.choices and response.choices[0].message.content:
                        text = response.choices[0].message.content.strip()
                        if text:
                            non_fraud_texts.append(text)
                            
            except ImportError:
                logger.warning("OpenAI package not installed. Falling back to synthetic data generation.")
                use_openai = False
            except Exception as e:
                logger.warning(f"Error using OpenAI API: {str(e)}. Falling back to synthetic data generation.")
                use_openai = False
    
    except Exception as e:
        logger.error(f"Error in AI-based data generation: {str(e)}")
        logger.warning("Falling back to synthetic data generation.")
        use_agno = False
        use_openai = False
    
    # If both Agno and OpenAI methods failed or were not used, fall back to synthetic data generation
    if not fraud_texts or not non_fraud_texts:
        logger.info("Using synthetic data generation method...")
        
        # Define fraud keywords for synthetic generation
        fraud_keywords = [
            "unauthorized transaction", "suspicious activity", "identity theft",
            "fraudulent charge", "account takeover", "money laundering",
            "phishing attempt", "fake invoice", "ponzi scheme", "financial fraud",
            "social security number", "wire transfer", "shell company", "offshore account",
            "cryptocurrency wallet", "market manipulation", "insider trading"
        ]
        
        # Generate synthetic fraud data
        if len(fraud_texts) < num_fraud_samples:
            needed_samples = num_fraud_samples - len(fraud_texts)
            logger.info(f"Generating {needed_samples} synthetic fraud samples...")
            
            for i in range(needed_samples):
                # Generate random text with fraud keywords
                length = random.randint(100, 800)
                base_text = ' '.join([f"word{random.randint(1, 5000)}" for _ in range(length)])
                
                # Insert 1-3 fraud keywords
                num_keywords = random.randint(1, 3)
                keywords = random.sample(fraud_keywords, num_keywords)
                
                for keyword in keywords:
                    insert_pos = random.randint(0, len(base_text.split()))
                    base_text_parts = base_text.split()
                    base_text_parts.insert(insert_pos, keyword)
                    base_text = ' '.join(base_text_parts)
                
                fraud_texts.append(base_text)
        
        # Generate synthetic non-fraud data
        if len(non_fraud_texts) < num_non_fraud_samples:
            needed_samples = num_non_fraud_samples - len(non_fraud_texts)
            logger.info(f"Generating {needed_samples} synthetic non-fraud samples...")
            
            for i in range(needed_samples):
                length = random.randint(100, 800)
                text = ' '.join([f"word{random.randint(1, 5000)}" for _ in range(length)])
                non_fraud_texts.append(text)
    
    # Save the generated data
    fraud_df = pd.DataFrame({'text': fraud_texts})
    fraud_path = os.path.join(output_dir, 'fraud_data.csv')
    fraud_df.to_csv(fraud_path, index=False)
    
    non_fraud_df = pd.DataFrame({'text': non_fraud_texts})
    non_fraud_path = os.path.join(output_dir, 'non_fraud_data.csv')
    non_fraud_df.to_csv(non_fraud_path, index=False)
    
    logger.info(f"Generated {len(fraud_texts)} fraud samples and {len(non_fraud_texts)} non-fraud samples")
    logger.info(f"Fraud data saved to {fraud_path}")
    logger.info(f"Non-fraud data saved to {non_fraud_path}")
    
    return fraud_path, non_fraud_path

def process_fineweb_data(fineweb_path, output_dir, model_path=None, threshold=0.003):
    """
    Process FineWeb data to identify financial fraud content
    
    Args:
        fineweb_path (str): Path to FineWeb data
        output_dir (str): Directory to save processed data
        model_path (str): Path to pre-trained fraud detection model
        threshold (float): Threshold for classifying text as fraud
        
    Returns:
        tuple: Paths to processed fraud and non-fraud data files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load FineWeb data
    logger.info(f"Loading FineWeb data from {fineweb_path}")
    fineweb_data = pd.read_csv(fineweb_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model and tokenizer
    if model_path:
        logger.info(f"Loading pre-trained model from {model_path}")
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
    else:
        logger.info("Loading default BERT model")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    model.to(device)
    model.eval()
    
    # Process FineWeb data
    logger.info("Processing FineWeb data...")
    fraud_texts = []
    non_fraud_texts = []
    
    for text in tqdm(fineweb_data['text']):
        # Tokenize and classify
        inputs = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)
            score = probabilities[0, 1].item()  # Probability of fraud class
        
        # Classify based on threshold
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
    
    logger.info(f"Processed {len(fraud_texts)} fraud texts and {len(non_fraud_texts)} non-fraud texts")
    logger.info(f"Fraud data saved to {fraud_path}")
    logger.info(f"Non-fraud data saved to {non_fraud_path}")
    
    return fraud_path, non_fraud_path

def main():
    parser = argparse.ArgumentParser(description="Prepare data for financial fraud detection")
    
    # Data arguments
    parser.add_argument("--fineweb_path", type=str, help="Path to FineWeb data")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--model_path", type=str, help="Path to pre-trained fraud detection model")
    parser.add_argument("--threshold", type=float, default=0.003, help="Threshold for classifying text as fraud")
    parser.add_argument("--sample_size", type=int, default=10000, help="Number of FineWeb samples to download")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Mode arguments
    parser.add_argument("--download_fineweb", action="store_true", help="Download FineWeb sample")
    parser.add_argument("--download_fraud_data", action="store_true", help="Download financial fraud data")
    parser.add_argument("--process_fineweb", action="store_true", help="Process FineWeb data")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download FineWeb sample if requested
    if args.download_fineweb:
        fineweb_path = os.path.join(args.output_dir, 'fineweb_sample.csv')
        download_fineweb_sample(fineweb_path, args.sample_size)
        args.fineweb_path = fineweb_path
    
    # Download financial fraud data if requested
    if args.download_fraud_data:
        fraud_path, non_fraud_path = download_financial_fraud_data(args.output_dir)
        logger.info("Financial fraud data downloaded successfully")
    
    # Process FineWeb data if requested
    if args.process_fineweb:
        if not args.fineweb_path:
            logger.error("FineWeb path not provided. Use --fineweb_path or --download_fineweb")
            return
        
        process_fineweb_data(
            args.fineweb_path,
            args.output_dir,
            args.model_path,
            args.threshold
        )
    
    logger.info("Data preparation complete!")

if __name__ == "__main__":
    main()
