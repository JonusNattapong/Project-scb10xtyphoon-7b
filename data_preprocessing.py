import os
import argparse
import logging
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import re
import json
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_text(text, keep_newlines=True):
    """Clean text by removing unnecessary whitespace and special characters."""
    if not isinstance(text, str):
        return ""
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Keep newlines if specified
    if keep_newlines:
        # Replace multiple newlines with a single newline
        text = re.sub(r'\n+', '\n', text)
    else:
        # Replace newlines with a space
        text = text.replace('\n', ' ')
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text

def prepare_conversation_dataset(dataset, format_template="User: {question}\nAssistant: {answer}"):
    """Prepare a conversation dataset using the provided template."""
    logger.info("Preparing conversation dataset...")
    
    # Function to format conversations
    def format_conversations(examples):
        conversations = []
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            answer = examples['answer'][i] if 'answer' in examples else ""
            
            if not question or not answer:
                continue
            
            # Clean the text
            question = clean_text(question)
            answer = clean_text(answer)
            
            # Format the conversation
            conversation = format_template.format(question=question, answer=answer)
            conversations.append(conversation)
        
        return {"text": conversations}
    
    # Process the dataset
    processed_dataset = dataset.map(
        format_conversations,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Processed {len(processed_dataset)} conversation examples")
    return processed_dataset

def prepare_instruction_dataset(dataset, format_template="Instruction: {instruction}\nResponse: {output}"):
    """Prepare an instruction dataset using the provided template."""
    logger.info("Preparing instruction dataset...")
    
    # Function to format instructions
    def format_instructions(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            instruction = examples['instruction'][i]
            input_text = examples['input'][i] if 'input' in examples else ""
            output = examples['output'][i] if 'output' in examples else ""
            
            if not instruction or not output:
                continue
            
            # Clean the text
            instruction = clean_text(instruction)
            input_text = clean_text(input_text)
            output = clean_text(output)
            
            # Combine instruction and input if input exists
            if input_text:
                instruction = f"{instruction}\n{input_text}"
            
            # Format the instruction-output pair
            text = format_template.format(instruction=instruction, output=output)
            texts.append(text)
        
        return {"text": texts}
    
    # Process the dataset
    processed_dataset = dataset.map(
        format_instructions,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Processed {len(processed_dataset)} instruction examples")
    return processed_dataset

def prepare_text_dataset(dataset, text_column="text"):
    """Prepare a text dataset by cleaning the text."""
    logger.info("Preparing text dataset...")
    
    # Function to clean text
    def clean_dataset_text(examples):
        cleaned_texts = [clean_text(text) for text in examples[text_column]]
        return {text_column: cleaned_texts}
    
    # Process the dataset
    processed_dataset = dataset.map(
        clean_dataset_text,
        batched=True
    )
    
    logger.info(f"Processed {len(processed_dataset)} text examples")
    return processed_dataset

def tokenize_dataset(dataset, tokenizer, max_length=512, text_column="text"):
    """Tokenize the dataset."""
    logger.info(f"Tokenizing dataset with max_length={max_length}...")
    
    # Function to tokenize text
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column],
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Tokenize the dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=[text_column],
        desc="Tokenizing dataset"
    )
    
    logger.info(f"Tokenized {len(tokenized_dataset)} examples")
    return tokenized_dataset

def load_custom_dataset(file_path, dataset_type="csv"):
    """Load a custom dataset from file."""
    logger.info(f"Loading custom dataset from {file_path}")
    
    if dataset_type == "csv":
        df = pd.read_csv(file_path)
        dataset = Dataset.from_pandas(df)
    elif dataset_type == "json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    elif dataset_type == "jsonl":
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        dataset = Dataset.from_list(data)
    elif dataset_type == "text":
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        dataset = Dataset.from_dict({"text": [line.strip() for line in lines if line.strip()]})
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    logger.info(f"Loaded dataset with {len(dataset)} examples")
    return dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data Preprocessing for Typhoon Model")
    parser.add_argument("--input_file", type=str, help="Input file path for custom dataset")
    parser.add_argument("--input_type", type=str, default="csv", choices=["csv", "json", "jsonl", "text"], help="Input file type")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed dataset")
    parser.add_argument("--tokenizer_name", type=str, default="scb10x/typhoon-7b", help="Tokenizer to use")
    parser.add_argument("--dataset_type", type=str, default="text", choices=["text", "conversation", "instruction"], help="Type of dataset to prepare")
    parser.add_argument("--hf_dataset", type=str, help="Hugging Face dataset to use instead of input file")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # Load dataset
    if args.hf_dataset:
        logger.info(f"Loading Hugging Face dataset: {args.hf_dataset}")
        dataset = load_dataset(args.hf_dataset, split="train")
    elif args.input_file:
        dataset = load_custom_dataset(args.input_file, args.input_type)
    else:
        logger.error("Either --input_file or --hf_dataset must be specified")
        return
    
    # Process dataset based on type
    if args.dataset_type == "conversation":
        processed_dataset = prepare_conversation_dataset(dataset)
    elif args.dataset_type == "instruction":
        processed_dataset = prepare_instruction_dataset(dataset)
    else:
        processed_dataset = prepare_text_dataset(dataset, args.text_column)
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(processed_dataset, tokenizer, args.max_length)
    
    # Create train/test split
    dataset_dict = DatasetDict({
        "train": tokenized_dataset.select(range(int(len(tokenized_dataset) * 0.9))),
        "test": tokenized_dataset.select(range(int(len(tokenized_dataset) * 0.9), len(tokenized_dataset)))
    })
    
    # Save processed dataset
    dataset_dict.save_to_disk(args.output_dir)
    logger.info(f"Saved processed dataset to {args.output_dir}")
    
    # Save a sample of the processed text for inspection
    sample_size = min(10, len(processed_dataset))
    sample_texts = processed_dataset["text"][:sample_size]
    
    with open(os.path.join(args.output_dir, "sample_texts.txt"), "w", encoding="utf-8") as f:
        for i, text in enumerate(sample_texts):
            f.write(f"=== Sample {i+1} ===\n{text}\n\n")
    
    logger.info(f"Saved {sample_size} sample texts to {os.path.join(args.output_dir, 'sample_texts.txt')}")
    logger.info("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
