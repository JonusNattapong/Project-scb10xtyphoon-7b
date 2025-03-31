import os
import argparse
import logging
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoProcessor
import re
import json
from tqdm import tqdm
import torch
from PIL import Image
import base64
from io import BytesIO
from vision_utils import VisionEnhancer
from transformers import AutoModelForCausalLM, AutoTokenizer as TeacherTokenizer
import torch.nn.functional as F

# Setup logging directory
LOG_DIR = "outputs/logs/data_preprocessing"
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "data_preprocessing.log")),
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

def encode_image(image_path):
    """Encode image to base64 string."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize if too large (optional)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            # Convert to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

def prepare_vision_dataset(dataset, image_column="image", text_column="text", caption_column="caption"):
    """Prepare a vision dataset with images and text/captions using advanced techniques."""
    logger.info("Preparing vision dataset with advanced processing...")
    
    # Initialize vision enhancer
    enhancer = VisionEnhancer()
    
    def process_vision_data(examples):
        processed_data = {
            "text": [],
            "image_data": [],
            "features": [],
            "attention_maps": []
        }
        
        for i in range(len(examples[image_column])):
            # Get data
            image_path = examples[image_column][i]
            text = examples.get(text_column, [""] * len(examples[image_column]))[i]
            caption = examples.get(caption_column, [""] * len(examples[image_column]))[i]
            
            if isinstance(image_path, str) and os.path.isfile(image_path):
                try:
                    # Enhance image quality
                    enhanced_image = enhancer.enhance_image(image_path)
                    
                    # Extract features and attention maps
                    vision_info = enhancer.extract_features(
                        enhanced_image,
                        text_prompt=caption if caption else text
                    )
                    
                    # Encode enhanced image
                    buffer = BytesIO()
                    enhanced_image.save(buffer, format="JPEG")
                    image_base64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Store all information
                    processed_data["image_data"].append(image_base64)
                    processed_data["features"].append(vision_info["image_features"].cpu().numpy())
                    processed_data["attention_maps"].append(vision_info["attention_maps"].cpu().numpy())
                    
                    # Combine text with features
                    combined_text = text
                    if caption:
                        combined_text = f"{text}\nCaption: {caption}" if text else caption
                    if vision_info["similarity_score"] is not None:
                        combined_text += f"\nFeature Score: {float(vision_info['similarity_score'][0][0]):.3f}"
                    processed_data["text"].append(combined_text)
                    
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {str(e)}")
                    continue
            
        return processed_data
    
    # Process the dataset
    processed_dataset = dataset.map(
        process_vision_data,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    logger.info(f"Processed {len(processed_dataset)} vision examples")
    return processed_dataset

def prepare_conversation_dataset(dataset, format_template="User: {question}\nAssistant: {answer}", image_column=None):
    """Prepare a conversation dataset using the provided template."""
    logger.info("Preparing conversation dataset...")
    
    # Function to format conversations
    def format_conversations(examples):
        conversations = []
        image_data = []
        
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            answer = examples['answer'][i] if 'answer' in examples else ""
            
            if not question or not answer:
                continue
            
            # Clean the text
            question = clean_text(question)
            answer = clean_text(answer)
            
            # Handle image if available
            if image_column and image_column in examples:
                image = examples[image_column][i]
                if isinstance(image, str) and os.path.isfile(image):
                    image_base64 = encode_image(image)
                    if image_base64:
                        image_data.append(image_base64)
                    else:
                        continue
            
            # Format the conversation
            conversation = format_template.format(question=question, answer=answer)
            conversations.append(conversation)
        
        result = {"text": conversations}
        if image_data:
            result["image_data"] = image_data
        return result
    
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

def generate_teacher_outputs(dataset, teacher_model_name="microsoft/OmniParser-v2.0"):
    """Generate outputs from teacher model for knowledge distillation."""
    logger.info(f"Generating teacher outputs using {teacher_model_name}...")
    
    # Load teacher model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_tokenizer = TeacherTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name).to(device)
    teacher_model.eval()
    
    def get_teacher_logits(examples):
        teacher_outputs = []
        
        with torch.no_grad():
            for text in examples["text"]:
                # Tokenize input
                inputs = teacher_tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # Get teacher predictions
                outputs = teacher_model(**inputs)
                logits = outputs.logits
                
                # Get probabilities using softmax
                probs = F.softmax(logits, dim=-1)
                
                # Convert to CPU and numpy for storage
                probs = probs.cpu().numpy()
                
                teacher_outputs.append(probs)
        
        return {"teacher_logits": teacher_outputs}
    
    # Generate teacher outputs
    dataset_with_teacher = dataset.map(
        get_teacher_logits,
        batched=True,
        batch_size=8,  # Adjust based on GPU memory
        desc="Generating teacher outputs"
    )
    
    logger.info("Completed teacher output generation")
    return dataset_with_teacher

def prepare_teacher_student_dataset(dataset, text_column="text"):
    """Prepare dataset for teacher-student learning."""
    logger.info("Preparing teacher-student dataset...")
    
    # Generate teacher outputs
    dataset_with_teacher = generate_teacher_outputs(dataset)
    
    # Clean the text
    def clean_dataset_text(examples):
        cleaned_texts = [clean_text(text) for text in examples[text_column]]
        return {text_column: cleaned_texts}
    
    processed_dataset = dataset_with_teacher.map(
        clean_dataset_text,
        batched=True
    )
    
    logger.info(f"Processed {len(processed_dataset)} teacher-student examples")
    return processed_dataset

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Data Preprocessing for Typhoon Model")
    parser.add_argument("--input_file", type=str, help="Input file path for custom dataset")
    parser.add_argument("--input_type", type=str, default="csv", choices=["csv", "json", "jsonl", "text"], help="Input file type")
    parser.add_argument("--output_dir", type=str, default="outputs/processed_data", help="Base output directory for processed dataset")
    parser.add_argument("--tokenizer_name", type=str, default="scb10x/typhoon-7b", help="Tokenizer to use")
    parser.add_argument("--processor_name", type=str, help="Vision processor to use for image processing")
    parser.add_argument("--dataset_type", type=str, default="text", 
                       choices=["text", "conversation", "instruction", "vision", "teacher_student"], 
                       help="Type of dataset to prepare")
    parser.add_argument("--hf_dataset", type=str, help="Hugging Face dataset to use instead of input file")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text data")
    parser.add_argument("--image_column", type=str, help="Column name for image data")
    parser.add_argument("--caption_column", type=str, help="Column name for image captions")
    args = parser.parse_args()
    
    # Determine specific output path based on dataset type
    specific_output_dir = os.path.join(args.output_dir, args.dataset_type)
    if args.input_file:
        dataset_name = os.path.splitext(os.path.basename(args.input_file))[0]
        specific_output_dir = os.path.join(specific_output_dir, dataset_name)
    elif args.hf_dataset:
        dataset_name = args.hf_dataset.replace("/", "_")
        specific_output_dir = os.path.join(specific_output_dir, dataset_name)
    
    # Create output directory
    os.makedirs(specific_output_dir, exist_ok=True)
    
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
    
    # Load processor if needed
    processor = None
    if args.processor_name:
        logger.info(f"Loading processor: {args.processor_name}")
        processor = AutoProcessor.from_pretrained(args.processor_name)
    
    # Process dataset based on type
    if args.dataset_type == "conversation":
        processed_dataset = prepare_conversation_dataset(dataset, image_column=args.image_column)
    elif args.dataset_type == "instruction":
        processed_dataset = prepare_instruction_dataset(dataset)
    elif args.dataset_type == "vision":
        processed_dataset = prepare_vision_dataset(dataset, args.image_column, args.text_column, args.caption_column)
    elif args.dataset_type == "teacher_student":
        processed_dataset = prepare_teacher_student_dataset(dataset, args.text_column)
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
    dataset_dict.save_to_disk(specific_output_dir)
    logger.info(f"Saved processed dataset to {specific_output_dir}")
    
    # Save a sample of the processed text for inspection
    sample_size = min(10, len(processed_dataset))
    sample_texts = processed_dataset["text"][:sample_size]
    
    sample_file_path = os.path.join(specific_output_dir, "sample_texts.txt")
    with open(sample_file_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(sample_texts):
            f.write(f"=== Sample {i+1} ===\n{text}\n\n")
    
    logger.info(f"Saved {sample_size} sample texts to {sample_file_path}")
    logger.info("Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
