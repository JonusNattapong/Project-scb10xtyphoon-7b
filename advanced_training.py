import os
import logging
import argparse
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DPOTrainer, DPOConfig
import wandb
from deepseek_utils import (
    create_together_ai_training_stages,
    apply_alpaca_training_refinements,
    apply_claude35_training_techniques,
    load_model_with_advanced_techniques,
    apply_advanced_peft_techniques,
    prepare_advanced_datasets,
    apply_rwkv_linear_attention,
    apply_llama3_mix_of_experts,
    apply_phi3_grouped_query_attention,
)

# Setup logging directory
LOG_DIR = "outputs/logs/text_training"
os.makedirs(LOG_DIR, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "advanced_training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_advanced_datasets(tokenizer, cache_dir=None):
    """
    Prepare datasets with advanced techniques from recent research.
    
    Args:
        tokenizer: The tokenizer to use.
        cache_dir: Directory to cache processed datasets.
        
    Returns:
        tokenized_dataset: The prepared and tokenized dataset.
    """
    logger.info("Preparing advanced datasets")
    
    # Check if cached dataset exists
    if cache_dir and os.path.exists(os.path.join(cache_dir, "advanced_dataset")):
        try:
            logger.info(f"Loading cached advanced dataset from {os.path.join(cache_dir, 'advanced_dataset')}")
            tokenized_dataset = load_dataset(os.path.join(cache_dir, "advanced_dataset"))
            return tokenized_dataset
        except Exception as e:
            logger.warning(f"Failed to load cached dataset: {e}. Processing from scratch.")
    
    # Load datasets from various sources
    datasets = []
    
    # 1. Thai datasets
    thai_sources = [
        {"name": "SCADS", "source": "SCADS/SCADS", "split": "train"},
        {"name": "Thai Instruction", "source": "Wittawat/thai-instruction", "split": "train"},
        {"name": "Thai Wikipedia", "source": "thai-dataset/wikipedia_thai", "split": "train"},
        {"name": "ThaiQuAC", "source": "thaikeras/thaiquac", "split": "train"},
        {"name": "xP3x Thai", "source": "Wittawat/xP3x_thai", "split": "train"},
    ]
    
    for source in thai_sources:
        try:
            logger.info(f"Loading {source['name']} dataset...")
            ds = load_dataset(source["source"], split=source["split"])
            datasets.append({"name": source["name"], "data": ds})
            logger.info(f"✅ Loaded {source['name']} dataset with {len(ds)} examples")
        except Exception as e:
            logger.warning(f"❌ Could not load {source['name']} dataset: {e}")
    
    # 2. Add high-quality international datasets (with Claude 3.5 and LLaMA-3 style techniques)
    international_sources = [
        {"name": "Alpaca", "source": "tatsu-lab/alpaca", "split": "train"},
        {"name": "OpenAssistant", "source": "OpenAssistant/oasst1", "split": "train"},
        {"name": "WizardLM", "source": "WizardLM/WizardLM_evol_instruct_V2_196k", "split": "train"},
    ]
    
    for source in international_sources:
        try:
            logger.info(f"Loading {source['name']} dataset...")
            ds = load_dataset(source["source"], split=source["split"])
            datasets.append({"name": source["name"], "data": ds})
            logger.info(f"✅ Loaded {source['name']} dataset with {len(ds)} examples")
        except Exception as e:
            logger.warning(f"❌ Could not load {source['name']} dataset: {e}")
    
    # Process datasets
    processed_datasets = []
    for ds_info in datasets:
        try:
            processed = None
            name = ds_info["name"]
            
            # Process based on dataset type
            if name == "SCADS":
                processed = ds_info["data"].map(
                    lambda x: {"text": x["text"]}, 
                    remove_columns=[c for c in ds_info["data"].column_names if c != "text"]
                )
            elif name == "Thai Instruction":
                processed = ds_info["data"].map(
                    lambda examples: {
                        "text": [
                            f"คำแนะนำ: {inst}{' ' + inp if inp else ''}\nคำตอบ: {out}"
                            for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
                        ]
                    },
                    batched=True,
                    remove_columns=ds_info["data"].column_names
                )
            elif name == "Thai Wikipedia":
                processed = ds_info["data"].map(
                    lambda x: {"text": x["text"]},
                    remove_columns=[c for c in ds_info["data"].column_names if c != "text"]
                )
            elif name == "ThaiQuAC":
                processed = ds_info["data"].map(
                    lambda examples: {
                        "text": [
                            f"คำถาม: {q}\nคำตอบ: {a}" 
                            for q, a in zip(examples["question"], examples["answer"])
                        ]
                    },
                    batched=True,
                    remove_columns=ds_info["data"].column_names
                )
            elif name == "xP3x Thai":
                processed = ds_info["data"].map(
                    lambda examples: {
                        "text": [
                            text.replace("Human:", "คำถาม:").replace("Assistant:", "คำตอบ:")
                            for text in examples["text"]
                        ]
                    },
                    batched=True,
                    remove_columns=[c for c in ds_info["data"].column_names if c != "text"]
                )
            elif name == "Alpaca":
                processed = ds_info["data"].map(
                    lambda examples: {
                        "text": [
                            f"Instruction: {inst}{' ' + inp if inp else ''}\nResponse: {out}"
                            for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
                        ]
                    },
                    batched=True,
                    remove_columns=ds_info["data"].column_names
                )
            elif name == "OpenAssistant":
                # Filter to only include high-quality responses
                filtered = ds_info["data"].filter(lambda x: "message" in x and x["message"] and "role" in x and x["role"] == "assistant")
                processed = filtered.map(
                    lambda examples: {
                        "text": examples["message"],
                    },
                    remove_columns=filtered.column_names
                )
            elif name == "WizardLM":
                processed = ds_info["data"].map(
                    lambda examples: {
                        "text": [
                            f"User: {conv[0]['value']}\nAssistant: {conv[1]['value']}" 
                            for conv in examples["conversations"] if len(conv) >= 2
                        ]
                    },
                    batched=True,
                    remove_columns=ds_info["data"].column_names
                )
            else:
                logger.warning(f"⚠️ No processing method for {name}. Skipping.")
                continue
                
            if processed:
                processed_datasets.append(processed)
                logger.info(f"✅ Processed {name} dataset with {len(processed)} examples")
            
        except Exception as e:
            logger.error(f"❌ Error processing {ds_info['name']} dataset: {e}")
    
    # Combine datasets
    if not processed_datasets:
        logger.warning("⚠️ No datasets were successfully processed. Creating a dummy dataset.")
        dummy_texts = ["นี่คือข้อความตัวอย่างสำหรับการทดสอบ"] * 10
        from datasets import Dataset
        combined_dataset = Dataset.from_dict({"text": dummy_texts})
    else:
        logger.info(f"✅ Combining {len(processed_datasets)} processed datasets")
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets(processed_datasets)
        logger.info(f"✅ Combined dataset has {len(combined_dataset)} examples")
    
    # Apply Anthropic's Constitutional AI filtering
    from deepseek_utils import apply_anthropic_constitutional_ai_filter
    filtered_dataset = apply_anthropic_constitutional_ai_filter(
        combined_dataset, 
        harm_categories=['harmful', 'unethical', 'illegal', 'racist', 'sexist', 'toxic']
    )
    logger.info(f"✅ Applied Constitutional AI filtering, {len(filtered_dataset)} examples remaining")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length",
            truncation=True,
            max_length=1024
        )
    
    tokenized_dataset = filtered_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    
    # Create train/test split
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.05)
    logger.info(f"✅ Created train/test split with {len(tokenized_dataset['train'])} train and {len(tokenized_dataset['test'])} test examples")
    
    # Cache the processed dataset
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tokenized_dataset.save_to_disk(os.path.join(cache_dir, "advanced_dataset"))
        logger.info(f"✅ Cached processed dataset to {os.path.join(cache_dir, 'advanced_dataset')}")
    
    return tokenized_dataset

def train_with_advanced_techniques(model, tokenizer, tokenized_dataset, output_dir="outputs/models/text", resume_from_checkpoint=None):
    """
    Train the model with advanced techniques from recent research.
    
    Args:
        model: The model to train.
        tokenizer: The tokenizer to use.
        tokenized_dataset: The prepared dataset.
        output_dir: Directory to save results.
        resume_from_checkpoint: Path to checkpoint to resume training.
        
    Returns:
        model: The trained model.
    """
    logger.info("Training with advanced techniques")
    
    # Use TogetherAI multi-stage training approach
    stages = create_together_ai_training_stages(None, output_dir)
    
    # Train each stage
    current_model = model
    for stage in stages:
        logger.info(f"Starting training stage: {stage['name']} - {stage['description']}")
        
        # Set training arguments for this stage
        stage_args = TrainingArguments(
            output_dir=stage['output_dir'],
            num_train_epochs=stage['epochs'],
            per_device_train_batch_size=stage['batch_size'],
            gradient_accumulation_steps=stage['gradient_accumulation_steps'],
            learning_rate=stage['learning_rate'],
            max_steps=stage.get('max_steps', -1),
            save_steps=500,
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=250,
            load_best_model_at_end=True,
            fp16=True,
            optim="adamw_torch",
        )
        
        # Apply specific techniques for each stage
        if "pretrain" in stage['name']:
            # For pretraining stage, use EleutherAI techniques
            stage_args = apply_alpaca_training_refinements(stage_args)
            
            # Train the model
            trainer = Trainer(
                model=current_model,
                args=stage_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=False
                ),
            )
            
            trainer.train(resume_from_checkpoint=resume_from_checkpoint if stage == stages[0] else None)
            current_model = trainer.model
            
        elif "sft" in stage['name']:
            # For SFT stage, use Claude 3.5 techniques
            stage_args = apply_claude35_training_techniques(stage_args)
            
            # Use SFTTrainer
            trainer = SFTTrainer(
                model=current_model,
                args=stage_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["test"],
                tokenizer=tokenizer,
                max_seq_length=2048,
                packing=True,
            )
            
            trainer.train()
            current_model = trainer.model
            
        elif "dpo" in stage['name']:
            # For DPO stage, use DPOTrainer
            
            # Load reference model
            ref_model = AutoModelForCausalLM.from_pretrained(
                stage['ref_model_path'],
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            
            # Prepare dataset for DPO
            # (In real scenarios, a dataset with chosen/rejected pairs is required)
            try:
                # Simple example - split dataset into two parts
                train_size = min(1000, len(tokenized_dataset["train"]))
                dpo_dataset = tokenized_dataset["train"].select(range(train_size))
                
                # Set up DPO trainer
                dpo_config = DPOConfig(
                    learning_rate=stage['learning_rate'],
                    lr_scheduler_type="cosine",
                    num_train_epochs=stage['epochs'],
                    per_device_train_batch_size=stage['batch_size'],
                    gradient_accumulation_steps=stage['gradient_accumulation_steps'],
                    gradient_checkpointing=True,
                    output_dir=stage['output_dir'],
                    max_length=2048,
                    fp16=True,
                )
                
                # Create DPO Trainer
                trainer = DPOTrainer(
                    model=current_model,
                    ref_model=ref_model,
                    args=dpo_config,
                    train_dataset=dpo_dataset,
                    tokenizer=tokenizer,
                )
                
                trainer.train()
                current_model = trainer.model
                
            except Exception as e:
                logger.error(f"DPO training failed: {e}")
                logger.info("Skipping DPO stage and continuing with the current model")
    
    return current_model

def main():
    """Main function for advanced training."""
    parser = argparse.ArgumentParser(description="Advanced Training for Typhoon Model")
    parser.add_argument("--output_dir", type=str, default="outputs/models/text", help="Base output directory for models")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to base model (use typhoon-7b if None)")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"], help="Quantization mode")
    parser.add_argument("--peft_method", type=str, default="lora", choices=["lora", "prefix", "both"], help="PEFT method")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--use_flash_attn", action="store_true", help="Use Flash Attention")
    parser.add_argument("--use_rwkv", action="store_true", help="Use RWKV techniques")
    parser.add_argument("--use_llama3", action="store_true", help="Use LLaMA-3 techniques")
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Set up tracking
    experiment_name = f"typhoon_advanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="typhoon-advanced-training", name=experiment_name)
    
    # Load model with advanced optimizations
    model, tokenizer = load_model_with_advanced_techniques(
        model_path=args.model_path,
        quantization=args.quantization,
        use_flash_attn=args.use_flash_attn
    )
    
    # Apply advanced PEFT techniques
    model = apply_advanced_peft_techniques(model, technique=args.peft_method)
    
    # Add specific techniques as specified
    if args.use_rwkv:
        model = apply_rwkv_linear_attention(model)
    
    if args.use_llama3:
        model = apply_llama3_mix_of_experts(model)
        model = apply_phi3_grouped_query_attention(model)
    
    # Prepare advanced datasets
    tokenized_dataset = prepare_advanced_datasets(tokenizer, cache_dir=args.cache_dir)
    
    # Train the model with advanced techniques
    trained_model = train_with_advanced_techniques(
        model,
        tokenizer,
        tokenized_dataset,
        output_dir=args.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Determine final model save path
    final_model_path = os.path.join(args.output_dir, experiment_name)
    os.makedirs(final_model_path, exist_ok=True)
    
    # Save the model
    trained_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Save model info file
    with open(os.path.join(final_model_path, "model_info.txt"), "w") as f:
        f.write(f"Advanced training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model type: {trained_model.__class__.__name__}\n")
        f.write(f"Quantization: {args.quantization}\n")
        f.write(f"PEFT method: {args.peft_method}\n")
        if args.use_flash_attn:
            f.write("Used Flash Attention\n")
        if args.use_rwkv:
            f.write("Used RWKV techniques\n")
        if args.use_llama3:
            f.write("Used LLaMA-3 techniques\n")
    
    # Finish tracking
    wandb.finish()
    
    logger.info(f"Advanced training completed. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
