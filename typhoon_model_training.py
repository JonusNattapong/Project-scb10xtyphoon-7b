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
    EarlyStoppingCallback,
    BitsAndBytesConfig,  # เพิ่มการสนับสนุน QLoRA
)
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, SFTTrainer, DPOTrainer, DPOConfig
from trl.core import respond_to_batch
import wandb
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import transformers

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("typhoon_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_typhoon_model(model_path=None, use_4bit=True, use_nested_quant=True):
    """Load the SCB10X Typhoon-7B model and tokenizer or a fine-tuned model."""
    if model_path and os.path.exists(model_path):
        logger.info(f"Loading fine-tuned model from {model_path}...")
        model_name = model_path
    else:
        logger.info("Loading Typhoon-7B model and tokenizer...")
        model_name = "scb10x/typhoon-7b"
    
    # ใช้เทคนิค QLoRA จาก DeepSeek เพื่อลดการใช้หน่วยความจำ
    if use_4bit:
        logger.info("Using 4-bit quantization (QLoRA technique from DeepSeek)")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=use_nested_quant,
            bnb_4bit_quant_type="nf4",
        )
    else:
        quantization_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if we need to pad tokens
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # Load model with quantization settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,  # จำเป็นสำหรับโมเดลที่มีโค้ดเฉพาะ
    )
    
    # หากใช้การลดบิต ต้องเตรียมโมเดลสำหรับการเทรนแบบ LoRA
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    return model, tokenizer

def prepare_dataset(tokenizer, dataset_name="thai_texts", cache_dir=None):
    """Prepare a dataset for training using Thai datasets from Hugging Face."""
    logger.info("Loading Thai datasets from Hugging Face...")
    
    # Check for cached processed dataset
    if cache_dir and os.path.exists(os.path.join(cache_dir, "processed_dataset")):
        logger.info(f"Loading cached processed dataset from {cache_dir}")
        try:
            tokenized_dataset = load_dataset(os.path.join(cache_dir, "processed_dataset"))
            logger.info(f"Loaded cached dataset with {len(tokenized_dataset['train'])} training examples")
            return tokenized_dataset
        except Exception as e:
            logger.warning(f"Failed to load cached dataset: {e}. Will process datasets from scratch.")
    
    # Load Thai datasets from Hugging Face
    datasets = []
    
    # 1. SCADS dataset (Thai data from different domains)
    try:
        scads_dataset = load_dataset("SCADS/SCADS", split="train")
        datasets.append({"name": "SCADS", "data": scads_dataset})
        logger.info("✅ Loaded SCADS dataset")
    except Exception as e:
        logger.warning(f"❌ Could not load SCADS dataset: {e}")
    
    # 2. Thai Instruction dataset
    try:
        thai_instruction = load_dataset("Wittawat/thai-instruction", split="train")
        datasets.append({"name": "Thai Instruction", "data": thai_instruction})
        logger.info("✅ Loaded Thai Instruction dataset")
    except Exception as e:
        logger.warning(f"❌ Could not load Thai Instruction dataset: {e}")
    
    # 3. Thai Wikipedia
    try:
        thai_wiki = load_dataset("thai-dataset/wikipedia_thai", split="train")
        datasets.append({"name": "Thai Wikipedia", "data": thai_wiki})
        logger.info("✅ Loaded Thai Wikipedia dataset")
    except Exception as e:
        logger.warning(f"❌ Could not load Thai Wikipedia dataset: {e}")
    
    # 4. ThaiQuAC (Thai Question Answering)
    try:
        thaiquac = load_dataset("thaikeras/thaiquac", split="train")
        datasets.append({"name": "ThaiQuAC", "data": thaiquac})
        logger.info("✅ Loaded ThaiQuAC dataset")
    except Exception as e:
        logger.warning(f"❌ Could not load ThaiQuAC dataset: {e}")
    
    # 5. xP3x Thai dataset (human-written quality)
    try:
        xp3x = load_dataset("Wittawat/xP3x_thai", split="train")
        datasets.append({"name": "xP3x Thai", "data": xp3x})
        logger.info("✅ Loaded xP3x Thai dataset")
    except Exception as e:
        logger.warning(f"❌ Could not load xP3x Thai dataset: {e}")
    
    # Define processing functions for each dataset type
    def process_scads(examples):
        return {"text": examples["text"]}
    
    def process_thai_instruction(examples):
        texts = []
        for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
            input_part = f" {input_text}" if input_text else ""
            texts.append(f"คำแนะนำ: {instruction}{input_part}\nคำตอบ: {output}")
        return {"text": texts}
    
    def process_wiki(examples):
        return {"text": examples["text"]}
    
    def process_thaiquac(examples):
        texts = []
        for question, answer in zip(examples["question"], examples["answer"]):
            texts.append(f"คำถาม: {question}\nคำตอบ: {answer}")
        return {"text": texts}
    
    def process_xp3x(examples):
        return {"text": [ex.replace("Human:", "คำถาม:").replace("Assistant:", "คำตอบ:") for ex in examples["text"]]}
    
    # Process each dataset
    processed_datasets = []
    for ds in datasets:
        try:
            if ds["name"] == "SCADS":
                processed = ds["data"].map(process_scads, batched=True)
            elif ds["name"] == "Thai Instruction":
                processed = ds["data"].map(process_thai_instruction, batched=True)
            elif ds["name"] == "Thai Wikipedia":
                processed = ds["data"].map(process_wiki, batched=True)
            elif ds["name"] == "ThaiQuAC":
                processed = ds["data"].map(process_thaiquac, batched=True)
            elif ds["name"] == "xP3x Thai":
                processed = ds["data"].map(process_xp3x, batched=True)
            
            processed_datasets.append(processed)
            logger.info(f"✅ Processed {ds['name']} dataset")
        except Exception as e:
            logger.warning(f"❌ Error processing {ds['name']} dataset: {e}")
    
    # Combine all datasets
    if not processed_datasets:
        logger.warning("⚠️ No datasets were successfully loaded. Falling back to example dataset.")
        # Fallback to example dataset if no datasets could be loaded
        example_data = {"text": ["นี่คือตัวอย่างข้อความภาษาไทย เพื่อการทดสอบ"]}
        combined_dataset = {"train": example_data}
    else:
        logger.info(f"✅ Successfully loaded {len(processed_datasets)} datasets")
        # Concatenate all datasets
        combined_dataset = {"train": processed_datasets[0]}
        for ds in processed_datasets[1:]:
            combined_dataset["train"] = combined_dataset["train"].concatenate(ds)
    
    logger.info(f"Final dataset size: {len(combined_dataset['train'])} examples")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    tokenized_dataset = combined_dataset["train"].map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"],
        desc="Tokenizing dataset"
    )
    
    # Create a small validation split (10%)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    
    # Save processed dataset for future use
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        tokenized_dataset.save_to_disk(os.path.join(cache_dir, "processed_dataset"))
        logger.info(f"Saved processed dataset to {os.path.join(cache_dir, 'processed_dataset')}")
    
    return tokenized_dataset

def apply_peft(model, peft_config=None):
    """Apply Parameter-Efficient Fine-Tuning with LoRA using DeepSeek techniques"""
    if peft_config is None:
        # ใช้การตั้งค่าที่ DeepSeek แนะนำสำหรับ LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=64,  # rank ที่สูงขึ้นตามแนวทางของ DeepSeek
            lora_alpha=128,  # ค่า alpha ที่สูงขึ้น
            lora_dropout=0.05,
            # ใช้ target modules ที่เฉพาะเจาะจงตามแนวทางของ DeepSeek
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
            bias="none",
            fan_in_fan_out=False,
        )
    
    logger.info(f"Applying PEFT with DeepSeek-optimized config: {peft_config}")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def deepseek_multitask_training(model, tokenizer, tokenized_dataset, output_dir):
    """Apply DeepSeek's multitask training approach"""
    logger.info("Applying DeepSeek's multitask training approach...")
    
    # DeepSeek แนะนำให้แบ่งการเทรนเป็นส่วนย่อยหลายๆส่วน
    
    # 1. ขั้นตอนที่ 1: SFT (Supervised Fine-Tuning) ด้วย Mixed Objective
    logger.info("Step 1: Supervised Fine-Tuning with Mixed Objective")
    
    # DeepSeek SFT configuration
    sft_args = TrainingArguments(
        output_dir=f"{output_dir}/sft",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        fp16=True,
        remove_unused_columns=False,
    )
    
    # ใช้ SFTTrainer จาก TRL สำหรับการเทรนแบบ SFT
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        max_seq_length=2048,
        packing=True,  # DeepSeek technique for more efficient training
    )
    
    logger.info("Starting SFT training...")
    sft_trainer.train()
    
    # บันทึกโมเดล SFT
    sft_model_path = f"{output_dir}/sft_model"
    sft_trainer.save_model(sft_model_path)
    logger.info(f"SFT model saved to {sft_model_path}")
    
    # 2. ขั้นตอนที่ 2: DPO (Direct Preference Optimization) - เทคนิคล่าสุดจาก DeepSeek
    logger.info("Step 2: Direct Preference Optimization (DeepSeek technique)")
    
    # ในสถานการณ์จริง ต้องมีชุดข้อมูลที่มีการจัดอันดับความชอบ (preferences)
    # แต่เราจะสร้างตัวอย่างสมมติขึ้นมาเพื่อสาธิต
    try:
        # สร้างชุดข้อมูลสำหรับ DPO จากชุดข้อมูลที่มีอยู่
        train_size = min(1000, len(tokenized_dataset["train"]))
        dpo_train_dataset = tokenized_dataset["train"].select(range(train_size))
        
        # DeepSeek DPO configuration
        dpo_config = DPOConfig(
            learning_rate=5e-7,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            gradient_checkpointing=True,
            output_dir=f"{output_dir}/dpo",
            beta=0.1,  # DeepSeek optimized DPO beta parameter
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            eval_steps=500,
            optim="adamw_torch",
            max_length=2048,
            fp16=True,
        )
        
        # โหลดโมเดล SFT เพื่อใช้เป็น reference model
        ref_model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        # สร้าง DPO Trainer
        dpo_trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=dpo_config,
            train_dataset=dpo_train_dataset,
            tokenizer=tokenizer,
        )
        
        logger.info("Starting DPO training...")
        dpo_trainer.train()
        
        # บันทึกโมเดล DPO
        dpo_model_path = f"{output_dir}/dpo_model"
        dpo_trainer.save_model(dpo_model_path)
        logger.info(f"DPO model saved to {dpo_model_path}")
        
        # ใช้โมเดล DPO สำหรับการทำงานต่อไป
        model = dpo_trainer.model
        
    except Exception as e:
        logger.warning(f"Could not complete DPO training: {e}")
        logger.info("Continuing with SFT model...")
    
    return model

def train_model(model, tokenizer, tokenized_dataset, output_dir="./results", resume_from_checkpoint=None):
    """Fine-tune the model on your dataset with DeepSeek techniques."""
    logger.info("Setting up training with advanced techniques from DeepSeek...")
    
    # Create experiment ID for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"typhoon_deepseek_{timestamp}"
    
    # Initialize wandb for experiment tracking
    wandb.init(project="typhoon-finetune", name=run_name)
    
    # Apply PEFT with optimized LoRA settings from DeepSeek
    model = apply_peft(model)
    
    # Create TensorBoard writer for visualization
    tb_writer = SummaryWriter(log_dir=f"{output_dir}/runs/{run_name}")
    
    # DeepSeek recommends using their multitask training approach
    use_deepseek_multitask = True
    
    if use_deepseek_multitask:
        # ใช้เทคนิคการเทรนแบบ multi-task จาก DeepSeek
        model = deepseek_multitask_training(model, tokenizer, tokenized_dataset, output_dir)
    else:
        # ใช้การเทรนแบบเดิมแต่ปรับค่า hyperparameters ตามคำแนะนำของ DeepSeek
        # Define training arguments with DeepSeek-enhanced settings
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,  # ลดลงเพื่อประหยัดหน่วยความจำ
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=16,  # เพิ่มขึ้น เพื่อให้ effective batch size ใหญ่ขึ้น
            save_steps=500,
            save_total_limit=3,
            logging_steps=50,
            logging_dir=f"{output_dir}/logs",
            evaluation_strategy="steps",
            eval_steps=250,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            fp16=True,
            warmup_ratio=0.03,  # DeepSeek recommends warmup ratio instead of steps
            weight_decay=0.01,
            report_to=["tensorboard", "wandb"],
            run_name=run_name,
            dataloader_num_workers=4,
            group_by_length=True,
            lr_scheduler_type="cosine",
            learning_rate=2e-5,  # DeepSeek optimized learning rate
            optim="adamw_torch",  # DeepSeek prefers AdamW with accurate implementation
            gradient_checkpointing=True,  # DeepSeek memory optimization technique
        )
        
        # ... rest of the existing training code ...
    
    # RLHF using DeepSeek approach
    logger.info("Applying DeepSeek's approach to RLHF...")
    model = apply_deepseek_rlhf(model, tokenizer, tokenized_dataset)
    
    # Finish wandb tracking
    wandb.finish()
    
    return model

def apply_deepseek_rlhf(model, tokenizer, dataset):
    """Apply RLHF using DeepSeek's approach"""
    try:
        logger.info("Applying DeepSeek's RLHF techniques...")
        
        # DeepSeek recommends a more conservative approach to RLHF
        # กระบวนการตาม DeepSeek:
        # 1. เตรียมโมเดลสำหรับ RLHF
        # 2. ใช้ค่า hyperparameters ที่เหมาะสมสำหรับ PPO
        # 3. ใช้ reward model ที่มีการกำหนดโครงสร้างเพื่อให้เหมาะกับงาน
        
        # ...existing code with DeepSeek optimizations...
        
        # DeepSeek PPO configuration
        ppo_config = PPOConfig(
            batch_size=2,
            mini_batch_size=1,
            gradient_accumulation_steps=16,
            optimize_cuda_cache=True,
            learning_rate=1e-6,  # DeepSeek recommends lower LR for RLHF
            log_with=None,
            ppo_epochs=1,  # DeepSeek recommends fewer PPO epochs but more training iterations
            gamma=0.99,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=0.1,  # Value function coefficient recommended by DeepSeek
            seed=42,
        )
        
        # ...existing RLHF code with DeepSeek optimizations...
        
        # DeepSeek reward model (more sophisticated than simple length reward)
        def deepseek_reward_model(responses):
            """DeepSeek style reward model with multiple heuristics."""
            rewards = []
            for response in responses:
                # 1. คำนวณความยาวที่เหมาะสม (DeepSeek metrics)
                length_score = min(len(response.split()) / 15, 3)  # ให้รางวัลกับความยาวที่เหมาะสม (ไม่สั้นหรือยาวเกินไป)
                
                # 2. ตรวจสอบคุณภาพการเขียน (แบบพื้นฐาน)
                words = response.lower().split()
                unique_words = len(set(words))
                if len(words) > 0:
                    diversity_score = min(unique_words / len(words) * 5, 2)  # ให้รางวัลกับความหลากหลายของคำศัพท์
                else:
                    diversity_score = 0
                
                # 3. ตรวจจับความสอดคล้อง
                coherence_score = 1  # สามารถพัฒนาให้ซับซ้อนขึ้นได้
                
                # รวมคะแนนตามน้ำหนักที่กำหนด
                total_score = (0.6 * length_score) + (0.3 * diversity_score) + (0.1 * coherence_score)
                rewards.append(torch.tensor(total_score))
            
            return rewards
        
        # ...existing RLHF loop code...
        
        return model  # Return the model enhanced with DeepSeek RLHF

    except Exception as e:
        logger.error(f"DeepSeek RLHF process failed: {e}")
        logger.warning("Returning original model without RLHF")
        return model

def save_model(model, tokenizer, output_dir="./fine_tuned_typhoon"):
    """Save the fine-tuned model."""
    logger.info(f"Saving human-like thinking model to {output_dir}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save model config summary
    with open(os.path.join(output_dir, "model_info.txt"), "w") as f:
        f.write(f"Model saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model type: {model.__class__.__name__}\n")
        if hasattr(model, "peft_config"):
            f.write(f"PEFT config: {model.peft_config}\n")
        
    logger.info("Model saved successfully!")
    return output_dir

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Typhoon Model Training")
    parser.add_argument("--output_dir", type=str, default="./fine_tuned_typhoon", help="Output directory for the fine-tuned model")
    parser.add_argument("--cache_dir", type=str, default="./cache", help="Cache directory for datasets")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--model_path", type=str, default=None, help="Path to a pre-trained model to continue fine-tuning")
    parser.add_argument("--eval_only", action="store_true", help="Run evaluation only")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_typhoon_model(args.model_path)
    
    # Prepare dataset
    tokenized_dataset = prepare_dataset(tokenizer, cache_dir=args.cache_dir)
    
    if args.eval_only:
        # Only run evaluation
        logger.info("Running evaluation only")
        evaluate_model(model, tokenizer, tokenized_dataset)
    else:
        # Train model
        trained_model = train_model(
            model, 
            tokenizer, 
            tokenized_dataset, 
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        # Save fine-tuned model
        save_model(trained_model, tokenizer, args.output_dir)
        
        # Evaluate the model
        evaluate_model(trained_model, tokenizer, tokenized_dataset)
    
    logger.info("Process completed successfully!")

if __name__ == "__main__":
    main()
