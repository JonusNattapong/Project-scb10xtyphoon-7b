"""
ประเมินประสิทธิภาพโมเดล Typhoon แบบละเอียด
"""

import os
import json
import argparse
import logging
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer for evaluation."""
    logger.info(f"Loading model from {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model with optimizations for evaluation
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
        
def load_evaluation_dataset(dataset_name="thaiqa"):
    """Load evaluation dataset."""
    logger.info(f"Loading evaluation dataset: {dataset_name}")
    
    if dataset_name == "thaiqa":
        try:
            # Try to load from Hugging Face first
            dataset = load_dataset("thaikeras/thaiquac", split="test")
            logger.info(f"Loaded ThaiQA dataset with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.warning(f"Could not load ThaiQA from Hugging Face: {e}")
            logger.info("Creating a small sample test dataset instead")
            
            # Create a small sample dataset
            questions = [
                "กรุงเทพมหานครเป็นเมืองหลวงของประเทศใด?",
                "ภาษาราชการของประเทศไทยคือภาษาอะไร?",
                "ทุเรียนมีชื่อเรียกว่าอะไร?",
                "คำว่า AI ย่อมาจากอะไร?",
                "จังหวัดไหนอยู่เหนือสุดของประเทศไทย?"
            ]
            
            answers = [
                "กรุงเทพมหานครเป็นเมืองหลวงของประเทศไทย",
                "ภาษาราชการของประเทศไทยคือภาษาไทย",
                "ทุเรียนมีชื่อเรียกว่าราชาแห่งผลไม้",
                "AI ย่อมาจาก Artificial Intelligence หรือ ปัญญาประดิษฐ์",
                "จังหวัดเชียงรายอยู่เหนือสุดของประเทศไทย"
            ]
            
            from datasets import Dataset
            return Dataset.from_dict({
                "question": questions,
                "answer": answers
            })
    
    elif dataset_name == "xquad_th":
        try:
            dataset = load_dataset("xquad", "xquad.th", split="validation")
            logger.info(f"Loaded XQuAD-TH dataset with {len(dataset)} examples")
            return dataset
        except Exception as e:
            logger.warning(f"Could not load XQuAD-TH: {e}")
            logger.warning("Falling back to sample dataset")
            return load_evaluation_dataset()
    
    else:
        logger.warning(f"Unknown dataset: {dataset_name}, using default")
        return load_evaluation_dataset()

def generate_response(model, tokenizer, prompt, max_length=256):
    """Generate model response for evaluation."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_beams=4,
            no_repeat_ngram_size=3,
            temperature=0.7,
            top_p=0.9,
            early_stopping=True,
        )
    
    # Get only the generated text, not the prompt
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    response = generated_text[len(prompt_text):].strip()
    
    return response

def evaluate_model(model, tokenizer, dataset, output_dir):
    """Evaluate model on the dataset."""
    logger.info("Starting evaluation")
    
    results = []
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    
    for idx, example in enumerate(tqdm(dataset, desc="Evaluating")):
        if idx >= 100:  # Limit to first 100 examples for speed
            break
            
        # Prepare question
        if "question" in example:
            question = example["question"]
        elif "context" in example and "question" in example:
            # For datasets like XQuAD
            question = f"บริบท: {example['context']}\n\nคำถาม: {example['question']}"
        else:
            logger.warning(f"Skipping example {idx}: no question found")
            continue
            
        # Prepare reference answer
        if "answer" in example:
            reference = example["answer"]
        elif "answers" in example and "text" in example["answers"]:
            # For datasets like XQuAD
            reference = example["answers"]["text"][0]
        else:
            logger.warning(f"Skipping example {idx}: no answer found")
            continue
            
        # Generate model response
        prompt = f"คำถาม: {question}\nคำตอบ: "
        try:
            response = generate_response(model, tokenizer, prompt)
            
            # Calculate ROUGE scores
            rouge_scores = rouge_scorer_instance.score(reference, response)
            
            # Save result
            result = {
                "question": question,
                "reference": reference,
                "response": response,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure,
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error evaluating example {idx}: {e}")
    
    # Calculate overall metrics
    rouge1_scores = [r["rouge1"] for r in results]
    rouge2_scores = [r["rouge2"] for r in results]
    rougeL_scores = [r["rougeL"] for r in results]
    
    overall_metrics = {
        "avg_rouge1": np.mean(rouge1_scores),
        "avg_rouge2": np.mean(rouge2_scores),
        "avg_rougeL": np.mean(rougeL_scores),
        "num_examples": len(results)
    }
    
    logger.info(f"Evaluation results: {overall_metrics}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "metrics": overall_metrics
        }, f, ensure_ascii=False, indent=2)
    
    # Create plots
    create_evaluation_plots(results, output_dir)
    
    return overall_metrics, results

def create_evaluation_plots(results, output_dir):
    """Create evaluation visualizations."""
    logger.info("Creating evaluation visualizations")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # 1. ROUGE score distributions
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df[["rouge1", "rouge2", "rougeL"]], bins=20, kde=True)
    plt.title("Distribution of ROUGE Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, "rouge_distribution.png"))
    
    # 2. Response length vs. Reference length
    plt.figure(figsize=(12, 6))
    df["ref_length"] = df["reference"].apply(len)
    df["resp_length"] = df["response"].apply(len)
    plt.scatter(df["ref_length"], df["resp_length"], alpha=0.6)
    plt.title("Response Length vs. Reference Length")
    plt.xlabel("Reference Length (chars)")
    plt.ylabel("Response Length (chars)")
    
    # Add diagonal line
    max_len = max(df["ref_length"].max(), df["resp_length"].max())
    plt.plot([0, max_len], [0, max_len], 'r--', label="Perfect Length Match")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "length_comparison.png"))
    
    # 3. ROUGE scores table
    plt.figure(figsize=(8, 4))
    plt.axis('tight')
    plt.axis('off')
    table_data = [
        ["Metric", "Value"],
        ["Average ROUGE-1", f"{df['rouge1'].mean():.4f}"],
        ["Average ROUGE-2", f"{df['rouge2'].mean():.4f}"],
        ["Average ROUGE-L", f"{df['rougeL'].mean():.4f}"],
        ["Median ROUGE-1", f"{df['rouge1'].median():.4f}"],
        ["Median ROUGE-2", f"{df['rouge2'].median():.4f}"],
        ["Median ROUGE-L", f"{df['rougeL'].median():.4f}"],
    ]
    plt.table(cellText=table_data, loc='center', cellLoc='center')
    plt.savefig(os.path.join(output_dir, "metrics_table.png"), bbox_inches='tight')
    
    logger.info(f"Saved evaluation visualizations to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Typhoon Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset", type=str, default="thaiqa", choices=["thaiqa", "xquad_th"], help="Evaluation dataset")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_path)
        
        # Load evaluation dataset
        dataset = load_evaluation_dataset(args.dataset)
        
        # Run evaluation
        metrics, results = evaluate_model(model, tokenizer, dataset, args.output_dir)
        
        # Print summary
        print("\n===== Evaluation Summary =====")
        print(f"Model: {args.model_path}")
        print(f"Dataset: {args.dataset} ({len(results)} examples)")
        print(f"ROUGE-1: {metrics['avg_rouge1']:.4f}")
        print(f"ROUGE-2: {metrics['avg_rouge2']:.4f}")
        print(f"ROUGE-L: {metrics['avg_rougeL']:.4f}")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
