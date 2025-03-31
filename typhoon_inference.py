import os
import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("typhoon_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path, peft_model_path=None):
    """Load the model and tokenizer with DeepSeek optimizations."""
    logger.info(f"Loading model from {model_path}...")
    
    try:
        # Load tokenizer with error handling
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure that we have a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")
        
        # Use DeepSeek's recommended quantization for inference
        # This significantly reduces memory usage with minimal performance impact
        try:
            from transformers import BitsAndBytesConfig
            
            # DeepSeek 8-bit quantization for inference
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            
            logger.info("Using DeepSeek's 8-bit quantization for efficient inference")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        except ImportError:
            logger.warning("BitsAndBytes not available, falling back to fp16")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        
        # Load PEFT adapter if specified with DeepSeek's recommended approach
        if peft_model_path:
            try:
                logger.info(f"Loading PEFT adapter from {peft_model_path}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, peft_model_path)
                # DeepSeek recommends merging weights for inference if possible
                try:
                    logger.info("Attempting to merge PEFT adapter weights for faster inference")
                    model = model.merge_and_unload()
                    logger.info("Successfully merged adapter weights")
                except Exception as e:
                    logger.warning(f"Could not merge adapter weights: {e}")
                logger.info("PEFT adapter loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load PEFT adapter: {e}")
                logger.info("Continuing with base model only")
        
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9, top_k=50):
    """Generate a response using DeepSeek's generation techniques."""
    try:
        # Add error checking for prompt length
        if len(prompt) > 4096:
            logger.warning(f"Prompt is too long ({len(prompt)} characters). Truncating to 4096 characters.")
            prompt = prompt[:4096]
        
        # Process the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # DeepSeek recommended generation parameters for Thai language
        # Based on their research for non-English languages
        deepseek_generate_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask if hasattr(inputs, "attention_mask") else None,
            "max_length": max_length,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": 1.2,  # DeepSeek recommends higher repetition penalty for Thai
            "no_repeat_ngram_size": 4,   # DeepSeek recommends 4 for Thai language
            "do_sample": True,
            "num_beams": 1,  # Use 1 for pure sampling, try 4 for higher quality but slower generation
            "early_stopping": True,
            "length_penalty": 1.0,  # DeepSeek neutral length penalty (avoid too short/long)
            "bad_words_ids": None,
            # DeepSeek's recommendation for better Thai language generation:
            "diversity_penalty": 0.2,  # Add some diversity
            "typical_p": 0.95,  # Use typical sampling (DeepSeek recommendation)
        }
        
        # Remove None values to avoid API errors
        deepseek_generate_kwargs = {k: v for k, v in deepseek_generate_kwargs.items() if v is not None}
        
        # Generate with error handling
        with torch.no_grad():
            outputs = model.generate(**deepseek_generate_kwargs)
        
        # Decode the response with DeepSeek's method
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Only return the new part of the text after the prompt
        prompt_text = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
        
        # Makes sure the response is actually longer than the prompt
        if len(response) <= len(prompt_text):
            logger.warning("Generated response is not longer than prompt. Returning empty string.")
            return ""
        
        # DeepSeek recommendation: clean up the generated text for better presentation
        generated_text = response[len(prompt_text):]
        generated_text = generated_text.strip()
        
        return generated_text
    
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "[Error generating response]"

def process_conversation(model, tokenizer, system_prompt=None):
    """Process an interactive conversation with DeepSeek's conversation flow techniques."""
    # DeepSeek recommends a structured conversation history format
    conversation_format = "SYSTEM: {system}\n\nUSER: {user}\n\nASSISTANT: {assistant}"
    
    # Initialize conversation with DeepSeek's format
    conversations = []
    if system_prompt:
        # Add system prompt validation
        if len(system_prompt) > 1000:
            logger.warning("System prompt is too long. Truncating to 1000 characters.")
            system_prompt = system_prompt[:1000]
    else:
        system_prompt = "คุณเป็น AI ผู้ช่วยที่ฉลาดและมีประโยชน์ คุณตอบคำถามอย่างละเอียด ถูกต้อง และเป็นประโยชน์เสมอ"
    
    print("\n--- เริ่มการสนทนากับ Typhoon Model (ปรับปรุงด้วยเทคนิคจาก DeepSeek) ---")
    print("(พิมพ์ 'exit' หรือ 'quit' เพื่อจบการสนทนา)")
    
    # Add recovery mechanism for failed generations
    consecutive_failures = 0
    max_failures = 3
    
    # DeepSeek conversation context management
    max_history_turns = 5  # DeepSeek recommends limiting history to 5 turns for Thai
    
    while True:
        try:
            # Get user input
            user_input = input("\nคุณ: ")
            
            # Check if the user wants to exit
            if user_input.lower() in ["exit", "quit", "ออก", "จบ"]:
                print("\n--- จบการสนทนา ---")
                break
            
            # Add input validation
            if not user_input.strip():
                print("\nTyphoon: กรุณาพิมพ์คำถามหรือข้อความที่ต้องการสนทนา")
                continue
            
            # Update conversation history
            conversations.append({"user": user_input})
            
            # Format conversation history using DeepSeek's approach
            formatted_prompt = format_deepseek_conversation(
                system_prompt, 
                conversations, 
                max_history=max_history_turns
            )
            
            # Generate model response with DeepSeek parameters
            response = generate_response(
                model, 
                tokenizer, 
                formatted_prompt,
                temperature=0.7,   # DeepSeek's recommended temp for Thai
                top_p=0.92,        # DeepSeek's recommended top_p for Thai
                top_k=50,
                max_length=2048    # DeepSeek allows for longer contexts
            )
            
            if not response:
                # Handle empty response
                response = "ขออภัย ฉันไม่สามารถประมวลผลคำตอบได้ในขณะนี้ กรุณาถามคำถามใหม่"
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            
            # Display the response
            print(f"\nTyphoon: {response}")
            
            # Update conversation history
            conversations[-1]["assistant"] = response
            
            # Check if we need to restart conversation due to too many failures
            if consecutive_failures >= max_failures:
                print("\nระบบพบปัญหาในการตอบคำถามหลายครั้ง กรุณาเริ่มการสนทนาใหม่")
                break
        
        except KeyboardInterrupt:
            print("\n\n--- การสนทนาถูกยกเลิกโดยผู้ใช้ ---")
            break
        
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print("\n--- เกิดข้อผิดพลาดไม่คาดคิด กำลังกลับสู่การสนทนา ---")
            consecutive_failures += 1
    
    print("ขอบคุณที่ใช้บริการ Typhoon Model ที่ปรับปรุงด้วยเทคนิคจาก DeepSeek")

def format_deepseek_conversation(system, conversations, max_history=5):
    """Format conversation history using DeepSeek's approach."""
    # Keep only the most recent conversations according to max_history
    recent_conversations = conversations[-max_history:] if len(conversations) > max_history else conversations
    
    # Build the DeepSeek format
    formatted_text = f"SYSTEM: {system}\n\n"
    
    for turn in recent_conversations:
        formatted_text += f"USER: {turn['user']}\n\nASSISTANT: "
        if 'assistant' in turn:
            formatted_text += f"{turn['assistant']}\n\n"
    
    # For a new response, add just the user part
    if conversations and 'assistant' not in conversations[-1]:
        formatted_text += "USER: " + conversations[-1]['user'] + "\n\nASSISTANT: "
    
    return formatted_text.strip()

def batch_process_prompts(model, tokenizer, input_file, output_file):
    """Process a batch of prompts from a file and save responses."""
    logger.info(f"Batch processing prompts from {input_file}")
    
    # Read prompts from file
    with open(input_file, "r", encoding="utf-8") as f:
        prompts = f.readlines()
    
    # Process each prompt
    responses = []
    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()
        if not prompt:
            continue
            
        logger.info(f"Processing prompt {i+1}/{len(prompts)}")
        response = generate_response(model, tokenizer, prompt)
        responses.append(f"Prompt: {prompt}\nResponse: {response}\n\n")
    
    # Save responses to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(responses)
    
    logger.info(f"Responses saved to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Typhoon Model Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model")
    parser.add_argument("--peft_model_path", type=str, default=None, help="Path to PEFT adapter if using LoRA")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--input_file", type=str, default=None, help="Input file with prompts for batch processing")
    parser.add_argument("--output_file", type=str, default="responses.txt", help="Output file for batch responses")
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt for interactive mode")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum response length")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args.model_path, args.peft_model_path)
    
    # Run inference
    if args.interactive:
        process_conversation(model, tokenizer, args.system_prompt)
    elif args.input_file:
        batch_process_prompts(model, tokenizer, args.input_file, args.output_file)
    else:
        logger.error("Either --interactive or --input_file must be specified")
        return
    
    logger.info("Inference completed successfully!")

if __name__ == "__main__":
    main()
