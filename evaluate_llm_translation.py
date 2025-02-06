import ollama
import json
from openai import OpenAI
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import time
import tiktoken
import re
from typing import Tuple, Dict
import torch
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# List of models to test
MODELS = [
    "llama3.2:3b-instruct-fp16",
    "llama3.1:8b-instruct-fp16",
    "deepseek-r1:14b-qwen-distill-q4_K_M",
    "deepseek-r1:14b-qwen-distill-fp16",
    "qwen2.5:1.5b-instruct-fp16",
    "qwen2.5:3b-instruct-fp16",
    "qwen2.5:7b-instruct-fp16",
    "qwen2.5:14b-instruct-q4_0",
    "qwen2.5:14b-instruct-fp16"
]

def count_tokens(text: str) -> int:
    """
    Count tokens in text using GPT-4's tokenizer
    """
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except Exception as e:
        logging.error(f"Error counting tokens: {str(e)}")
        return 0

def calculate_time_per_token(time_taken: float, total_tokens: int) -> float:
    """
    Calculate average time per token, handling edge cases.
    """
    return time_taken / total_tokens if total_tokens > 0 else 0

def simple_translation(text: str, model_name: str, target_lang: str) -> Tuple[str, float, Dict[str, int]]:
    """
    Translate text using Ollama model with error handling, timing, and token counting.
    Returns tuple of (translation, execution_time, token_counts)
    """
    start_time = time.time()
    token_counts = {
        "input_tokens": count_tokens(text),
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    try:
        prompt = f'''Translate the given text formally and strictly in {target_lang} language only.\n\nTEXT:{text}'''
        token_counts["prompt_tokens"] = count_tokens(prompt)
        
        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0}
        )
        
        translation = response['message']['content']
        
        # Remove <think>...</think>\n\n if present
        translation = re.sub(r'^<think>.*?</think>\s*\n\n', '', translation, flags=re.DOTALL)
        
        token_counts["output_tokens"] = count_tokens(translation)
        token_counts["total_tokens"] = token_counts["prompt_tokens"] + token_counts["output_tokens"]
        
        execution_time = time.time() - start_time
        return translation, execution_time, token_counts
    except Exception as e:
        logging.error(f"Translation error with model {model_name}: {str(e)}")
        execution_time = time.time() - start_time
        return "", execution_time, token_counts

def evaluate_translation(original: str, translations: Dict[str, str], api_key: str) -> Tuple[Dict[str, float], float, Dict[str, int]]:
    """
    Evaluate translations using GPT-4 with error handling, timing, and token counting.
    Returns tuple of (scores, execution_time, token_counts)
    """
    start_time = time.time()
    token_counts = {
        "input_tokens": count_tokens(original),
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    prompt = f"""
    You are a translation evaluator. Given the original text and translations from different models, 
    assign a score (0-10) for each based on accuracy, fluency, and consistency. 
    Return JSON format: {{"model_name": score, ...}}
    
    Original: {original}
    
    Translations:
    {json.dumps(translations, indent=2)}
    """
    
    token_counts["prompt_tokens"] = count_tokens(prompt)
    
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Extract the content and clean it up
        content = response.choices[0].message.content
        content = content.replace('```json', '').replace('```', '').strip()
        token_counts["output_tokens"] = count_tokens(content)
        token_counts["total_tokens"] = token_counts["prompt_tokens"] + token_counts["output_tokens"]
        
        try:
            scores = json.loads(content)
            logging.info(f"Parsed scores: {scores}")
            execution_time = time.time() - start_time
            return scores, execution_time, token_counts
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse response JSON: {content}")
            logging.error(f"JSON parse error: {str(e)}")
            execution_time = time.time() - start_time
            return {model: 0 for model in translations.keys()}, execution_time, token_counts
            
    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        execution_time = time.time() - start_time
        return {model: 0 for model in translations.keys()}, execution_time, token_counts

def main():
    # Configuration
    config = {
        "target_lang": "Korean",
        "input_file": "para.txt",
        "output_file": "translated_outputs.json",
        "scores_file": "final_scores.json",
        "timing_file": "timing_statistics.json",
        "token_file": "token_statistics.json",
        "performance_file": "performance_metrics.json",
        "openai_api_key": "SECRET"  # Replace with your API key 
    }
    
    # Ensure input file exists
    if not Path(config["input_file"]).exists():
        logging.error(f"Input file {config['input_file']} not found!")
        return
    
    try:
        with open(config["input_file"], "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f.readlines() if line.strip()]
        
        # Prepare a list to store per-text results.
        # Each entry will store the original text, translations and associated timings/tokens.
        all_results = [{
            "original": text,
            "translations": {},  # {model: translation}
            "translation_times": {},  # {model: translation_time}
            "translation_tokens": {}  # {model: token_counts}
        } for text in texts]
        
        # Global stats
        overall_scores = {model: 0 for model in MODELS}
        timing_stats = {
            "translation_times": {model: [] for model in MODELS},
            "evaluation_times": [],
            "per_text_times": []  # total time per text (sum of its translations and evaluation)
        }
        token_stats = {
            "translation_tokens": {model: [] for model in MODELS},
            "evaluation_tokens": []
        }
        
        # ----- Translation Phase: Process one model at a time -----
        for model in MODELS:
            if torch.cuda.is_available():
                print("START CLEAR")
                gc.collect() 
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("CLEAR")

            logging.info(f"Loading and processing translations for model: {model}")
            # (If there is an explicit model load function, call it here.)
            for idx, text in enumerate(texts):
                translation, trans_time, trans_tokens = simple_translation(text, model, config["target_lang"])
                # Save the translation if successful
                if translation:
                    all_results[idx]["translations"][model] = translation
                all_results[idx]["translation_times"][model] = trans_time
                all_results[idx]["translation_tokens"][model] = trans_tokens
                
                timing_stats["translation_times"][model].append(trans_time)
                token_stats["translation_tokens"][model].append(trans_tokens)
        
        # ----- Evaluation Phase: Process evaluation per text -----
        for idx, result in enumerate(all_results):
            text = result["original"]
            translations = result["translations"]
            # Only evaluate if there is at least one successful translation.
            if translations:
                eval_start = time.time()
                scores, eval_time, eval_tokens = evaluate_translation(text, translations, config["openai_api_key"])
                result["scores"] = scores
                result["evaluation_time"] = eval_time
                result["evaluation_tokens"] = eval_tokens
                timing_stats["evaluation_times"].append(eval_time)
                token_stats["evaluation_tokens"].append(eval_tokens)
                # Sum the translation times for this text from all models plus evaluation time
                total_text_time = sum(result["translation_times"].values()) + eval_time
                result["total_time"] = total_text_time
                timing_stats["per_text_times"].append(total_text_time)
                
                # Update overall scores
                for model, score in scores.items():
                    overall_scores[model] += score
            else:
                # If no translations succeeded, still record a zero evaluation
                result["scores"] = {model: 0 for model in MODELS}
                result["evaluation_time"] = 0
                result["evaluation_tokens"] = {"input_tokens": 0, "prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                result["total_time"] = sum(result["translation_times"].values())
                timing_stats["per_text_times"].append(result["total_time"])
        
        # Calculate averages and summaries
        num_texts = len(texts)
        avg_scores = {model: overall_scores[model] / num_texts for model in overall_scores}
        
        # Timing summary
        timing_summary = {
            "average_translation_times": {
                model: sum(times)/len(times) if times else 0 
                for model, times in timing_stats["translation_times"].items()
            },
            "average_evaluation_time": sum(timing_stats["evaluation_times"])/len(timing_stats["evaluation_times"]) 
                if timing_stats["evaluation_times"] else 0,
            "average_total_time_per_text": sum(timing_stats["per_text_times"])/len(timing_stats["per_text_times"])
                if timing_stats["per_text_times"] else 0,
            "total_execution_time": sum(timing_stats["per_text_times"])
        }
        
        # Token summary
        token_summary = {
            "average_translation_tokens": {
                model: {
                    "input": sum(t["input_tokens"] for t in tokens)/len(tokens) if tokens else 0,
                    "output": sum(t["output_tokens"] for t in tokens)/len(tokens) if tokens else 0,
                    "total": sum(t["total_tokens"] for t in tokens)/len(tokens) if tokens else 0
                }
                for model, tokens in token_stats["translation_tokens"].items()
            },
            "average_evaluation_tokens": {
                "input": sum(t["input_tokens"] for t in token_stats["evaluation_tokens"])/len(token_stats["evaluation_tokens"]) 
                    if token_stats["evaluation_tokens"] else 0,
                "output": sum(t["output_tokens"] for t in token_stats["evaluation_tokens"])/len(token_stats["evaluation_tokens"]) 
                    if token_stats["evaluation_tokens"] else 0,
                "total": sum(t["total_tokens"] for t in token_stats["evaluation_tokens"])/len(token_stats["evaluation_tokens"]) 
                    if token_stats["evaluation_tokens"] else 0
            },
            "total_tokens": {
                "translation": sum(sum(t["total_tokens"] for t in tokens) for tokens in token_stats["translation_tokens"].values()),
                "evaluation": sum(t["total_tokens"] for t in token_stats["evaluation_tokens"]),
            }
        }
        token_summary["total_tokens"]["overall"] = (
            token_summary["total_tokens"]["translation"] + 
            token_summary["total_tokens"]["evaluation"]
        )
        
        # Performance metrics
        performance_metrics = {
            "per_model_metrics": {},
            "evaluation_metrics": {
                "time_per_token": calculate_time_per_token(
                    sum(timing_stats["evaluation_times"]),
                    token_summary["total_tokens"]["evaluation"]
                ),
                "tokens_per_second": token_summary["total_tokens"]["evaluation"] / 
                    sum(timing_stats["evaluation_times"]) if sum(timing_stats["evaluation_times"]) > 0 else 0
            },
            "overall_metrics": {}
        }
        
        # Calculate per-model metrics
        for model in MODELS:
            model_total_time = sum(timing_stats["translation_times"][model])
            model_total_tokens = sum(t["total_tokens"] for t in token_stats["translation_tokens"][model])
            
            performance_metrics["per_model_metrics"][model] = {
                "time_per_token": calculate_time_per_token(model_total_time, model_total_tokens),
                "tokens_per_second": model_total_tokens / model_total_time if model_total_time > 0 else 0,
                "average_metrics_per_text": {
                    "avg_time_per_token": (
                        sum(calculate_time_per_token(time, tokens["total_tokens"])
                            for time, tokens in zip(
                                timing_stats["translation_times"][model],
                                token_stats["translation_tokens"][model]
                            )
                        ) / len(timing_stats["translation_times"][model])
                        if timing_stats["translation_times"][model] else 0
                    ),
                    "avg_tokens_per_second": (
                        sum(tokens["total_tokens"] / time 
                            for time, tokens in zip(
                                timing_stats["translation_times"][model],
                                token_stats["translation_tokens"][model]
                            )
                            if time > 0
                        ) / len(timing_stats["translation_times"][model])
                        if timing_stats["translation_times"][model] else 0
                    )
                },
                "detailed_stats": {
                    "total_time": model_total_time,
                    "total_tokens": model_total_tokens,
                    "input_tokens": sum(t["input_tokens"] for t in token_stats["translation_tokens"][model]),
                    "output_tokens": sum(t["output_tokens"] for t in token_stats["translation_tokens"][model])
                }
            }
        
        # Calculate overall metrics
        total_time = sum(timing_stats["per_text_times"])
        total_tokens = token_summary["total_tokens"]["overall"]
        
        performance_metrics["overall_metrics"] = {
            "time_per_token": calculate_time_per_token(total_time, total_tokens),
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "total_processing_time": total_time,
            "total_tokens_processed": total_tokens
        }
        
        # Save all results
        output_files = [
            (config["output_file"], all_results),
            (config["scores_file"], {"total_scores": overall_scores, "average_scores": avg_scores}),
            (config["timing_file"], {
                "detailed_timing": timing_stats,
                "timing_summary": timing_summary
            }),
            (config["token_file"], {
                "detailed_tokens": token_stats,
                "token_summary": token_summary
            }),
            (config["performance_file"], performance_metrics)
        ]
        
        for file_path, data in output_files:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        
        # Enhanced logging
        logging.info("Processing complete! Results saved.")
        logging.info("Performance Metrics Summary:")
        for model in MODELS:
            metrics = performance_metrics["per_model_metrics"][model]
            logging.info(f"\n{model}:")
            logging.info(f"  Time per token: {metrics['time_per_token']:.4f} seconds")
            logging.info(f"  Tokens per second: {metrics['tokens_per_second']:.2f}")
            logging.info(f"  Average time per token per text: {metrics['average_metrics_per_text']['avg_time_per_token']:.4f} seconds")
        
        logging.info("\nOverall Performance:")
        logging.info(f"  Total time per token: {performance_metrics['overall_metrics']['time_per_token']:.4f} seconds")
        logging.info(f"  Overall tokens per second: {performance_metrics['overall_metrics']['tokens_per_second']:.2f}")
        
    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")

if __name__ == "__main__":
    main()
