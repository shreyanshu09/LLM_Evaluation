# LLM Translation Benchmark

A Python framework for benchmarking multiple local LLMs on translation quality, speed, and token efficiency — evaluated automatically using GPT-4o-mini.

---

## Overview

This tool runs translation tasks across a suite of locally hosted models via [Ollama](https://ollama.com/), then scores each output using OpenAI's GPT-4o-mini as an impartial judge. It produces structured JSON reports covering translation quality, latency, and token usage.

**Key capabilities:**

- Translate a batch of texts using multiple Ollama-hosted models in sequence
- Auto-evaluate translation quality (accuracy, fluency, consistency) via GPT-4o-mini
- Track per-model timing and token statistics
- Export detailed reports for analysis

---

## Models Tested

| Model | Type |
|---|---|
| `llama3.2:3b-instruct-fp16` | LLaMA 3.2 (3B, FP16) |
| `llama3.1:8b-instruct-fp16` | LLaMA 3.1 (8B, FP16) |
| `deepseek-r1:14b-qwen-distill-q4_K_M` | DeepSeek-R1 (14B, Q4) |
| `deepseek-r1:14b-qwen-distill-fp16` | DeepSeek-R1 (14B, FP16) |
| `qwen2.5:1.5b-instruct-fp16` | Qwen 2.5 (1.5B, FP16) |
| `qwen2.5:3b-instruct-fp16` | Qwen 2.5 (3B, FP16) |
| `qwen2.5:7b-instruct-fp16` | Qwen 2.5 (7B, FP16) |
| `qwen2.5:14b-instruct-q4_0` | Qwen 2.5 (14B, Q4) |
| `qwen2.5:14b-instruct-fp16` | Qwen 2.5 (14B, FP16) |

The model list is fully configurable via the `MODELS` constant at the top of the script.

---

## Requirements

### System
- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- CUDA-capable GPU (recommended for FP16 models)

### Python Dependencies

```bash
pip install ollama openai tiktoken torch
```

### API Key

An OpenAI API key is required for the GPT-4o-mini evaluation step. Set it in the `config` dictionary inside `main()`:

```python
"openai_api_key": "your-openai-api-key-here"
```

---

## Usage

### 1. Prepare input

Create a plain text file named `para.txt` in the same directory as the script. Each non-empty line will be treated as a separate text to translate.

```
The quick brown fox jumps over the lazy dog.
Machine translation has improved dramatically in recent years.
Please provide your identification at the front desk.
```

### 2. Run the script

```bash
python translation_benchmark.py
```

### 3. Review outputs

Five JSON files are generated in the working directory:

| File | Contents |
|---|---|
| `translated_outputs.json` | All translations and per-text metadata |
| `final_scores.json` | Total and average GPT-4o-mini scores per model |
| `timing_statistics.json` | Detailed and summarized latency data |
| `token_statistics.json` | Token counts (input, output, total) per model |
| `performance_metrics.json` | Tokens/second and time/token efficiency metrics |

---

## Output Structure

### `translated_outputs.json`
```json
[
  {
    "original": "Source text",
    "translations": { "model_name": "Translated text", ... },
    "translation_times": { "model_name": 1.23, ... },
    "translation_tokens": { "model_name": { "input_tokens": 20, ... }, ... },
    "scores": { "model_name": 8.5, ... },
    "evaluation_time": 0.95,
    "total_time": 14.2
  }
]
```

### `final_scores.json`
```json
{
  "total_scores": { "model_name": 42.0, ... },
  "average_scores": { "model_name": 8.4, ... }
}
```

### `performance_metrics.json`
```json
{
  "per_model_metrics": {
    "model_name": {
      "time_per_token": 0.012,
      "tokens_per_second": 83.4,
      ...
    }
  },
  "overall_metrics": { ... }
}
```

---

## How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  para.txt   │────▶│  Ollama Models   │────▶│  Translations +   │
│ (input texts)│     │  (one at a time) │     │  Timing / Tokens  │
└─────────────┘     └──────────────────┘     └────────┬──────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                            │  GPT-4o-mini Judge   │
                                            │  (scores 0–10 each)  │
                                            └──────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                            │   JSON Report Files  │
                                            └──────────────────────┘
```

**Translation phase:** Models are loaded one at a time. After each model finishes all texts, CUDA memory is explicitly cleared (`torch.cuda.empty_cache()`) before loading the next model — preventing OOM errors on consumer GPUs.

**Evaluation phase:** After all models have translated all texts, each original+translations pair is sent to GPT-4o-mini for scoring. The model returns a JSON object mapping model names to scores (0–10).

**Token counting:** Uses OpenAI's `tiktoken` with the GPT-4 encoding as a consistent, model-agnostic token counter.

---

## Configuration

All key settings live in the `config` dict inside `main()`:

```python
config = {
    "target_lang": "Korean",       # Target translation language
    "input_file": "para.txt",      # Input texts (one per line)
    "output_file": "translated_outputs.json",
    "scores_file": "final_scores.json",
    "timing_file": "timing_statistics.json",
    "token_file": "token_statistics.json",
    "performance_file": "performance_metrics.json",
    "openai_api_key": "YOUR_KEY"   # OpenAI API key
}
```

To change the target language, models, or file paths, edit these values directly.

---

## Notes & Limitations

- **DeepSeek-R1 `<think>` tags:** The script automatically strips `<think>...</think>` reasoning blocks from DeepSeek-R1 model outputs before saving or evaluating translations.
- **Token counts are approximate:** `tiktoken` (GPT-4 encoding) is used uniformly for all models. Actual token counts will differ for models with different tokenizers.
- **Sequential model loading:** Models are processed one at a time to avoid VRAM exhaustion. This increases total runtime but keeps memory usage predictable.
- **Evaluation bias:** GPT-4o-mini is used as the evaluator. Results reflect its scoring preferences and may not perfectly align with human judgments.
