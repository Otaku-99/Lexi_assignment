# System Config Requirements

This LLM variant supports two modes.

## 1. Free local mode with Ollama

Recommended system guidance:

- `8 GB RAM`: use `qwen2.5:3b-instruct` or `llama3.2:3b`
- `16 GB RAM`: use `qwen2.5:7b-instruct`
- `24 GB+ RAM`: use `qwen2.5:14b-instruct` if you want a stronger local model
- CPU-only is fine for `3b` models, but response time will be slower
- A discrete GPU improves speed, but is not required

Practical recommendation for most laptops:

- Start with `qwen2.5:3b-instruct`
- Move to `qwen2.5:7b-instruct` only if RAM and latency are comfortable

## 2. OpenAI API mode

Required:

- internet access
- `OPENAI_API_KEY`

Recommended model choices:

- `gpt-4.1-mini` for cost-effective legal synthesis
- `gpt-4.1` for stronger reasoning quality

## Current code behavior

The app auto-detects your available RAM and CPU count at runtime and recommends a local Ollama model profile. If detection fails, it falls back to a safe recommendation of `qwen2.5:3b-instruct`.

