# LLM Variant

This is a separate, optional Ollama-powered version of the legal research agent. It does not modify or replace the existing non-LLM app.

## What it supports

- local inference through `ollama`
- automatic local model recommendation based on detected RAM and CPU
- visible planning, retrieval, and LLM synthesis trace in the UI

## Setup

Install Ollama first and pull a model:

```powershell
ollama pull qwen2.5:3b-instruct
```

Then run the app:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r llm_variant\requirements.txt
streamlit run llm_variant\app.py
```

## Local models

You can also pull other local models if you want to compare quality:

```powershell
ollama pull qwen2.5:3b-instruct
ollama pull qwen2.5:7b-instruct
ollama pull llama3:latest
ollama pull llama3.1:8b
```

The current project setup uses Ollama locally rather than OpenAI.

## Inspect your system and recommended model

```powershell
python llm_variant\system_profile.py
```

## Notes

- This variant reuses the shared non-LLM retrieval stack from `non_llm/src/legal_agent`, but uses an LLM for final research synthesis.
- Make sure the Ollama server is running before launching the app or evals.
- If no local model server is available, the app will show the retrieval trace and return a helpful error message instead of changing the original app.
