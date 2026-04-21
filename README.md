# Lexi Assignment Repo

This repo is now organized into two separate application folders that share the same judgment corpus.

## Structure

- [non_llm](/d:/lexi_assignment/non_llm): deterministic baseline app and core retrieval code
- [llm_variant](/d:/lexi_assignment/llm_variant): separate Ollama-based LLM variant
- [lexi_research_take_home_assessment_docs](/d:/lexi_assignment/lexi_research_take_home_assessment_docs): shared PDF corpus
- [evals](/d:/lexi_assignment/evals): automated evaluation framework
- [reports](/d:/lexi_assignment/reports): generated evaluation outputs
- [ADR.md](/d:/lexi_assignment/ADR.md): architecture decision record

## Run The Non-LLM App

```powershell
.venv\Scripts\Activate.ps1
pip install -r non_llm\requirements.txt
streamlit run non_llm\app.py
```

## Run The LLM App

Install Ollama first, make sure it is running, and pull the local model:

```powershell
ollama pull qwen2.5:3b-instruct
```

```powershell
.venv\Scripts\Activate.ps1
pip install -r llm_variant\requirements.txt
streamlit run llm_variant\app.py
```

## Run Evals

```powershell
.venv\Scripts\Activate.ps1
python -m evals.run_evals --backend non_llm
```

Evaluate the LLM variant with the same benchmark contract:

```powershell
.venv\Scripts\Activate.ps1
python -m evals.run_evals --backend llm --provider ollama --model qwen2.5:3b-instruct --sample-runs 3
```

The framework is common across both approaches. The prompts, gold labels, parser, and metric definitions stay the same; only the execution methodology changes. For LLM-backed runs, use multiple samples to measure variance.
