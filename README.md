# Lexi Assignment Repo

This repo is now organized into two separate application folders that share the same judgment corpus.

## Hosted App

Streamlit deployment: `https://lexiassignment-lvguxe8gqzy2ruthn8k3gf.streamlit.app/`

## Structure

- [non_llm](./non_llm): deterministic baseline app and core retrieval code
- [llm_variant](./llm_variant): separate Ollama-based LLM variant
- [lexi_research_take_home_assessment_docs](./lexi_research_take_home_assessment_docs): shared PDF corpus
- [evals](./evals): automated evaluation framework
- [reports](./reports): generated evaluation outputs and write-ups
- [ADR.md](./ADR.md): architecture decision record

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

### Evaluation framework (Deliverables)

The evaluation framework lives in `evals/run_evals.py` and evaluates both backends against the same benchmark contract (`evals/benchmark_cases.py`).

- **Outputs**: JSON + a Markdown report are written under `reports/`.
- **Latest run summaries**: see `reports/EVAL_RUN_SUMMARY.md`.
