# Lexi Legal Research Agent

A Streamlit app that researches the provided corpus of Indian motor accident judgments and shows its working: how it classified the query, which document queries it ran, which judgments it retrieved, and how it turned those into legal analysis.

## What it does

- Handles broad corpus questions such as `Which of these judgments involve commercial vehicles?`
- Handles deeper research prompts asking for supporting precedents, adverse precedents, and litigation strategy
- Extracts text from the PDF corpus, builds page-aware chunks, and ranks both whole documents and chunks
- Shows intermediate reasoning instead of hiding retrieval and ranking behind a final answer

## Project structure

- [app.py](/d:/lexi_assignment/non_llm/app.py)
- [src/legal_agent/agent.py](/d:/lexi_assignment/non_llm/src/legal_agent/agent.py)
- [src/legal_agent/retrieval.py](/d:/lexi_assignment/non_llm/src/legal_agent/retrieval.py)
- [src/legal_agent/planner.py](/d:/lexi_assignment/non_llm/src/legal_agent/planner.py)
- [src/legal_agent/synthesizer.py](/d:/lexi_assignment/non_llm/src/legal_agent/synthesizer.py)
- [ADR.md](/d:/lexi_assignment/ADR.md)

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r non_llm\requirements.txt
streamlit run non_llm\app.py
```

The app reads PDFs from `lexi_research_take_home_assessment_docs/` and writes a cached extracted index to `.cache/corpus_index.json`.

## Run the evaluation suite

```powershell
python -m evals.run_evals
```

This writes:

- `reports/eval_results.json`
- `reports/EVAL_REPORT.md`

## How the agent works

1. The planner decides whether the prompt is a quick corpus query or a deeper research task.
2. For deeper tasks, it expands the user prompt into several search queries based on detected legal issues.
3. The retriever ranks whole documents and page-aware chunks with a lightweight TF-IDF style scorer.
4. The synthesizer merges evidence across searches and produces a grounded answer with supporting and adverse authorities plus strategy.
5. The UI displays the plan, retrieval trace, and cited snippets.

## Notes

- The system is intentionally corpus-grounded and does not depend on external legal databases.
- The current synthesis layer is heuristic and extractive rather than a fully generative legal memo writer.
- If you want to host it, Streamlit Community Cloud or a small VM/container is sufficient for a 50-document corpus.
