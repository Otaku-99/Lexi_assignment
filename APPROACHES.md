# Approaches Overview

This repository includes two versions of the legal research agent built over the same judgment corpus.

- `non_llm`: a deterministic, retrieval-driven baseline
- `llm_variant`: an Ollama-based variant that reuses the same retrieval layer but uses a local LLM for final synthesis

The reason for keeping both is simple: they solve slightly different problems.

## 1. Non-LLM Approach

The `non_llm` version is the more controlled and predictable system.

It works in three stages:

1. `Planner`
   Decides whether the prompt is a simple corpus lookup or a deeper research task.

2. `Retriever`
   Extracts text from the PDFs, chunks the judgments, and ranks documents and chunks using lightweight lexical scoring plus issue-aware signals.

3. `Synthesizer`
   Merges the retrieved evidence and produces a structured answer with supporting precedents, adverse precedents, strategy, and compensation commentary where relevant.

### Why this approach

- It is deterministic, so the same prompt gives the same result.
- It is cheaper and easier to run locally.
- It is easier to debug because the reasoning path is explicit.
- It is a strong baseline for evaluation because output variation is very low.

### Strengths

- Reliable and reproducible
- Easier to inspect and tune
- Good fit for small corpora
- Better for regression testing

### Weaknesses

- Limited semantic understanding compared with stronger LLM-based reasoning
- Can miss relevant judgments when wording differs from the query
- Support vs adverse classification still depends on hand-built heuristics

## 2. LLM Variant

The `llm_variant` keeps the same planner and retriever, but changes the last step.

Instead of relying only on heuristic synthesis, it sends the retrieved evidence to a local Ollama model and asks the model to produce the final structured legal research answer.

In this repository, the intended setup is:

- Ollama installed locally
- a local model pulled first, such as `qwen2.5:3b-instruct`, `llama3:latest`, or `llama3.1:8b`
- no OpenAI dependency required for the main workflow

### Why this approach

- It produces more natural legal prose
- It can do better issue framing and explanation when the model is strong enough
- It is easier to extend into a richer memo-style research assistant

### Strengths

- Better writing quality than the purely heuristic baseline
- More flexible explanation style
- Can sometimes connect facts and legal principles more naturally

### Weaknesses

- Not deterministic, so outputs vary across runs
- Small local models can choose weak precedents or drift from the evidence
- Needs stronger prompt control and grounding to maintain precision and recall
- Harder to evaluate fairly unless multiple runs are used

## Shared Design

Both approaches share the same core corpus and much of the same retrieval pipeline.

That was intentional. It means:

- both systems are grounded in the same PDF judgments
- both can be evaluated against the same benchmark prompts
- improvements to retrieval can benefit both variants

So the real difference is not the data source. The real difference is how the final answer is generated.

- `non_llm` uses rule-based synthesis
- `llm_variant` uses LLM-based synthesis over the retrieved evidence

## When to Use Which

Use the `non_llm` version when you want:

- stable results
- easier debugging
- repeatable evaluation
- a strong baseline

Use the `llm_variant` when you want:

- more natural legal writing
- more flexible explanations
- a path toward richer research assistance

## Current Practical Takeaway

Right now, the `non_llm` approach is generally more trustworthy as a baseline because it is stable and easier to tune.

The `llm_variant` is useful because it shows how the same retrieval foundation can support a more natural research experience, but its quality depends heavily on model strength and prompt control. A stronger Ollama model such as `llama3` or `llama3.1:8b` is likely to perform better than a smaller model like `qwen2.5:3b-instruct`.

## Summary

The two approaches are not competing implementations of totally different systems. They are two layers built on the same retrieval foundation:

- one prioritizes control and determinism
- the other prioritizes fluency and flexible reasoning

Together, they show both a dependable baseline and a more ambitious local-LLM direction for the same legal research problem.
