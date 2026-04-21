# Evaluation Results

Backend: `llm`
Sample runs per prompt: `3`

## Framework

- Precision measures how many cited precedents are actually part of the benchmark's relevant set for that prompt.
- Recall measures how much of the benchmark's relevant precedent set appears in the final answer.
- Reasoning quality scores each explanation bullet for citation, quoted grounding, issue overlap, fact-pattern overlap, and stance clarity.
- Adverse identification measures whether the agent surfaced the benchmark's adverse cases, with precision and F1 as supporting signals.
- Retrieval recall ceiling is a diagnostic metric: it shows whether the retriever exposed the right cases before synthesis had a chance to cite them.

## Summary

- Precision: `0.652`
- Recall: `0.505`
- Reasoning quality: `0.843`
- Adverse identification: `0.0`
- Retrieval recall ceiling: `0.806`

## Stability

- Precision stddev: `0.068`
- Recall stddev: `0.02`
- Reasoning stddev: `0.036`
- Adverse stddev: `0.0`

## Per-case results

### licence_pay_and_recover
- Description: Fatal commercial truck accident where the insurer raises an unlicensed-driver defense.
- Mode: `deep_research`
- Precision: `0.822`
- Recall: `0.583`
- Support precision: `0.444`
- Support recall: `0.417`
- Retrieval recall ceiling: `1.0`
- Reasoning quality: `0.893`
- Adverse identification: `0.0`
- Predicted support docs: `doc_032.pdf, doc_027.pdf, doc_034.pdf, doc_006.pdf`
- Predicted adverse docs: `doc_025.pdf, doc_029.pdf`

### contributory_negligence_truck
- Description: Truck-accident prompt focused on contributory negligence and claimant-versus-insurer alignment.
- Mode: `deep_research`
- Precision: `0.4`
- Recall: `0.333`
- Support precision: `0.0`
- Support recall: `0.0`
- Retrieval recall ceiling: `0.75`
- Reasoning quality: `0.77`
- Adverse identification: `0.0`
- Predicted support docs: `doc_029.pdf, doc_006.pdf, doc_032.pdf`
- Predicted adverse docs: `doc_018.pdf, doc_009.pdf`

### commercial_vehicle_liability
- Description: Broad commercial-vehicle precedent search that should surface both useful and risky transport cases.
- Mode: `deep_research`
- Precision: `0.733`
- Recall: `0.6`
- Support precision: `1.0`
- Support recall: `1.0`
- Retrieval recall ceiling: `0.667`
- Reasoning quality: `0.867`
- Adverse identification: `0.0`
- Predicted support docs: `doc_032.pdf, doc_029.pdf, doc_027.pdf`
- Predicted adverse docs: `doc_025.pdf, doc_014.pdf`

## Where the agent fails

- Precision is still low, which means the agent is presenting too many documents as meaningful precedents when they are only loosely related.
- Final-answer recall trails retrieval recall, so the system is dropping relevant cases during support-versus-adverse classification or final synthesis.
- Retrieval itself is missing part of the benchmark universe, which caps downstream recall before reasoning even starts.
- Adverse precedent coverage is weak, which is especially risky because the agent can sound more favorable than the corpus actually is.
- The llm-backed path shows measurable run-to-run variance, so single-run scores are not enough for regression decisions.

## What I would fix first

- Tighten the precedent classifier first: add issue tags and a second-stage reranker so support/adverse labels reflect holdings, not isolated keywords.
- Improve retrieval next with hybrid matching or richer query expansion so synonym and paraphrase gaps do not hide relevant judgments.
- Expand the benchmark with more human-labeled prompts and gold documents, because recall claims are only as trustworthy as the benchmark coverage.

## Common Across Both Approaches?

This framework is intentionally common across the non-LLM and LLM approaches because both agents expose the same prompt-to-answer contract. The benchmark set, parsing rules, and metric definitions stay the same. What changes for the LLM path is methodology: you should usually run multiple samples and track variance because generation is less deterministic.
