# Evaluation Results

Backend: `non_llm`
Sample runs per prompt: `1`

## Framework

- Precision measures how many cited precedents are actually part of the benchmark's relevant set for that prompt.
- Recall measures how much of the benchmark's relevant precedent set appears in the final answer.
- Reasoning quality scores each explanation bullet for citation, quoted grounding, issue overlap, fact-pattern overlap, and stance clarity.
- Adverse identification measures whether the agent surfaced the benchmark's adverse cases, with precision and F1 as supporting signals.
- Retrieval recall ceiling is a diagnostic metric: it shows whether the retriever exposed the right cases before synthesis had a chance to cite them.

## Summary

- Precision: `0.637`
- Recall: `0.725`
- Reasoning quality: `0.971`
- Adverse identification: `0.167`
- Retrieval recall ceiling: `0.806`

## Per-case results

### licence_pay_and_recover
- Description: Fatal commercial truck accident where the insurer raises an unlicensed-driver defense.
- Mode: `deep_research`
- Precision: `0.778`
- Recall: `0.875`
- Support precision: `0.4`
- Support recall: `0.5`
- Retrieval recall ceiling: `1.0`
- Reasoning quality: `0.96`
- Adverse identification: `0.5`
- Predicted support docs: `doc_006.pdf, doc_027.pdf, doc_033.pdf, doc_029.pdf, doc_018.pdf`
- Predicted adverse docs: `doc_034.pdf, doc_032.pdf, doc_025.pdf, doc_031.pdf`

### contributory_negligence_truck
- Description: Truck-accident prompt focused on contributory negligence and claimant-versus-insurer alignment.
- Mode: `deep_research`
- Precision: `0.333`
- Recall: `0.5`
- Support precision: `0.4`
- Support recall: `0.667`
- Retrieval recall ceiling: `0.75`
- Reasoning quality: `0.97`
- Adverse identification: `0.0`
- Predicted support docs: `doc_018.pdf, doc_009.pdf, doc_006.pdf, doc_027.pdf, doc_015.pdf`
- Predicted adverse docs: `doc_025.pdf, doc_029.pdf, doc_032.pdf, doc_023.pdf`

### commercial_vehicle_liability
- Description: Broad commercial-vehicle precedent search that should surface both useful and risky transport cases.
- Mode: `deep_research`
- Precision: `0.8`
- Recall: `0.8`
- Support precision: `0.0`
- Support recall: `0.0`
- Retrieval recall ceiling: `0.667`
- Reasoning quality: `0.983`
- Adverse identification: `0.0`
- Predicted support docs: `doc_034.pdf`
- Predicted adverse docs: `doc_025.pdf, doc_032.pdf, doc_029.pdf, doc_027.pdf`

## Where the agent fails

- Precision is still low, which means the agent is presenting too many documents as meaningful precedents when they are only loosely related.
- Retrieval itself is missing part of the benchmark universe, which caps downstream recall before reasoning even starts.
- Adverse precedent coverage is weak, which is especially risky because the agent can sound more favorable than the corpus actually is.

## What I would fix first

- Tighten the precedent classifier first: add issue tags and a second-stage reranker so support/adverse labels reflect holdings, not isolated keywords.
- Improve retrieval next with hybrid matching or richer query expansion so synonym and paraphrase gaps do not hide relevant judgments.
- Expand the benchmark with more human-labeled prompts and gold documents, because recall claims are only as trustworthy as the benchmark coverage.

## Common Across Both Approaches?

This framework is intentionally common across the non-LLM and LLM approaches because both agents expose the same prompt-to-answer contract. The benchmark set, parsing rules, and metric definitions stay the same. What changes for the LLM path is methodology: you should usually run multiple samples and track variance because generation is less deterministic.
