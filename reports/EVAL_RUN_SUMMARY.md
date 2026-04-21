# Evaluation Run Summary (Reported Numbers)

This file captures the **exact summary JSON** outputs shared in the Deliverables discussion, split by backend.

## non_llm (sample_runs = 1)

```json
{
  "backend": "non_llm",
  "provider": null,
  "model": null,
  "sample_runs": 1,
  "precision": 0.637,
  "recall": 0.725,
  "reasoning_quality": 0.971,
  "adverse_identification": 0.167,
  "retrieval_recall_ceiling": 0.806,
  "precision_stddev": 0.0,
  "recall_stddev": 0.0,
  "reasoning_stddev": 0.0,
  "adverse_stddev": 0.0
}
```

## llm (provider = ollama, model = qwen2.5:3b-instruct, sample_runs = 3)

```json
{
  "backend": "llm",
  "provider": "ollama",
  "model": "qwen2.5:3b-instruct",
  "sample_runs": 3,
  "precision": 0.444,
  "recall": 0.175,
  "reasoning_quality": 0.752,
  "adverse_identification": 0.0,
  "retrieval_recall_ceiling": 0.917,
  "precision_stddev": 0.039,
  "recall_stddev": 0.0,
  "reasoning_stddev": 0.067,
  "adverse_stddev": 0.0
}
```

