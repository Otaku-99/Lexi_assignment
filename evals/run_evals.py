from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "non_llm" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_agent import LegalResearchAgent

from llm_variant.agent import LLMResearchAgent
from llm_variant.llm_client import LLMSettings

from .benchmark_cases import BENCHMARK_CASES


@dataclass
class ParsedLine:
    text: str
    docs: list[str]


@dataclass
class ParsedAnswer:
    support_docs: list[str]
    adverse_docs: list[str]
    support_lines: list[ParsedLine]
    adverse_lines: list[ParsedLine]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark evals for the legal research agents.")
    parser.add_argument(
        "--backend",
        choices=["non_llm", "llm"],
        default="non_llm",
        help="Which agent backend to evaluate.",
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "ollama"],
        default="openai",
        help="Provider for the llm backend.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for the llm backend. Falls back to a small default if omitted.",
    )
    parser.add_argument(
        "--sample-runs",
        type=int,
        default=1,
        help="How many times to run each prompt. Useful for measuring llm variance.",
    )
    return parser.parse_args()


def make_agent(args: argparse.Namespace) -> Any:
    corpus_dir = str(ROOT / "lexi_research_take_home_assessment_docs")
    cache_dir = str(ROOT / ".cache")

    if args.backend == "non_llm":
        return LegalResearchAgent(corpus_dir=corpus_dir, cache_dir=cache_dir)

    model = args.model or ("gpt-4.1-mini" if args.provider == "openai" else "qwen2.5:3b-instruct")
    return LLMResearchAgent(
        corpus_dir=corpus_dir,
        cache_dir=cache_dir,
        llm_settings=LLMSettings(provider=args.provider, model=model),
    )


def parse_section(answer: str, heading: str, next_heading: str | None = None) -> str:
    start = answer.find(heading)
    if start == -1:
        return ""
    start += len(heading)
    end = answer.find(next_heading, start) if next_heading else -1
    return answer[start:end].strip() if end != -1 else answer[start:].strip()


def parse_docs(section_text: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"(doc_\d+\.pdf)", section_text)))


def parse_bullets(section_text: str) -> list[ParsedLine]:
    bullets: list[str] = []
    current: list[str] = []

    for raw_line in section_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("- "):
            if current:
                bullets.append(" ".join(current))
            current = [line[2:].strip()]
        elif current:
            current.append(line)

    if current:
        bullets.append(" ".join(current))

    return [ParsedLine(text=bullet, docs=parse_docs(bullet)) for bullet in bullets]


def parse_answer(answer: str) -> ParsedAnswer:
    support_text = parse_section(answer, "### Supporting Precedents", "### Adverse Precedents")
    adverse_text = parse_section(answer, "### Adverse Precedents", "### Strategy Recommendation")
    support_lines = parse_bullets(support_text)
    adverse_lines = parse_bullets(adverse_text)
    return ParsedAnswer(
        support_docs=parse_docs(support_text),
        adverse_docs=parse_docs(adverse_text),
        support_lines=support_lines,
        adverse_lines=adverse_lines,
    )


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def precision(predicted: list[str], gold: list[str]) -> float:
    predicted_set = set(predicted)
    if not predicted_set:
        return 0.0
    return safe_divide(len(predicted_set & set(gold)), len(predicted_set))


def recall(predicted: list[str], gold: list[str]) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 1.0
    return safe_divide(len(set(predicted) & gold_set), len(gold_set))


def f1_score(precision_value: float, recall_value: float) -> float:
    if precision_value + recall_value == 0:
        return 0.0
    return 2 * precision_value * recall_value / (precision_value + recall_value)


def retrieval_union(response: Any) -> list[str]:
    docs: list[str] = []
    for result in response.search_results:
        docs.extend(doc.file_name for doc in result.documents[:8])
    return list(dict.fromkeys(docs))


def line_reasoning_score(
    line: ParsedLine,
    issue_terms: list[str],
    fact_terms: list[str],
    stance_terms: list[str],
) -> float:
    lowered = line.text.lower()
    components = [
        1.0 if line.docs else 0.0,
        1.0 if "relevant extract:" in lowered else 0.0,
        1.0 if any(term in lowered for term in issue_terms) else 0.0,
        1.0 if any(term in lowered for term in fact_terms) else 0.0,
        1.0 if any(term in lowered for term in stance_terms) else 0.0,
    ]
    return sum(components) / len(components)


def reasoning_quality(parsed: ParsedAnswer, case: dict[str, Any]) -> dict[str, float]:
    support_scores = [
        line_reasoning_score(
            line=line,
            issue_terms=case["support_reasoning_terms"],
            fact_terms=case["reasoning_fact_terms"],
            stance_terms=case["expected_support_signal_terms"],
        )
        for line in parsed.support_lines
    ]
    adverse_scores = [
        line_reasoning_score(
            line=line,
            issue_terms=case["adverse_reasoning_terms"],
            fact_terms=case["reasoning_fact_terms"],
            stance_terms=case["expected_adverse_signal_terms"],
        )
        for line in parsed.adverse_lines
    ]

    support_score = statistics.mean(support_scores) if support_scores else 0.0
    adverse_score = statistics.mean(adverse_scores) if adverse_scores else 0.0
    section_balance = 1.0 if parsed.support_lines and parsed.adverse_lines else 0.0

    return {
        "support_reasoning": support_score,
        "adverse_reasoning": adverse_score,
        "reasoning_quality": statistics.mean([support_score, adverse_score, section_balance]),
    }


def adverse_identification(parsed: ParsedAnswer, case: dict[str, Any]) -> dict[str, float]:
    adverse_recall = recall(parsed.adverse_docs, case["gold_adverse_docs"])
    adverse_precision = precision(parsed.adverse_docs, case["gold_adverse_docs"])
    return {
        "adverse_identification": adverse_recall,
        "adverse_precision": adverse_precision,
        "adverse_f1": f1_score(adverse_precision, adverse_recall),
    }


def evaluate_case(response: Any, case: dict[str, Any]) -> dict[str, Any]:
    parsed = parse_answer(response.answer_markdown)
    relevant_predicted = list(dict.fromkeys(parsed.support_docs + parsed.adverse_docs))
    retrieved_docs = retrieval_union(response)

    precision_value = precision(relevant_predicted, case["gold_relevant_docs"])
    recall_value = recall(relevant_predicted, case["gold_relevant_docs"])
    support_precision = precision(parsed.support_docs, case["gold_support_docs"])
    support_recall = recall(parsed.support_docs, case["gold_support_docs"])
    retrieval_recall = recall(retrieved_docs, case["retrieval_must_cover_docs"])
    reasoning = reasoning_quality(parsed, case)
    adverse = adverse_identification(parsed, case)

    return {
        "id": case["id"],
        "description": case["description"],
        "mode": response.mode,
        "precision": round(precision_value, 3),
        "recall": round(recall_value, 3),
        "support_precision": round(support_precision, 3),
        "support_recall": round(support_recall, 3),
        "retrieval_recall_ceiling": round(retrieval_recall, 3),
        "reasoning_quality": round(reasoning["reasoning_quality"], 3),
        "support_reasoning": round(reasoning["support_reasoning"], 3),
        "adverse_reasoning": round(reasoning["adverse_reasoning"], 3),
        "adverse_identification": round(adverse["adverse_identification"], 3),
        "adverse_precision": round(adverse["adverse_precision"], 3),
        "adverse_f1": round(adverse["adverse_f1"], 3),
        "predicted_support_docs": parsed.support_docs,
        "predicted_adverse_docs": parsed.adverse_docs,
        "predicted_relevant_docs": relevant_predicted,
        "retrieved_docs": retrieved_docs,
        "gold_support_docs": case["gold_support_docs"],
        "gold_adverse_docs": case["gold_adverse_docs"],
        "gold_relevant_docs": case["gold_relevant_docs"],
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def variance(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def run() -> dict[str, Any]:
    args = parse_args()
    agent = make_agent(args)
    run_samples: list[dict[str, Any]] = []

    for sample_index in range(args.sample_runs):
        case_results = []
        for case in BENCHMARK_CASES:
            response = agent.run(case["prompt"])
            case_results.append(evaluate_case(response, case))
        run_samples.append({"sample_index": sample_index + 1, "cases": case_results})

    aggregated_cases = []
    for case in BENCHMARK_CASES:
        case_runs = [
            next(item for item in sample["cases"] if item["id"] == case["id"])
            for sample in run_samples
        ]
        aggregated_cases.append(
            {
                "id": case["id"],
                "description": case["description"],
                "mode": case_runs[-1]["mode"],
                "precision": round(mean([run["precision"] for run in case_runs]), 3),
                "recall": round(mean([run["recall"] for run in case_runs]), 3),
                "support_precision": round(mean([run["support_precision"] for run in case_runs]), 3),
                "support_recall": round(mean([run["support_recall"] for run in case_runs]), 3),
                "retrieval_recall_ceiling": round(mean([run["retrieval_recall_ceiling"] for run in case_runs]), 3),
                "reasoning_quality": round(mean([run["reasoning_quality"] for run in case_runs]), 3),
                "support_reasoning": round(mean([run["support_reasoning"] for run in case_runs]), 3),
                "adverse_reasoning": round(mean([run["adverse_reasoning"] for run in case_runs]), 3),
                "adverse_identification": round(mean([run["adverse_identification"] for run in case_runs]), 3),
                "adverse_precision": round(mean([run["adverse_precision"] for run in case_runs]), 3),
                "adverse_f1": round(mean([run["adverse_f1"] for run in case_runs]), 3),
                "predicted_support_docs": case_runs[-1]["predicted_support_docs"],
                "predicted_adverse_docs": case_runs[-1]["predicted_adverse_docs"],
                "predicted_relevant_docs": case_runs[-1]["predicted_relevant_docs"],
                "retrieved_docs": case_runs[-1]["retrieved_docs"],
                "gold_support_docs": case_runs[-1]["gold_support_docs"],
                "gold_adverse_docs": case_runs[-1]["gold_adverse_docs"],
                "gold_relevant_docs": case_runs[-1]["gold_relevant_docs"],
                "precision_stddev": round(variance([run["precision"] for run in case_runs]), 3),
                "recall_stddev": round(variance([run["recall"] for run in case_runs]), 3),
                "reasoning_stddev": round(variance([run["reasoning_quality"] for run in case_runs]), 3),
                "adverse_stddev": round(variance([run["adverse_identification"] for run in case_runs]), 3),
            }
        )

    summary = {
        "backend": args.backend,
        "provider": args.provider if args.backend == "llm" else None,
        "model": args.model if args.backend == "llm" else None,
        "sample_runs": args.sample_runs,
        "precision": round(mean([item["precision"] for item in aggregated_cases]), 3),
        "recall": round(mean([item["recall"] for item in aggregated_cases]), 3),
        "reasoning_quality": round(mean([item["reasoning_quality"] for item in aggregated_cases]), 3),
        "adverse_identification": round(mean([item["adverse_identification"] for item in aggregated_cases]), 3),
        "retrieval_recall_ceiling": round(mean([item["retrieval_recall_ceiling"] for item in aggregated_cases]), 3),
        "precision_stddev": round(mean([item["precision_stddev"] for item in aggregated_cases]), 3),
        "recall_stddev": round(mean([item["recall_stddev"] for item in aggregated_cases]), 3),
        "reasoning_stddev": round(mean([item["reasoning_stddev"] for item in aggregated_cases]), 3),
        "adverse_stddev": round(mean([item["adverse_stddev"] for item in aggregated_cases]), 3),
    }
    return {"summary": summary, "cases": aggregated_cases, "samples": run_samples}


def weakness_analysis(results: dict[str, Any]) -> list[str]:
    weaknesses: list[str] = []
    summary = results["summary"]
    if summary["precision"] < 0.7:
        weaknesses.append(
            "- Precision is still low, which means the agent is presenting too many documents as meaningful precedents when they are only loosely related."
        )
    if summary["recall"] < 0.7:
        weaknesses.append(
            "- Final-answer recall trails retrieval recall, so the system is dropping relevant cases during support-versus-adverse classification or final synthesis."
        )
    if summary["retrieval_recall_ceiling"] < 0.85:
        weaknesses.append(
            "- Retrieval itself is missing part of the benchmark universe, which caps downstream recall before reasoning even starts."
        )
    if summary["reasoning_quality"] < 0.75:
        weaknesses.append(
            "- Explanation quality is inconsistent: some bullets cite a case but do not clearly connect the fact pattern, the legal issue, and the litigation posture."
        )
    if summary["adverse_identification"] < 0.7:
        weaknesses.append(
            "- Adverse precedent coverage is weak, which is especially risky because the agent can sound more favorable than the corpus actually is."
        )
    if summary["sample_runs"] > 1 and any(
        summary[key] > 0.05 for key in ["precision_stddev", "recall_stddev", "reasoning_stddev", "adverse_stddev"]
    ):
        weaknesses.append(
            "- The llm-backed path shows measurable run-to-run variance, so single-run scores are not enough for regression decisions."
        )
    if not weaknesses:
        weaknesses.append("- The automated benchmark did not surface a dominant failure mode, but the gold set is still small and should be expanded.")
    return weaknesses


def fix_analysis(results: dict[str, Any]) -> list[str]:
    summary = results["summary"]
    fixes: list[str] = []
    if summary["precision"] < 0.7 or summary["adverse_identification"] < 0.7:
        fixes.append(
            "- Tighten the precedent classifier first: add issue tags and a second-stage reranker so support/adverse labels reflect holdings, not isolated keywords."
        )
    if summary["retrieval_recall_ceiling"] < 0.85:
        fixes.append(
            "- Improve retrieval next with hybrid matching or richer query expansion so synonym and paraphrase gaps do not hide relevant judgments."
        )
    if summary["reasoning_quality"] < 0.75:
        fixes.append(
            "- Add a stronger reasoning judge and train the answer format to always state facts, rule, and why the case helps or hurts the client."
        )
    fixes.append(
        "- Expand the benchmark with more human-labeled prompts and gold documents, because recall claims are only as trustworthy as the benchmark coverage."
    )
    return fixes


def write_report(results: dict[str, Any]) -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    backend = results["summary"]["backend"]
    report_stub = backend if backend == "non_llm" else f"{backend}_{results['summary']['provider']}"
    json_path = reports_dir / f"eval_results_{report_stub}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    common_note = (
        "This framework is intentionally common across the non-LLM and LLM approaches because both agents expose the same prompt-to-answer contract. "
        "The benchmark set, parsing rules, and metric definitions stay the same. What changes for the LLM path is methodology: you should usually run multiple samples and track variance because generation is less deterministic."
    )

    lines = [
        "# Evaluation Results",
        "",
        f"Backend: `{results['summary']['backend']}`",
        f"Sample runs per prompt: `{results['summary']['sample_runs']}`",
        "",
        "## Framework",
        "",
        "- Precision measures how many cited precedents are actually part of the benchmark's relevant set for that prompt.",
        "- Recall measures how much of the benchmark's relevant precedent set appears in the final answer.",
        "- Reasoning quality scores each explanation bullet for citation, quoted grounding, issue overlap, fact-pattern overlap, and stance clarity.",
        "- Adverse identification measures whether the agent surfaced the benchmark's adverse cases, with precision and F1 as supporting signals.",
        "- Retrieval recall ceiling is a diagnostic metric: it shows whether the retriever exposed the right cases before synthesis had a chance to cite them.",
        "",
        "## Summary",
        "",
        f"- Precision: `{results['summary']['precision']}`",
        f"- Recall: `{results['summary']['recall']}`",
        f"- Reasoning quality: `{results['summary']['reasoning_quality']}`",
        f"- Adverse identification: `{results['summary']['adverse_identification']}`",
        f"- Retrieval recall ceiling: `{results['summary']['retrieval_recall_ceiling']}`",
        "",
    ]

    if results["summary"]["sample_runs"] > 1:
        lines.extend(
            [
                "## Stability",
                "",
                f"- Precision stddev: `{results['summary']['precision_stddev']}`",
                f"- Recall stddev: `{results['summary']['recall_stddev']}`",
                f"- Reasoning stddev: `{results['summary']['reasoning_stddev']}`",
                f"- Adverse stddev: `{results['summary']['adverse_stddev']}`",
                "",
            ]
        )

    lines.extend(["## Per-case results", ""])
    for case in results["cases"]:
        lines.extend(
            [
                f"### {case['id']}",
                f"- Description: {case['description']}",
                f"- Mode: `{case['mode']}`",
                f"- Precision: `{case['precision']}`",
                f"- Recall: `{case['recall']}`",
                f"- Support precision: `{case['support_precision']}`",
                f"- Support recall: `{case['support_recall']}`",
                f"- Retrieval recall ceiling: `{case['retrieval_recall_ceiling']}`",
                f"- Reasoning quality: `{case['reasoning_quality']}`",
                f"- Adverse identification: `{case['adverse_identification']}`",
                f"- Predicted support docs: `{', '.join(case['predicted_support_docs']) or 'none'}`",
                f"- Predicted adverse docs: `{', '.join(case['predicted_adverse_docs']) or 'none'}`",
                "",
            ]
        )

    lines.extend(["## Where the agent fails", ""])
    lines.extend(weakness_analysis(results))
    lines.extend(["", "## What I would fix first", ""])
    lines.extend(fix_analysis(results))
    lines.extend(["", "## Common Across Both Approaches?", "", common_note, ""])

    md_path = reports_dir / "EVAL_REPORT.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        output = run()
    except Exception as exc:
        if "OPENAI_API_KEY" in str(exc) and os.environ.get("OPENAI_API_KEY") is None:
            raise RuntimeError("OPENAI_API_KEY is not set for the llm backend.") from exc
        raise
    write_report(output)
    print(json.dumps(output["summary"], indent=2))
