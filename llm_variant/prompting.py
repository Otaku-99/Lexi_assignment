from __future__ import annotations

from collections import defaultdict

from legal_agent.types import QueryPlan, SearchResult


def build_system_prompt() -> str:
    return (
        "You are a legal research assistant working only from a provided corpus of Indian motor accident judgments. "
        "Ground every claim in the retrieved materials. "
        "Be explicit about which cases support the client and which are adverse. "
        "If evidence is mixed, say so honestly. "
        "Do not invent citations. "
        "Prefer crisp, professional legal analysis over generic prose. "
        "When citing authority, cite the corpus document id exactly as provided, for example `doc_032.pdf`, and do not replace it with an external citation."
    )


def build_user_prompt(prompt: str, plan: QueryPlan, search_results: list[SearchResult]) -> str:
    parts = [
        "User request:",
        prompt,
        "",
        f"Planned mode: {plan.mode}",
        f"Rationale: {plan.rationale}",
        f"Issue tags: {', '.join(plan.issue_tags) if plan.issue_tags else 'none'}",
        "",
        "Retrieved evidence:",
    ]

    for result in search_results:
        parts.append(f"\nSearch query: {result.query}")
        parts.append("Top documents:")
        for doc in result.documents[:5]:
            parts.append(
                f"- {doc.file_name} | {doc.title} | score={doc.score:.3f} | matched={', '.join(doc.matched_terms) or 'n/a'}"
            )
            parts.append(f"  Snippet: {doc.snippet}")
        parts.append("Top chunks:")
        for chunk in result.chunks[:5]:
            parts.append(
                f"- {chunk.file_name} pages {chunk.page_start}-{chunk.page_end} | score={chunk.score:.3f} | matched={', '.join(chunk.matched_terms) or 'n/a'}"
            )
            parts.append(f"  Snippet: {chunk.snippet}")

    parts.extend(["", "Allowed corpus authorities:"])
    seen = set()
    for result in search_results:
        for doc in result.documents[:6]:
            if doc.file_name in seen:
                continue
            seen.add(doc.file_name)
            parts.append(f"- `{doc.file_name}` | {doc.title} | matched={', '.join(doc.matched_terms) or 'n/a'}")

    support_candidates, adverse_candidates = build_candidate_lists(search_results)
    parts.extend(["", "Suggested candidate precedents:"])
    parts.append("Support-leaning candidates:")
    for line in support_candidates:
        parts.append(line)
    parts.append("Adverse-leaning candidates:")
    for line in adverse_candidates:
        parts.append(line)

    parts.extend(
        [
            "",
            "Preferred behavior:",
            "- Choose precedents only from the allowed corpus authorities listed above.",
            "- Start from the suggested candidate precedents before considering other allowed authorities.",
            "- Unless the evidence is very strong, stay inside the suggested candidate precedents.",
            "- Prefer authorities that recur across queries or have the closest issue overlap in their matched terms and snippets.",
            "- Usually cite 2 to 4 supporting precedents and 1 to 3 adverse precedents, not a long list.",
            "- If a document is only broadly about motor accidents but not about the prompt's legal issue, do not include it.",
            "",
            "If the prompt is a deep research task, return markdown in exactly this structure:",
            "## Research Answer",
            "### Supporting Precedents",
            "- Each bullet must begin with a corpus id from the allowed list in this exact form: `doc_XXX.pdf` - **Case title**. Relevant extract: ...",
            "- For each precedent, explain which facts align and what principle it supports, using the retrieved snippet rather than general legal background.",
            "- Do not cite cases that are mentioned inside a retrieved judgment unless they are also listed as corpus documents above.",
            "### Adverse Precedents",
            "- Each bullet must begin with a corpus id from the allowed list in this exact form: `doc_XXX.pdf` - **Case title**. Relevant extract: ...",
            "- For each adverse precedent, explain the litigation risk and how it may be distinguished or countered.",
            "- If you cannot find a strong adverse precedent, say that explicitly instead of forcing a weak one.",
            "### Strategy Recommendation",
            "- Include prioritized arguments, realistic litigation risk, and what to emphasize before the tribunal.",
            "### Compensation View",
            "- Give a realistic range only if the prompt includes facts like age, income, and dependents.",
            "",
            "Use only the corpus document ids supplied in the retrieved evidence. If you are unsure, omit the precedent rather than inventing one.",
            "It is better to cite fewer but more accurate documents than to include a weak or generic match.",
            "If the prompt is a simpler corpus question, return a direct concise answer grounded in the retrieved documents.",
            "Do not mention that you are using an LLM, a model, or internal prompts.",
        ]
    )
    return "\n".join(parts)


def build_candidate_lists(search_results: list[SearchResult]) -> tuple[list[str], list[str]]:
    """
    Build a *suggested* shortlist to steer the LLM.

    Key design goal: do not rely only on the query phrasing (which can be neutral);
    incorporate doc title/snippet/matched_terms so adverse candidates tend to be
    genuine insurer-defence / policy-breach authorities.
    """

    support_scores: dict[str, float] = defaultdict(float)
    adverse_scores: dict[str, float] = defaultdict(float)
    titles: dict[str, str] = {}

    adverse_query_markers = {"not liable", "breach", "defence", "defense", "exonerated", "help the insurer", "conscious breach"}
    support_query_markers = {"pay and recover", "claimant", "compensation", "liable", "third party"}

    adverse_content_markers = {
        "not liable",
        "no liability",
        "exonerated",
        "exonerate",
        "breach",
        "breach of policy",
        "conscious breach",
        "wilful",
        "willful",
        "violation",
        "invalid licence",
        "invalid license",
        "no valid licence",
        "no valid license",
        "defence",
        "defense",
        "insurer",
    }
    support_content_markers = {
        "pay and recover",
        "third party",
        "third-party",
        "claimant",
        "compensation",
        "liable",
        "award",
        "satisfy the award",
        "section 149",
        "victim should not",
    }

    for result in search_results:
        lowered_query = result.query.lower()
        query_support_bias = any(marker in lowered_query for marker in support_query_markers)
        query_adverse_bias = any(marker in lowered_query for marker in adverse_query_markers)

        for rank, doc in enumerate(result.documents[:6]):
            titles[doc.file_name] = doc.title
            weight = max(0.2, 1.0 - rank * 0.12)

            haystack = " ".join(
                [
                    (doc.title or ""),
                    (doc.snippet or ""),
                    " ".join(doc.matched_terms or []),
                ]
            ).lower()

            support_hits = sum(marker in haystack for marker in support_content_markers)
            adverse_hits = sum(marker in haystack for marker in adverse_content_markers)

            # Bias slightly toward the query's intent when present, but mostly trust content.
            support_scores[doc.file_name] += weight * (0.6 + 0.25 * query_support_bias) + support_hits * 0.35
            adverse_scores[doc.file_name] += weight * (0.45 + 0.35 * query_adverse_bias) + adverse_hits * 0.45

    support = sorted(support_scores.items(), key=lambda item: item[1], reverse=True)[:5]
    adverse = sorted(adverse_scores.items(), key=lambda item: item[1], reverse=True)[:5]
    support_lines = [f"- `{file_name}` | {titles.get(file_name, '')}".rstrip() for file_name, _ in support]
    adverse_lines = [f"- `{file_name}` | {titles.get(file_name, '')}".rstrip() for file_name, _ in adverse]
    return support_lines or ["- none"], adverse_lines or ["- none"]
