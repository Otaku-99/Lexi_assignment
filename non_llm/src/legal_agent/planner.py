from __future__ import annotations

import re

from .types import QueryPlan
from .utils import tokenize


class QueryPlanner:
    DEEP_RESEARCH_TERMS = {
        "precedent",
        "precedents",
        "supporting",
        "adverse",
        "distinguish",
        "counter",
        "strategy",
        "strategic",
        "risk",
        "liability",
        "compensation",
        "negligence",
        "contributory",
        "research",
        "analyse",
        "analyze",
        "arguments",
        "claim",
        "brief",
        "facts",
    }

    def plan(self, prompt: str) -> QueryPlan:
        normalized = prompt.strip()
        lowered = normalized.lower()
        tokens = tokenize(normalized)

        issue_tags = self._issue_tags(lowered)
        dense_fact_pattern = normalized.count("\n-") >= 2 or "key facts" in lowered or "client:" in lowered
        explicit_research = any(term in tokens for term in self.DEEP_RESEARCH_TERMS)
        question_count = normalized.count("?")

        if explicit_research or dense_fact_pattern or question_count > 1 or len(tokens) > 35:
            mode = "deep_research"
            rationale = "The prompt asks for precedent-driven legal analysis or includes a detailed factual brief, so the agent should run a multi-query research workflow."
            search_queries = self._build_research_queries(normalized, issue_tags)
        else:
            mode = "quick_answer"
            rationale = "The prompt looks like a narrower corpus question, so a concise retrieval-first answer is likely sufficient unless retrieval is weak."
            search_queries = [normalized]

        return QueryPlan(
            mode=mode,
            rationale=rationale,
            search_queries=search_queries,
            issue_tags=issue_tags,
        )

    def _issue_tags(self, lowered_prompt: str) -> list[str]:
        tags = []
        checks = {
            "licence_defect": ["driving licence", "driving license", "unlicensed", "license", "licence"],
            "policy_breach": ["policy", "breach", "void", "not liable", "exonerated", "defence"],
            "pay_and_recover": ["pay and recover"],
            "motor_accident": ["motor accident", "truck", "vehicle", "commercial vehicle", "goods carriage"],
            "commercial_vehicle": ["commercial vehicle", "goods carriage", "transport-company", "transport company", "truck"],
            "contributory_negligence": ["contributory negligence", "contributory", "negligence"],
            "adverse_scan": ["adverse", "risk", "risky", "against the claimant", "help the insurer", "unfavourable", "unfavorable"],
            "fatal_claim": ["death", "deceased", "fatal"],
            "compensation": ["compensation", "income", "dependents", "multiplier"],
        }

        for tag, patterns in checks.items():
            if any(pattern in lowered_prompt for pattern in patterns):
                tags.append(tag)
        return tags

    def _build_research_queries(self, prompt: str, issue_tags: list[str]) -> list[str]:
        cleaned = re.sub(r"\s+", " ", prompt).strip()
        queries = [cleaned]

        issue_query_parts = []
        if "licence_defect" in issue_tags:
            issue_query_parts.append("unlicensed driver valid driving licence insurance liability")
        if "pay_and_recover" in issue_tags or ("licence_defect" in issue_tags and "policy_breach" in issue_tags):
            issue_query_parts.append("pay and recover insurer liable despite breach of policy")
        if "licence_defect" in issue_tags and "policy_breach" in issue_tags:
            issue_query_parts.append("insurer not liable unlicensed driver conscious breach of policy owner entrusted vehicle")
        if "contributory_negligence" in issue_tags:
            issue_query_parts.append("contributory negligence truck accident claimant burden of proof")
            if "adverse_scan" in issue_tags:
                issue_query_parts.append("contributory negligence insurer defence claimant at fault deduction")
        if "commercial_vehicle" in issue_tags:
            issue_query_parts.append("commercial vehicle goods carriage transport company truck motor accident claim")
            if "adverse_scan" in issue_tags:
                issue_query_parts.append("commercial vehicle goods carriage truck insurer policy breach risk not liable")
        elif "motor_accident" in issue_tags:
            issue_query_parts.append("commercial vehicle truck motor accident claim")
        if "fatal_claim" in issue_tags and "compensation" in issue_tags:
            issue_query_parts.append("death claim multiplier dependents monthly income compensation")

        compressed = " ".join(tokenize(prompt)[:24])
        if compressed:
            queries.append(compressed)
        queries.extend(issue_query_parts)

        unique_queries = []
        seen = set()
        for query in queries:
            if query and query not in seen:
                unique_queries.append(query)
                seen.add(query)
        return unique_queries[:6]
