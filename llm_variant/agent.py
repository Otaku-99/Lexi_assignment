from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "non_llm" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from legal_agent.planner import QueryPlanner
from legal_agent.retrieval import CorpusIndex
from legal_agent.types import AgentResponse

from .answer_contract import normalize_llm_answer
from .llm_client import LLMClient, LLMSettings
from .prompting import build_system_prompt, build_user_prompt


class LLMResearchAgent:
    def __init__(self, corpus_dir: str, cache_dir: str, llm_settings: LLMSettings) -> None:
        self.index = CorpusIndex(corpus_dir=corpus_dir, cache_dir=cache_dir)
        self.planner = QueryPlanner()
        self.client = LLMClient(llm_settings)
        self.llm_settings = llm_settings

    def run(self, prompt: str) -> AgentResponse:
        plan = self.planner.plan(prompt)
        search_results = []
        for search_query in plan.search_queries:
            search_results.append(self.index.search(search_query, top_k_docs=8, top_k_chunks=12))

        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(prompt, plan, search_results)
        answer = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        answer = normalize_llm_answer(answer, search_results)

        trace = [
            {
                "step": "planning",
                "mode": plan.mode,
                "rationale": plan.rationale,
                "issue_tags": plan.issue_tags,
            },
            {
                "step": "retrieval",
                "queries": [
                    {
                        "query": result.query,
                        "top_documents": [doc.file_name for doc in result.documents[:5]],
                        "top_chunks": [chunk.chunk_id for chunk in result.chunks[:5]],
                    }
                    for result in search_results
                ],
            },
            {
                "step": "llm",
                "provider": self.llm_settings.provider,
                "model": self.llm_settings.model,
            },
        ]

        return AgentResponse(
            answer_markdown=answer,
            mode=plan.mode,
            plan=plan,
            search_results=search_results,
            trace=trace,
        )

    def stats(self) -> dict[str, Any]:
        return self.index.stats()
