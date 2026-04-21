from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .planner import QueryPlanner
from .retrieval import CorpusIndex
from .synthesizer import AnswerSynthesizer
from .types import AgentResponse


class LegalResearchAgent:
    def __init__(self, corpus_dir: str, cache_dir: str = ".cache") -> None:
        self.index = CorpusIndex(corpus_dir=corpus_dir, cache_dir=cache_dir)
        self.planner = QueryPlanner()
        self.synthesizer = AnswerSynthesizer()

    def run(self, prompt: str) -> AgentResponse:
        plan = self.planner.plan(prompt)

        search_results = []
        for search_query in plan.search_queries:
            result = self.index.search(search_query, top_k_docs=8, top_k_chunks=16)
            search_results.append(result)

        response = self.synthesizer.synthesize(
            prompt=prompt,
            plan=plan,
            search_results=search_results,
            corpus_stats=self.index.stats(),
        )
        return response

    def stats(self) -> dict[str, Any]:
        return self.index.stats()

    def debug_snapshot(self) -> dict[str, Any]:
        return {
            "stats": self.index.stats(),
            "sample_documents": [asdict(doc) for doc in self.index.documents[:3]],
        }

