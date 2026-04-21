from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    file_name: str
    title: str
    page_start: int
    page_end: int
    text: str


@dataclass
class DocumentRecord:
    doc_id: str
    file_name: str
    title: str
    citation_line: str
    court: str
    date_line: str
    text: str
    pages: int


@dataclass
class ScoredDocument:
    doc_id: str
    file_name: str
    title: str
    score: float
    matched_terms: list[str]
    snippet: str


@dataclass
class ScoredChunk:
    chunk_id: str
    doc_id: str
    file_name: str
    title: str
    score: float
    matched_terms: list[str]
    page_start: int
    page_end: int
    snippet: str


@dataclass
class SearchResult:
    query: str
    documents: list[ScoredDocument]
    chunks: list[ScoredChunk]


@dataclass
class QueryPlan:
    mode: str
    rationale: str
    search_queries: list[str]
    issue_tags: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    answer_markdown: str
    mode: str
    plan: QueryPlan
    search_results: list[SearchResult]
    trace: list[dict]

