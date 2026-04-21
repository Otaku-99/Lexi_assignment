from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from pypdf import PdfReader

from .types import ChunkRecord, DocumentRecord, ScoredChunk, ScoredDocument, SearchResult
from .utils import best_snippet, cosine_similarity, make_term_counts, normalize_whitespace


class CorpusIndex:
    CACHE_VERSION = 5

    def __init__(self, corpus_dir: str, cache_dir: str = ".cache") -> None:
        self.corpus_dir = Path(corpus_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "corpus_index.json"

        self.documents: list[DocumentRecord] = []
        self.chunks: list[ChunkRecord] = []
        self.doc_term_counts: dict[str, Counter[str]] = {}
        self.chunk_term_counts: dict[str, Counter[str]] = {}
        self.idf: dict[str, float] = {}
        self.doc_issue_tags: dict[str, list[str]] = {}

        self._load_or_build()

    def _load_or_build(self) -> None:
        pdf_files = sorted(self.corpus_dir.glob("*.pdf"))
        current_snapshot = {
            pdf.name: int(pdf.stat().st_mtime) for pdf in pdf_files
        }

        if self.cache_path.exists():
            cached = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if cached.get("snapshot") == current_snapshot and cached.get("cache_version") == self.CACHE_VERSION:
                self.documents = [DocumentRecord(**doc) for doc in cached["documents"]]
                self.chunks = [ChunkRecord(**chunk) for chunk in cached["chunks"]]
                self.doc_term_counts = {
                    key: Counter(value) for key, value in cached["doc_term_counts"].items()
                }
                self.chunk_term_counts = {
                    key: Counter(value) for key, value in cached["chunk_term_counts"].items()
                }
                self.idf = {key: float(value) for key, value in cached["idf"].items()}
                self.doc_issue_tags = cached.get("doc_issue_tags", {})
                return

        self._build(pdf_files)
        payload = {
            "cache_version": self.CACHE_VERSION,
            "snapshot": current_snapshot,
            "documents": [asdict(doc) for doc in self.documents],
            "chunks": [asdict(chunk) for chunk in self.chunks],
            "doc_term_counts": {key: dict(value) for key, value in self.doc_term_counts.items()},
            "chunk_term_counts": {key: dict(value) for key, value in self.chunk_term_counts.items()},
            "idf": self.idf,
            "doc_issue_tags": self.doc_issue_tags,
        }
        self.cache_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _build(self, pdf_files: list[Path]) -> None:
        self.documents = []
        self.chunks = []

        for pdf in pdf_files:
            reader = PdfReader(str(pdf))
            raw_page_texts = [page.extract_text() or "" for page in reader.pages]
            page_texts = [normalize_whitespace(text) for text in raw_page_texts]
            full_text = "\n".join(page_texts)
            title, citation_line, court, date_line = self._extract_header(raw_page_texts)

            doc = DocumentRecord(
                doc_id=pdf.stem,
                file_name=pdf.name,
                title=title or pdf.stem,
                citation_line=citation_line,
                court=court,
                date_line=date_line,
                text=full_text,
                pages=len(page_texts),
            )
            self.documents.append(doc)
            self.chunks.extend(self._chunk_document(doc, page_texts))

        self.doc_term_counts = {
            doc.doc_id: make_term_counts(
                f"{doc.title} {doc.citation_line} {doc.court} {doc.date_line} {doc.text}"
            )
            for doc in self.documents
        }
        self.chunk_term_counts = {
            chunk.chunk_id: make_term_counts(f"{chunk.title} {chunk.text}") for chunk in self.chunks
        }
        self.doc_issue_tags = {
            doc.doc_id: self._extract_issue_tags(f"{doc.title} {doc.citation_line} {doc.text}")
            for doc in self.documents
        }
        self.idf = self._compute_idf()

    def _extract_header(self, raw_page_texts: list[str]) -> tuple[str, str, str, str]:
        joined = "\n".join(raw_page_texts[:2])
        lines = [line.strip() for line in joined.splitlines() if line.strip()]
        title = self._clean_header_line(lines[0]) if lines else ""
        citation_line = self._clean_header_line(lines[1]) if len(lines) > 1 else ""

        court = ""
        date_line = ""
        for line in lines[:20]:
            upper = line.upper()
            if "HIGH COURT" in upper or "SUPREME COURT" in upper or "TRIBUNAL" in upper or "COURT OF" in upper:
                court = self._clean_header_line(line)
                break

        for line in lines[:20]:
            if re.search(r"\b\d{1,2}\s+\w+\s*,?\s+\d{4}\b", line) or "DATED" in line.upper() or "Pronounced on" in line:
                date_line = self._clean_header_line(line)
                break

        return title, citation_line, court, date_line

    def _clean_header_line(self, line: str) -> str:
        clean = normalize_whitespace(line)
        clean = re.sub(r"Indian Kanoon.*", "", clean, flags=re.IGNORECASE).strip()
        return clean[:220]

    def _chunk_document(self, doc: DocumentRecord, page_texts: list[str], target_words: int = 220, overlap_words: int = 40) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        current_words: list[str] = []
        current_page_start = 1
        current_page_end = 1
        chunk_index = 0

        for page_number, page_text in enumerate(page_texts, start=1):
            words = page_text.split()
            if not words:
                continue

            cursor = 0
            while cursor < len(words):
                if not current_words:
                    current_page_start = page_number
                take = min(target_words - len(current_words), len(words) - cursor)
                current_words.extend(words[cursor: cursor + take])
                current_page_end = page_number
                cursor += take

                if len(current_words) >= target_words:
                    chunk_text = " ".join(current_words)
                    chunks.append(
                        ChunkRecord(
                            chunk_id=f"{doc.doc_id}_chunk_{chunk_index}",
                            doc_id=doc.doc_id,
                            file_name=doc.file_name,
                            title=doc.title,
                            page_start=current_page_start,
                            page_end=current_page_end,
                            text=chunk_text,
                        )
                    )
                    chunk_index += 1
                    current_words = current_words[-overlap_words:]
                    current_page_start = current_page_end

        if current_words:
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{doc.doc_id}_chunk_{chunk_index}",
                    doc_id=doc.doc_id,
                    file_name=doc.file_name,
                    title=doc.title,
                    page_start=current_page_start,
                    page_end=current_page_end,
                    text=" ".join(current_words),
                )
            )

        return chunks

    def _compute_idf(self) -> dict[str, float]:
        df: defaultdict[str, int] = defaultdict(int)
        populations = list(self.doc_term_counts.values()) + list(self.chunk_term_counts.values())

        for counts in populations:
            for term in counts:
                df[term] += 1

        total = max(len(populations), 1)
        return {term: math.log((1 + total) / (1 + freq)) + 1.0 for term, freq in df.items()}

    def search(self, query: str, top_k_docs: int = 8, top_k_chunks: int = 16) -> SearchResult:
        query_counts = make_term_counts(query)
        query_issue_tags = self._extract_issue_tags(query)

        scored_docs = []
        for doc in self.documents:
            score, matched_terms = cosine_similarity(query_counts, self.doc_term_counts[doc.doc_id], self.idf)
            if score <= 0:
                continue
            issue_overlap = len(set(query_issue_tags) & set(self.doc_issue_tags.get(doc.doc_id, [])))
            score += self._metadata_bonus(doc, query)
            score += issue_overlap * 0.03
            scored_docs.append(
                ScoredDocument(
                    doc_id=doc.doc_id,
                    file_name=doc.file_name,
                    title=doc.title,
                    score=score,
                    matched_terms=matched_terms,
                    snippet=best_snippet(doc.text, matched_terms),
                )
            )
        scored_docs.sort(key=lambda item: item.score, reverse=True)
        top_doc_ids = {doc.doc_id for doc in scored_docs[: max(top_k_docs * 2, 10)]}

        scored_chunks = []
        doc_chunk_scores: dict[str, list[float]] = defaultdict(list)
        doc_chunk_terms: dict[str, set[str]] = defaultdict(set)
        for chunk in self.chunks:
            score, matched_terms = cosine_similarity(query_counts, self.chunk_term_counts[chunk.chunk_id], self.idf)
            if score <= 0:
                continue
            if chunk.doc_id in top_doc_ids:
                score *= 1.08
                doc_chunk_scores[chunk.doc_id].append(score)
                doc_chunk_terms[chunk.doc_id].update(matched_terms)
            scored_chunks.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    file_name=chunk.file_name,
                    title=chunk.title,
                    score=score,
                    matched_terms=matched_terms,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    snippet=best_snippet(chunk.text, matched_terms),
                )
            )
        scored_chunks.sort(key=lambda item: item.score, reverse=True)

        reranked_docs = []
        for doc in scored_docs[: max(top_k_docs * 3, 12)]:
            top_chunk_scores = sorted(doc_chunk_scores.get(doc.doc_id, []), reverse=True)[:3]
            chunk_bonus = sum(top_chunk_scores) * 0.12
            term_bonus = min(len(doc_chunk_terms.get(doc.doc_id, set())), 6) * 0.01
            reranked_docs.append(
                ScoredDocument(
                    doc_id=doc.doc_id,
                    file_name=doc.file_name,
                    title=doc.title,
                    score=doc.score + chunk_bonus + term_bonus,
                    matched_terms=doc.matched_terms,
                    snippet=doc.snippet,
                )
            )
        reranked_docs.sort(key=lambda item: item.score, reverse=True)

        return SearchResult(
            query=query,
            documents=reranked_docs[:top_k_docs],
            chunks=scored_chunks[:top_k_chunks],
        )

    def _metadata_bonus(self, doc: DocumentRecord, query: str) -> float:
        lowered_query = query.lower()
        bonus = 0.0
        for field in [doc.title, doc.citation_line, doc.court]:
            lowered = field.lower()
            if lowered and lowered in lowered_query:
                bonus += 0.12
        if "supreme court" in doc.court.lower():
            bonus += 0.02
        return bonus

    def _extract_issue_tags(self, text: str) -> list[str]:
        lowered = text.lower()
        issue_map = {
            "licence_defect": ["driving licence", "driving license", "license", "licence", "unlicensed"],
            "policy_breach": ["breach", "policy", "not liable", "exonerated", "defence", "defense"],
            "pay_and_recover": ["pay and recover", "recover from the owner", "third party", "recover the same from the insured"],
            "contributory_negligence": ["contributory negligence", "contributory", "rash and negligent", "claimant at fault", "deduction for negligence"],
            "commercial_vehicle": ["commercial vehicle", "goods carriage", "transport vehicle", "truck", "lorry", "transport endorsement"],
            "fatal_claim": ["death", "deceased", "fatal"],
            "compensation": ["compensation", "multiplier", "loss of dependency", "income"],
            "insurer_liability": ["insurer liable", "insurance company liable", "statutory liability", "award first"],
            "owner_knowledge": ["owner entrusted", "conscious breach", "wilful breach", "knowingly allowed"],
        }
        return [tag for tag, patterns in issue_map.items() if any(pattern in lowered for pattern in patterns)]

    def stats(self) -> dict[str, int | str]:
        return {
            "corpus_dir": os.fspath(self.corpus_dir),
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "cache_path": os.fspath(self.cache_path),
        }
