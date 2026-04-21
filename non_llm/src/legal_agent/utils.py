from __future__ import annotations

import math
import re
from collections import Counter


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "under",
    "this",
    "these",
    "those",
    "their",
    "they",
    "them",
    "his",
    "her",
    "our",
    "your",
    "which",
    "who",
    "whom",
    "what",
    "when",
    "where",
    "how",
    "why",
    "into",
    "than",
    "then",
    "there",
    "here",
    "also",
    "such",
    "any",
    "all",
    "not",
    "no",
    "if",
    "but",
    "can",
    "may",
    "would",
    "should",
    "could",
    "shall",
    "do",
    "does",
    "did",
    "have",
    "had",
    "having",
    "been",
    "being",
}

ALIASES = {
    "licence": "license",
    "licences": "license",
    "licensed": "license",
    "unlicensed": "unlicense",
    "insurer": "insurance",
    "insured": "insurance",
    "insurers": "insurance",
    "claimants": "claimant",
    "claims": "claim",
    "vehicles": "vehicle",
    "trucks": "truck",
    "goods": "goods",
    "carriages": "carriage",
    "negligently": "negligent",
    "compensate": "compensation",
    "compensated": "compensation",
    "compensating": "compensation",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_token(token: str) -> str:
    normalized = token.lower().strip("-/")
    normalized = ALIASES.get(normalized, normalized)
    return ALIASES.get(normalized, normalized)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9/-]{1,}", text.lower())
    normalized = [normalize_token(token) for token in tokens]
    return [token for token in normalized if token and token not in STOPWORDS]


def make_term_counts(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def cosine_similarity(query_counts: Counter[str], doc_counts: Counter[str], idf: dict[str, float]) -> tuple[float, list[str]]:
    shared = [term for term in query_counts if term in doc_counts]
    if not shared:
        return 0.0, []

    dot = 0.0
    query_norm = 0.0
    doc_norm = 0.0

    for term, q_count in query_counts.items():
        weight = idf.get(term, 1.0)
        q_value = (1.0 + math.log(q_count)) * weight
        query_norm += q_value * q_value
        d_count = doc_counts.get(term, 0)
        if d_count:
            d_value = (1.0 + math.log(d_count)) * weight
            dot += q_value * d_value

    for term, d_count in doc_counts.items():
        weight = idf.get(term, 1.0)
        d_value = (1.0 + math.log(d_count)) * weight
        doc_norm += d_value * d_value

    if not query_norm or not doc_norm:
        return 0.0, []

    score = dot / math.sqrt(query_norm * doc_norm)
    return score, sorted(shared)[:8]


def best_snippet(text: str, matched_terms: list[str], window: int = 320) -> str:
    clean = normalize_whitespace(text)
    if not clean:
        return ""
    if not matched_terms:
        return clean[:window]

    lowered = clean.lower()
    positions = [lowered.find(term.lower()) for term in matched_terms if lowered.find(term.lower()) != -1]
    if not positions:
        return clean[:window]

    start = max(0, min(positions) - 120)
    end = min(len(clean), start + window)
    return clean[start:end]
