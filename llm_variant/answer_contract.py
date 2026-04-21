from __future__ import annotations

import re
from dataclasses import dataclass

from legal_agent.types import SearchResult


@dataclass(frozen=True)
class KnownAuthority:
    file_name: str
    title: str
    normalized_title: str


def normalize_llm_answer(answer: str, search_results: list[SearchResult]) -> str:
    known = _known_authorities(search_results)
    support = _normalize_section(
        answer=answer,
        heading="### Supporting Precedents",
        next_heading="### Adverse Precedents",
        known=known,
    )
    adverse = _normalize_section(
        answer=answer,
        heading="### Adverse Precedents",
        next_heading="### Strategy Recommendation",
        known=known,
    )

    updated = answer
    if support is not None:
        updated = _replace_section(updated, "### Supporting Precedents", "### Adverse Precedents", support)
    if adverse is not None:
        updated = _replace_section(updated, "### Adverse Precedents", "### Strategy Recommendation", adverse)
    return updated


def _known_authorities(search_results: list[SearchResult]) -> list[KnownAuthority]:
    seen: set[str] = set()
    authorities: list[KnownAuthority] = []
    for result in search_results:
        for doc in result.documents:
            if doc.file_name in seen:
                continue
            seen.add(doc.file_name)
            authorities.append(
                KnownAuthority(
                    file_name=doc.file_name,
                    title=doc.title,
                    normalized_title=_normalize_name(doc.title),
                )
            )
    return authorities


def _normalize_section(
    answer: str,
    heading: str,
    next_heading: str,
    known: list[KnownAuthority],
) -> str | None:
    section_text = _extract_section(answer, heading, next_heading)
    if section_text is None:
        return None

    lines = section_text.splitlines()
    normalized_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("- "):
            normalized_lines.append(line)
            continue
        if "`doc_" in stripped:
            normalized_lines.append(line)
            continue

        matched = _match_authority(stripped, known)
        if matched is None:
            normalized_lines.append(line)
            continue

        bullet = stripped[2:].strip()
        normalized_lines.append(f"- `{matched.file_name}` — {bullet}")
    return "\n".join(normalized_lines).strip()


def _extract_section(answer: str, heading: str, next_heading: str) -> str | None:
    start = answer.find(heading)
    if start == -1:
        return None
    start += len(heading)
    end = answer.find(next_heading, start)
    if end == -1:
        return answer[start:].strip()
    return answer[start:end].strip()


def _replace_section(answer: str, heading: str, next_heading: str, new_section_text: str) -> str:
    start = answer.find(heading)
    if start == -1:
        return answer
    content_start = start + len(heading)
    end = answer.find(next_heading, content_start)
    if end == -1:
        return f"{answer[:content_start]}\n\n{new_section_text}\n"
    return f"{answer[:content_start]}\n\n{new_section_text}\n\n{answer[end:]}"


def _match_authority(line: str, known: list[KnownAuthority]) -> KnownAuthority | None:
    normalized_line = _normalize_name(line)
    if not normalized_line:
        return None

    best: KnownAuthority | None = None
    best_score = 0.0
    for authority in known:
        score = _title_match_score(normalized_line, authority.normalized_title)
        if score > best_score:
            best = authority
            best_score = score

    return best if best_score >= 0.72 else None


def _title_match_score(line: str, title: str) -> float:
    line_tokens = set(line.split())
    title_tokens = set(title.split())
    if not line_tokens or not title_tokens:
        return 0.0
    overlap = len(line_tokens & title_tokens)
    union = len(line_tokens | title_tokens)
    containment = overlap / max(1, len(title_tokens))
    jaccard = overlap / max(1, union)
    return max(containment, jaccard)


def _normalize_name(value: str) -> str:
    lowered = value.lower()
    lowered = lowered.replace("vs.", "vs").replace("v.", "v").replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\b(and|ors|another|anr|limited|ltd|co|company|on|the)\b", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered
