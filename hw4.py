"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek
from rank_bm25 import BM25Okapi

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
# =============================================================================

_CORPUS_COMPANIES: dict[str, str] = {}

_TOKEN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "did", "do", "does", "for", "from", "had", "has", "have", "how", "in",
    "into", "is", "it", "its", "may", "more", "most", "much", "of", "on", "or",
    "our", "per", "the", "their", "them", "there", "these", "this", "those", "to",
    "was", "were", "what", "when", "where", "which", "who", "whom", "will", "with",
    "would", "your", "annual", "call", "document", "earnings", "fiscal", "fy", "report",
    "review", "results", "year", "december", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november", "usd",
    "million", "millions", "thousand", "percent", "percentage", "share", "shares", "month",
    "months", "end", "ended", "company", "companies", "s",
}

_ANCHOR_STOPWORDS = _TOKEN_STOPWORDS | {
    "higher", "lower", "held", "hold", "ask", "asking", "according", "announced", "announce",
    "guide", "guided", "set", "give", "gives", "forecast", "forecasts", "earn", "earned",
    "spend", "spent", "number", "value", "total", "consolidated", "financial", "statement",
    "statements", "firm", "management", "investment", "q", "earning", "large", "generate",
    "asia", "pacific", "europe", "middle", "east", "africa", "americas", "america", "latin",
}

_TOKEN_ALIASES = {
    "revenue": {"revenue", "sales", "turnover"}, "sale": {"sale", "sales", "revenue", "turnover"},
    "employee": {"employee", "staff", "headcount", "workforce", "people"},
    "staff": {"employee", "staff", "headcount", "workforce", "people"},
    "headcount": {"employee", "staff", "headcount", "workforce", "people"},
    "borrow": {"borrow", "borrowing", "debt", "loan"}, "borrowing": {"borrow", "borrowing", "debt", "loan"},
    "inventory": {"inventory", "stock"}, "auditor": {"auditor", "audit", "audited"},
    "audit": {"auditor", "audit", "audited"}, "audited": {"auditor", "audit", "audited"},
    "headquartered": {"headquartered", "headquarter", "headquarters", "head office", "located", "location"},
    "headquarter": {"headquartered", "headquarter", "headquarters", "head office", "located", "location"},
    "headquarters": {"headquartered", "headquarter", "headquarters", "head office", "located", "location"},
    "city": {"city", "headquartered", "headquarter", "headquarters", "head office", "located", "location"},
    "cfo": {"cfo", "chief", "financial", "officer"}, "ceo": {"ceo", "chief", "executive", "officer"},
}

_QUERY_EXPANSIONS = (
    ("cash and cash equivalents", "cash equivalents balance sheet"),
    ("cash equivalents", "cash equivalent cash"),
    ("net accounts receivable", "accounts receivable receivables net"),
    ("accounts receivable", "receivables debtor"),
    ("long term borrowing", "long-term borrowings debt"),
    ("long-term borrowing", "long term borrowings debt"),
    ("borrowings", "borrowing debt loans"), ("total assets", "assets balance sheet"),
    ("total equity", "equity shareholders equity"),
    ("net cash from operating activities", "operating cash flow cash flows"),
    ("operating activities", "operating cash flow"), ("gross profit", "gross margin"),
    ("net income", "profit earnings"), ("operating income", "operating profit"),
    ("employees", "employee staff headcount workforce people"),
    ("chief financial officer", "cfo finance officer"), ("chief executive officer", "ceo"),
    ("audited", "auditor audit independent"), ("audit firm", "auditor independent auditor"),
    ("headquartered", "headquarters head office location city"),
    ("target price", "price target 12 month valuation"), ("investment rating", "rating buy hold reduce"),
    ("share repurchase", "buyback repurchase programme program"),
    ("revenue growth", "sales growth guidance outlook"),
)

_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
_LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}
_BROKER_WORDS = {"capital", "securities", "research", "equity", "broker", "bank", "partners"}
_REGION_ENTITY_KEYS = {
    "asia pacific", "europe middle east and africa", "middle east", "middle east and africa",
    "north america", "south america", "latin america", "americas",
}
_GENERIC_ENTITY_WORDS = {
    "how", "what", "which", "who", "where", "when", "why", "chief", "financial", "executive",
    "officer", "annual", "report", "fourth", "full", "year", "usd", "december", "january",
    "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
}


def _plain_text(node: Any) -> str:
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _entity_key(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().replace("&", " and ")
    words = re.findall(r"[a-z0-9]+", text)
    while words and words[-1] in _LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


def _canonical_token(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("s") and len(word) > 4 and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def _raw_tokens(text: Any) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold().replace("_", " ")
    return [_canonical_token(word) for word in re.findall(r"\d{4}|[a-z]+|\d+(?:\.\d+)?", normalized)]


def _tokenize(text: Any) -> list[str]:
    return [word for word in _raw_tokens(text) if word not in _TOKEN_STOPWORDS]


def _get_document_metadata(file_name: str, html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    heading = soup.find("h1")
    company = _plain_text(heading) if heading else ""
    if not company and soup.title:
        company = _plain_text(soup.title)
    company = re.sub(r"\s*\([^()]*\)\s*$", "", company).strip(" |—-\t")
    visible = soup.get_text(" ", strip=True)
    lowered = visible.casefold()

    # Earnings transcripts mention analysts' broker affiliations, so identify them first.
    if re.search(r"earnings (?:conference )?call|earnings call transcript", lowered):
        doc_type = "earnings_call"
    elif "equity research" in lowered or "results review" in lowered:
        doc_type = "research_report"
    else:
        doc_type = "annual_report"

    year_match = None
    if doc_type == "annual_report":
        year_match = re.search(r"annual\s+report\s+(?:fy\s*)?(20\d{2})", visible, re.IGNORECASE)
    elif doc_type == "earnings_call":
        year_match = re.search(r"full\s+year\s+(20\d{2})", visible, re.IGNORECASE)
    else:
        year_match = re.search(r"fy\s*(20\d{2})\s+results\s+review", visible, re.IGNORECASE)
    if not year_match:
        year_match = re.search(r"\b(?:FY\s*)?(20\d{2})\b", visible, re.IGNORECASE)
    if not year_match:
        raise ValueError(f"could not determine fiscal year for {file_name}")
    if not company:
        raise ValueError(f"could not determine company for {file_name}")
    return {"source": file_name, "company": company, "fiscal_year": int(year_match.group(1)),
            "doc_type": doc_type}


def _safe_span(cell: Any, name: str) -> int:
    try:
        return max(1, int(cell.get(name, 1)))
    except (TypeError, ValueError):
        return 1


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    ### YOUR CODE HERE
    root = BeautifulSoup(table_html, "lxml")
    table = root.find("table") or root
    caption_node = table.find("caption", recursive=False)
    caption = _plain_text(caption_node) if caption_node else ""
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    if not rows:
        return caption

    grid: list[list[tuple[Any, int, int] | None]] = [[] for _ in rows]
    for row_index, row in enumerate(rows):
        column = 0
        for cell in row.find_all(["th", "td"], recursive=False):
            while column < len(grid[row_index]) and grid[row_index][column] is not None:
                column += 1
            colspan, rowspan = _safe_span(cell, "colspan"), _safe_span(cell, "rowspan")
            for r in range(row_index, min(len(rows), row_index + rowspan)):
                while len(grid[r]) < column + colspan:
                    grid[r].append(None)
                for c in range(column, column + colspan):
                    if grid[r][c] is None:
                        grid[r][c] = (cell, row_index, column)
            column += colspan

    header_rows = {i for i, row in enumerate(rows) if row.find_parent("thead") is not None}
    if not header_rows:
        for i, row in enumerate(rows):
            cells = row.find_all(["th", "td"], recursive=False)
            if not cells or not all(cell.name == "th" for cell in cells):
                break
            header_rows.add(i)

    width = max((len(row) for row in grid), default=0)
    headers: list[str] = []
    for column in range(width):
        labels, seen = [], set()
        for row_index in sorted(header_rows):
            if column >= len(grid[row_index]) or grid[row_index][column] is None:
                continue
            cell, origin_row, origin_col = grid[row_index][column]
            if (origin_row, origin_col) != (row_index, column):
                continue
            text = _plain_text(cell)
            if text and text.casefold() not in seen:
                labels.append(text)
                seen.add(text.casefold())
        headers.append(" | ".join(labels))

    lines = [caption] if caption else []
    for row_index, row in enumerate(grid):
        if row_index in header_rows:
            continue
        cells, seen_origins = [], set()
        for column, placement in enumerate(row):
            if placement is None:
                continue
            cell, origin_row, origin_col = placement
            if origin_row != row_index or (origin_row, origin_col) in seen_origins:
                continue
            seen_origins.add((origin_row, origin_col))
            text = _plain_text(cell)
            if text:
                cells.append((column, cell, text))
        # A single cell spanning a section heading is not a data row.
        if len(cells) < 2:
            continue
        _, label_cell, row_label = cells[0]
        if _safe_span(label_cell, "colspan") > 1:
            continue
        for column, _, value in cells[1:]:
            column_header = headers[column] if column < len(headers) and headers[column] else f"Column {column + 1}"
            lines.append(f"{row_label} | {column_header}: {value}")
    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    ### YOUR CODE HERE
    meta = _get_document_metadata(file_name, html)
    _CORPUS_COMPANIES[_entity_key(meta["company"])] = meta["company"]

    soup = BeautifulSoup(html, "lxml")
    for unwanted in soup.find_all(["script", "style", "noscript", "nav", "footer", "svg"]):
        unwanted.decompose()
    blocks: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "p", "li", "table"]):
        if element.name != "table":
            if element.find_parent("table"):
                continue
            if element.name == "p" and element.find_parent("li"):
                continue
            if element.name == "li" and element.find_parent("li"):
                continue
            text = _plain_text(element)
        else:
            text = table_to_text(str(element))
        if text:
            blocks.append(text)
    if not blocks:
        blocks = [s for s in soup.get_text("\n", strip=True).splitlines() if s.strip()]
    body = "\n".join(blocks)

    header = (f"source: {meta['source']} | company: {meta['company']} | "
              f"fiscal_year: {meta['fiscal_year']} | doc_type: {meta['doc_type']}")
    body_limit = max(1, MAX_CHUNK_CHARS - len(header) - 1)
    pieces: list[str] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + body_limit)
        if end < len(body):
            floor = start + int(body_limit * 0.60)
            boundary = max(body.rfind("\n", floor, end), body.rfind(" ", floor, end))
            if boundary > start:
                end = boundary
        piece = body[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(body):
            break
        # Keep a little source text in both passages so facts are not split across chunks.
        desired = max(start + 1, end - 160)
        previous_space = max(body.rfind("\n", start + 1, desired), body.rfind(" ", start + 1, desired))
        start = previous_space + 1 if previous_space >= start else end
        while start < len(body) and body[start].isspace():
            start += 1
    return [Document(page_content=f"{header}\n{piece}", metadata=dict(meta)) for piece in pieces]


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    ### YOUR CODE HERE
    question = str(question or "")
    question_key = _entity_key(question)
    companies: list[str] = []
    seen = set()

    # Match corpus names first, but keep unknown named entities for abstention checks.
    for key, name in sorted(_CORPUS_COMPANIES.items(), key=lambda item: len(item[0]), reverse=True):
        if key and f" {key} " in f" {question_key} " and key not in seen:
            companies.append(name)
            seen.add(key)

    # Do not let a possessive suffix absorb a segment name into the company entity.
    word = r"[A-Z][a-z]+(?:[.-][A-Z]?[a-z]+)*"
    entity_pattern = re.compile(
        rf"\b{word}(?:\s+(?:(?i:and|of|the)|{word})){{1,4}}"
        rf"(?:\s+(?:Inc|Corp|Corporation|Ltd|Limited|PLC|LLC)\.?)?\b"
    )
    for match in entity_pattern.finditer(question):
        name = re.sub(r"['’]s$", "", match.group(0), flags=re.IGNORECASE).strip()
        key = _entity_key(name)
        words = set(key.split())
        if (not key or key in seen or key in _REGION_ENTITY_KEYS or words & _GENERIC_ENTITY_WORDS or
                words & _BROKER_WORDS or words & _MONTHS):
            continue
        companies.append(name)
        seen.add(key)

    year_match = re.search(r"\bFY\s*(20\d{2})\b", question, re.IGNORECASE)
    if not year_match:
        year_match = re.search(
            r"\b(?:31\s+)?(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
            question, re.IGNORECASE,
        )
    if not year_match:
        year_match = re.search(r"\b(20\d{2})\b", question)
    fiscal_year = int(year_match.group(1)) if year_match else None

    lowered = question.casefold()
    if re.search(r"earnings?\s+(?:conference\s+)?call|q[1-4]\s+fy\d{4}\s+earnings call", lowered):
        doc_type = "earnings_call"
    elif any(term in lowered for term in (
        "results review", "target price", "investment rating", "broker", "forecast", "research report"
    )):
        doc_type = "research_report"
    else:
        doc_type = "annual_report"
    return {"companies": companies, "fiscal_year": fiscal_year, "doc_type": doc_type}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    ### YOUR CODE HERE
    chunks = list(chunks or [])
    tokenized = [_tokenize(chunk.page_content) for chunk in chunks]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=0.55) if tokenized and any(tokenized) else None
    document_frequency: dict[str, int] = {}
    companies: dict[str, str] = {}
    for chunk, words in zip(chunks, tokenized):
        for word in set(words):
            document_frequency[word] = document_frequency.get(word, 0) + 1
        name = str((chunk.metadata or {}).get("company", "")).strip()
        key = _entity_key(name)
        if key:
            companies[key] = name
            _CORPUS_COMPANIES[key] = name
    return {"chunks": chunks, "tokenized": tokenized, "bm25": bm25,
            "document_frequency": document_frequency, "companies": companies}


def _aliases_for(token: str) -> set[str]:
    aliases = _TOKEN_ALIASES.get(token, {token})
    return {_canonical_token(word) for alias in aliases for word in _raw_tokens(alias)}


def _question_anchors(question: str, companies: list[str]) -> list[str]:
    company_tokens = {word for name in companies for word in _raw_tokens(name)}
    anchors = []
    for word in _raw_tokens(question):
        if (word in _ANCHOR_STOPWORDS or word in _MONTHS or word in company_tokens or
                word.isdigit() or re.fullmatch(r"20\d{2}", word)):
            continue
        if word not in anchors:
            anchors.append(word)
    return anchors


def _year_matches(metadata: dict, year: int | None) -> bool:
    if year is None:
        return True
    try:
        doc_year = int(metadata.get("fiscal_year"))
    except (TypeError, ValueError):
        return False
    doc_type = metadata.get("doc_type")
    if doc_type == "annual_report":
        return doc_year in {year, year + 1}
    if doc_type == "earnings_call":
        return doc_year in {year, year - 1}
    if doc_type == "research_report":
        # FY2025 reviews include FY2024 actuals and FY2026/FY2027 estimates.
        return doc_year in {year - 2, year - 1, year, year + 1}
    return False


def _term_is_present(term: str, tokens: set[str]) -> bool:
    return bool(_aliases_for(term) & tokens)


def _expand_query(question: str) -> list[str]:
    tokens = _tokenize(question)
    normalized = " ".join(_raw_tokens(question))
    for phrase, expansion in _QUERY_EXPANSIONS:
        phrase_tokens = " ".join(_raw_tokens(phrase))
        if phrase_tokens and phrase_tokens in normalized:
            tokens.extend(_tokenize(expansion))
    return tokens


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    ### YOUR CODE HERE
    if not isinstance(index, dict) or not index.get("chunks"):
        return []
    chunks: list[Document] = index["chunks"]
    parsed = parse_query(question)
    companies = parsed.get("companies") or []
    if not companies:
        return []

    company_lookup = index.get("companies", {})
    company_keys = []
    for name in companies:
        key = _entity_key(name)
        if key not in company_keys:
            company_keys.append(key)
    # Unknown or near-miss companies must not borrow evidence from a similarly named issuer.
    if any(key not in company_lookup for key in company_keys):
        return []

    year = parsed.get("fiscal_year")
    doc_type = parsed.get("doc_type")
    candidate_ids: dict[str, list[int]] = {key: [] for key in company_keys}
    for idx, chunk in enumerate(chunks):
        meta = chunk.metadata or {}
        key = _entity_key(meta.get("company", ""))
        if key in candidate_ids and meta.get("doc_type") == doc_type and _year_matches(meta, year):
            candidate_ids[key].append(idx)
    if any(not ids for ids in candidate_ids.values()):
        return []

    query_tokens = _expand_query(question)
    bm25 = index.get("bm25")
    scores = bm25.get_scores(query_tokens) if bm25 and query_tokens else [0.0] * len(chunks)
    anchors = _question_anchors(question, companies)
    document_frequency = index.get("document_frequency", {})

    # If every relevant form of a specific query term is absent from the corpus, abstain.
    for anchor in anchors:
        if not any(document_frequency.get(alias, 0) for alias in _aliases_for(anchor)):
            return []

    ranked_by_company: dict[str, list[tuple[float, int, set[str]]]] = {}
    for key, ids in candidate_ids.items():
        ranked = []
        for idx in ids:
            words = set(index["tokenized"][idx])
            anchor_hits = sum(_term_is_present(anchor, words) for anchor in anchors)
            score = float(scores[idx]) if idx < len(scores) else 0.0
            score += 1.25 * anchor_hits
            if year is not None and str(year) in chunks[idx].page_content:
                score += 1.5
            ranked.append((score, idx, words))
        ranked.sort(key=lambda item: item[0], reverse=True)
        ranked_by_company[key] = ranked

        # Check each comparison side independently. This rejects passages that mention
        # the company but do not contain enough of the requested fact to answer safely.
        if anchors:
            top_words = set().union(*(entry[2] for entry in ranked[:3])) if ranked else set()
            covered = sum(_term_is_present(anchor, top_words) for anchor in anchors)
            if covered / len(anchors) < 0.60:
                return []

    selected: list[Document] = []
    selected_ids = set()
    # Round-robin selection ensures a comparison gets evidence for each company.
    max_depth = max(1, MAX_RETRIEVED // max(1, len(company_keys)))
    for depth in range(max_depth):
        for key in company_keys:
            ranked = ranked_by_company[key]
            if depth < len(ranked) and ranked[depth][1] not in selected_ids:
                idx = ranked[depth][1]
                selected.append(chunks[idx])
                selected_ids.add(idx)
                if len(selected) >= MAX_RETRIEVED:
                    return selected

    remaining = sorted(
        (entry for ranked in ranked_by_company.values() for entry in ranked if entry[1] not in selected_ids),
        key=lambda item: item[0], reverse=True,
    )
    for _, idx, _ in remaining:
        selected.append(chunks[idx])
        selected_ids.add(idx)
        if len(selected) >= MAX_RETRIEVED:
            break
    return selected


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()
