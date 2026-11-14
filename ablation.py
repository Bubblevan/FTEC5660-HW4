"""Run the public-set ablation sequence for FTEC5660 Homework 4.

The variants are cumulative: B0 is the assignment's naive RAG baseline, and each
V row adds one retrieval component. Model calls are sequential; identical prompts
are cached so the fixed, temperature-zero generator is not called twice for the
same question and context.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, NavigableString
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


VARIANTS = [
    {"id": "B0", "name": "Naive RAG", "tables": False, "production_chunks": False,
     "inline_header": False, "metadata": False, "tokenizer": "whitespace", "bm25_b": 0.75,
     "query_expansion": False, "evidence_boost": False, "per_company": False, "abstention": False,
     "change": "Tutorial 7 baseline: raw HTML text, default RecursiveCharacterTextSplitter, required metadata, default rank_bm25 parameters, whole-corpus top-5, and never abstain."},
    {"id": "V1", "name": "+ Table serialization", "tables": True, "production_chunks": False,
     "inline_header": False, "metadata": False, "tokenizer": "whitespace", "bm25_b": 0.75,
     "query_expansion": False, "evidence_boost": False, "per_company": False, "abstention": False,
     "change": "Convert tables to row-label/column-header/value text before the same splitter."},
    {"id": "V2", "name": "+ Query parsing and metadata filters", "tables": True, "production_chunks": False,
     "inline_header": False, "metadata": True, "tokenizer": "whitespace", "bm25_b": 0.75,
     "query_expansion": False, "evidence_boost": False, "per_company": False, "abstention": False,
     "change": "Parse company, fiscal year and document type; filter BM25 candidates by those metadata fields."},
    {"id": "V3", "name": "+ Structure-aware chunking", "tables": True, "production_chunks": True,
     "inline_header": False, "metadata": True, "tokenizer": "whitespace", "bm25_b": 0.75,
     "query_expansion": False, "evidence_boost": False, "per_company": False, "abstention": False,
     "change": "Replace the recursive splitter with the submitted structure-aware, 1,500-character chunker; the fixed generator's metadata header stays unchanged."},
    {"id": "V4", "name": "+ Inline contextual header", "tables": True, "production_chunks": True,
     "inline_header": True, "metadata": True, "tokenizer": "whitespace", "bm25_b": 0.75,
     "query_expansion": False, "evidence_boost": False, "per_company": False, "abstention": False,
     "change": "Prepend source, company, fiscal year and document type to each chunk's searchable text."},
    {"id": "V5", "name": "+ Domain-aware tokenization", "tables": True, "production_chunks": True,
     "inline_header": True, "metadata": True, "tokenizer": "domain", "bm25_b": 0.75,
     "query_expansion": False, "evidence_boost": False, "per_company": False, "abstention": False,
     "change": "Use the submitted tokenizer, which removes common words and normalizes simple plural forms."},
    {"id": "V6", "name": "+ BM25 length normalization", "tables": True, "production_chunks": True,
     "inline_header": True, "metadata": True, "tokenizer": "domain", "bm25_b": 0.55,
     "query_expansion": False, "evidence_boost": False, "per_company": False, "abstention": False,
     "change": "Tune BM25's b parameter from the library default 0.75 to the submitted value 0.55."},
    {"id": "V7", "name": "+ Query expansion", "tables": True, "production_chunks": True,
     "inline_header": True, "metadata": True, "tokenizer": "domain", "bm25_b": 0.55,
     "query_expansion": True, "evidence_boost": False, "per_company": False, "abstention": False,
     "change": "Add financial/report synonyms to the query terms without changing scores beyond BM25."},
    {"id": "V8", "name": "+ Evidence weighting", "tables": True, "production_chunks": True,
     "inline_header": True, "metadata": True, "tokenizer": "domain", "bm25_b": 0.55,
     "query_expansion": True, "evidence_boost": True, "per_company": False, "abstention": False,
     "change": "Boost passages that contain requested fact anchors and the requested year."},
    {"id": "V9", "name": "+ Balanced comparison retrieval", "tables": True, "production_chunks": True,
     "inline_header": True, "metadata": True, "tokenizer": "domain", "bm25_b": 0.55,
     "query_expansion": True, "evidence_boost": True, "per_company": True, "abstention": False,
     "change": "For multi-company questions, select passages round-robin so each company gets evidence."},
    {"id": "V10", "name": "+ Evidence-based abstention", "tables": True, "production_chunks": True,
     "inline_header": True, "metadata": True, "tokenizer": "domain", "bm25_b": 0.55,
     "query_expansion": True, "evidence_boost": True, "per_company": True, "abstention": True,
     "change": "Apply the submitted answerability checks; this is the final hw4.py retrieval pipeline."},
]


def _chunks_from_html(corpus: dict[str, str], use_table_text: bool,
                      production_chunks: bool, inline_header: bool = False) -> tuple[list[Document], dict[str, list[Document]]]:
    chunks_by_file: dict[str, list[Document]] = {}
    for file_name, html in corpus.items():
        if production_chunks:
            production = hw4.chunk_document(file_name, html)
            if not inline_header:
                production = [
                    Document(page_content=chunk.page_content.split("\n", 1)[-1], metadata=dict(chunk.metadata))
                    for chunk in production
                ]
            chunks_by_file[file_name] = production
            continue

        metadata = hw4._get_document_metadata(file_name, html)
        soup = BeautifulSoup(html, "lxml")
        if use_table_text:
            for table in list(soup.find_all("table")):
                replacement = "\n" + hw4.table_to_text(str(table)) + "\n"
                table.replace_with(NavigableString(replacement))
        text = soup.get_text()
        splitter = RecursiveCharacterTextSplitter()
        chunks_by_file[file_name] = splitter.create_documents([text], metadatas=[metadata])
        hw4._CORPUS_COMPANIES[hw4._entity_key(metadata["company"])] = metadata["company"]
    all_chunks = [chunk for file_name in sorted(chunks_by_file) for chunk in chunks_by_file[file_name]]
    return all_chunks, chunks_by_file


def _tokenize(text: str, mode: str) -> list[str]:
    if mode == "domain":
        return hw4._tokenize(text)
    return str(text or "").lower().split()


def _build_index(chunks: list[Document], token_mode: str, bm25_b: float = 0.75) -> dict[str, Any]:
    tokenized = [_tokenize(chunk.page_content, token_mode) for chunk in chunks]
    bm25 = BM25Okapi(tokenized, k1=1.5, b=bm25_b) if tokenized and any(tokenized) else None
    document_frequency: dict[str, int] = {}
    company_lookup: dict[str, str] = {}
    for chunk, tokens in zip(chunks, tokenized):
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
        name = str((chunk.metadata or {}).get("company", "")).strip()
        key = hw4._entity_key(name)
        if key:
            company_lookup[key] = name
    return {"chunks": chunks, "tokenized": tokenized, "bm25": bm25,
            "document_frequency": document_frequency, "companies": company_lookup}


def _candidate_groups(index: dict[str, Any], parsed: dict[str, Any]) -> dict[str, list[int]]:
    companies = parsed.get("companies") or []
    keys = list(dict.fromkeys(hw4._entity_key(name) for name in companies))
    groups = {key: [] for key in keys}
    if not keys or any(key not in index["companies"] for key in keys):
        return {}
    for idx, chunk in enumerate(index["chunks"]):
        metadata = chunk.metadata or {}
        key = hw4._entity_key(metadata.get("company", ""))
        if (key in groups and metadata.get("doc_type") == parsed.get("doc_type") and
                hw4._year_matches(metadata, parsed.get("fiscal_year"))):
            groups[key].append(idx)
    if any(not indices for indices in groups.values()):
        return {}
    return groups


def _retrieve(index: dict[str, Any], question: str, variant: dict[str, Any]) -> list[Document]:
    chunks: list[Document] = index["chunks"]
    if not variant["metadata"]:
        query_tokens = _tokenize(question, variant["tokenizer"])
        if index["bm25"] and query_tokens:
            scores = index["bm25"].get_scores(query_tokens)
            ranked = sorted(range(len(chunks)), key=lambda i: float(scores[i]), reverse=True)
        else:
            ranked = list(range(len(chunks)))
        return [chunks[i] for i in ranked[:hw4.MAX_RETRIEVED]]

    parsed = hw4.parse_query(question)
    groups = _candidate_groups(index, parsed)
    if not groups:
        return []

    if variant["query_expansion"]:
        query_tokens = hw4._expand_query(question)
    else:
        query_tokens = _tokenize(question, variant["tokenizer"])
    scores = index["bm25"].get_scores(query_tokens) if index["bm25"] and query_tokens else [0.0] * len(chunks)
    anchors = hw4._question_anchors(question, parsed.get("companies", []))
    ranked_by_company: dict[str, list[tuple[float, int, set[str]]]] = {}

    for key, indices in groups.items():
        ranked = []
        for idx in indices:
            words = set(index["tokenized"][idx])
            score = float(scores[idx]) if idx < len(scores) else 0.0
            if variant["evidence_boost"]:
                score += 1.25 * sum(hw4._term_is_present(anchor, words) for anchor in anchors)
                year = parsed.get("fiscal_year")
                if year is not None and str(year) in chunks[idx].page_content:
                    score += 1.5
            ranked.append((score, idx, words))
        ranked.sort(key=lambda item: item[0], reverse=True)
        ranked_by_company[key] = ranked

        if variant["abstention"] and anchors:
            top_words = set().union(*(entry[2] for entry in ranked[:3])) if ranked else set()
            covered = sum(hw4._term_is_present(anchor, top_words) for anchor in anchors)
            if covered / len(anchors) < 0.60:
                return []

    if variant["abstention"]:
        df = index["document_frequency"]
        for anchor in anchors:
            if not any(df.get(token, 0) for token in hw4._aliases_for(anchor)):
                return []

    if variant["per_company"]:
        selected, selected_ids = [], set()
        max_depth = max(1, hw4.MAX_RETRIEVED // max(1, len(ranked_by_company)))
        for depth in range(max_depth):
            for key in ranked_by_company:
                ranked = ranked_by_company[key]
                if depth < len(ranked) and ranked[depth][1] not in selected_ids:
                    idx = ranked[depth][1]
                    selected.append(chunks[idx])
                    selected_ids.add(idx)
                    if len(selected) >= hw4.MAX_RETRIEVED:
                        return selected
        remaining = sorted(
            (entry for ranked in ranked_by_company.values() for entry in ranked if entry[1] not in selected_ids),
            key=lambda item: item[0], reverse=True,
        )
        for _, idx, _ in remaining:
            selected.append(chunks[idx])
            if len(selected) >= hw4.MAX_RETRIEVED:
                break
        return selected

    combined = sorted(
        (entry for ranked in ranked_by_company.values() for entry in ranked),
        key=lambda item: item[0], reverse=True,
    )
    return [chunks[idx] for _, idx, _ in combined[:hw4.MAX_RETRIEVED]]


def _table_cell_score(tables: list[dict], active: bool) -> float:
    if not active or not tables:
        return 0.0
    return sum(hw4.table_score(hw4.table_to_text(table["html"]), table) for table in tables) / len(tables)


def _chunk_pass_rate(chunks_by_file: dict[str, list[Document]], docs_meta: dict) -> float:
    if not docs_meta:
        return 0.0
    passed = 0
    for file_name, metadata in docs_meta.items():
        problems = hw4.document_problems(chunks_by_file.get(file_name), None, metadata, file_name)
        passed += not problems
    return passed / len(docs_meta)


def _safe_generate(question: str, chunks: list[Document], cache: dict[str, str], cache_path: Path) -> tuple[str, bool]:
    prompt_context = hw4.format_context(chunks)
    digest = hashlib.sha256((hw4.LLM_MODEL + "\n" + question + "\n" + prompt_context).encode("utf-8")).hexdigest()
    if digest in cache:
        return cache[digest], True
    answer = hw4.generate(question, chunks)
    cache[digest] = answer
    temp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(cache_path)
    return answer, False


def _usage_delta(before: dict, after: dict) -> dict[str, float | int]:
    input_tokens = after["llm_input_tokens"] - before["llm_input_tokens"]
    output_tokens = after["llm_output_tokens"] - before["llm_output_tokens"]
    calls = after["llm_calls"] - before["llm_calls"]
    cost = (input_tokens * hw4.PRICE_LLM_INPUT + output_tokens * hw4.PRICE_LLM_OUTPUT) / 1_000_000
    return {"generator_calls": calls, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "estimated_cost_usd": round(cost, 6)}


def _evaluate_variant(variant: dict, corpus: dict[str, str], questions: dict[str, str],
                      ground_truth: dict, docs_meta: dict, tables: list[dict],
                      cache: dict[str, str], cache_path: Path) -> tuple[dict, list[dict]]:
    hw4._CORPUS_COMPANIES.clear()
    if variant["production_chunks"]:
        # Populate the parser's corpus lexicon from document-derived metadata before parsing.
        for file_name, html in corpus.items():
            metadata = hw4._get_document_metadata(file_name, html)
            hw4._CORPUS_COMPANIES[hw4._entity_key(metadata["company"])] = metadata["company"]
        chunks, chunks_by_file = _chunks_from_html(corpus, True, True, variant["inline_header"])
    else:
        chunks, chunks_by_file = _chunks_from_html(corpus, variant["tables"], False, False)

    index = _build_index(chunks, variant["tokenizer"], variant["bm25_b"])
    details = []
    cache_hits = 0
    before_usage = hw4.GENERATOR_USAGE.summary()
    start_time = time.perf_counter()

    for number, (qid, question) in enumerate(questions.items(), 1):
        if variant["id"] == "V10":
            # Evaluate the final row with the exact implementation submitted in hw4.py.
            retrieved = hw4.retrieve(index, question)
        else:
            retrieved = _retrieve(index, question, variant)

        parsed = hw4.parse_query(question) if variant["metadata"] else None
        answer, was_cached = _safe_generate(question, retrieved, cache, cache_path)
        cache_hits += was_cached
        gold = ground_truth.get(qid)
        row = {
            "variant": variant["id"], "id": qid, "category": gold.get("category", "") if gold else "",
            "question": question, "parsed": json.dumps(parsed, ensure_ascii=False) if parsed is not None else "",
            "parse_score": round(hw4.parse_score(parsed, gold), 4) if parsed is not None and gold else "",
            "retrieved": ";".join(sorted({str(c.metadata.get("source", "")) for c in retrieved})),
            "retrieval_hit": hw4.retrieval_hit(retrieved, gold) if gold and gold.get("evidence") else "",
            "answer": answer, "correct": hw4.answer_correct(answer, gold) if gold else "",
            "expected": gold.get("answer", "") if gold and not hw4.answer_correct(answer, gold) else "",
            "chunk_count": len(retrieved), "cache_hit": was_cached,
        }
        details.append(row)
        if number % 10 == 0 or number == len(questions):
            print(f"  {variant['id']}: {number}/{len(questions)} questions", flush=True)

    elapsed = time.perf_counter() - start_time
    after_usage = hw4.GENERATOR_USAGE.summary()
    usage = _usage_delta(before_usage, after_usage)
    answerable = [qid for qid, gold in ground_truth.items() if gold.get("evidence")]
    unanswerable = [qid for qid, gold in ground_truth.items() if not gold.get("evidence")]
    by_id = {row["id"]: row for row in details}
    valid_nonempty = sum(bool(by_id[qid]["chunk_count"]) and by_id[qid]["chunk_count"] <= hw4.MAX_RETRIEVED
                         for qid in answerable) / max(1, len(answerable))
    recall = sum(bool(by_id[qid]["retrieval_hit"]) for qid in answerable) / max(1, len(answerable))
    kept = sum(bool(by_id[qid]["chunk_count"]) for qid in answerable) / max(1, len(answerable))
    rejected = sum(not by_id[qid]["chunk_count"] for qid in unanswerable) / max(1, len(unanswerable))
    accuracy = sum(bool(row["correct"]) for row in details) / max(1, len(details))
    parse_scores = [float(row["parse_score"]) for row in details if row["parse_score"] != ""]
    summary = {
        "variant": variant["id"], "name": variant["name"], "change": variant["change"],
        "table_cell_score": round(_table_cell_score(tables, variant["tables"]), 4),
        "chunk_pass_rate": round(_chunk_pass_rate(chunks_by_file, docs_meta), 4),
        "query_parse_score": round(sum(parse_scores) / len(parse_scores), 4) if parse_scores else "",
        "valid_nonempty_rate": round(valid_nonempty, 4), "recall_at_5": round(recall, 4),
        "answerable_kept_rate": round(kept, 4), "unanswerable_rejected_rate": round(rejected, 4),
        "answer_accuracy": round(accuracy, 4), "wrong_ids": ";".join(row["id"] for row in details if row["correct"] is False),
        "cache_hits": cache_hits, "elapsed_seconds": round(elapsed, 1), **usage,
    }
    print(
        f"{variant['id']} done: recall@5={recall:.1%}, kept={kept:.1%}, rejected={rejected:.1%}, "
        f"accuracy={accuracy:.1%}, API calls={usage['generator_calls']}, cache hits={cache_hits}",
        flush=True,
    )
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HW4 public-set ablation sequence.")
    parser.add_argument("--data-folder", default="public_test")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    ground_truth = hw4.load_json(folder / "ground_truth.json")
    docs_meta = hw4.load_json(folder / "docs.json")
    tables = hw4.load_json(folder / "tables.json") or []
    cache_path = output_dir / "ablation_answer_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    print(f"Ablation: {len(corpus)} documents, {len(questions)} questions; model={hw4.LLM_MODEL}", flush=True)
    summaries, all_details = [], []
    for variant in VARIANTS:
        print(f"\n{variant['id']} — {variant['name']}: {variant['change']}", flush=True)
        summary, details = _evaluate_variant(
            variant, corpus, questions, ground_truth, docs_meta, tables, cache, cache_path
        )
        summaries.append(summary)
        all_details.extend(details)
        with (output_dir / "ablation_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        with (output_dir / "ablation_details.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_details[0].keys()))
            writer.writeheader()
            writer.writerows(all_details)

    print("\nSummary:")
    for row in summaries:
        print(
            f"{row['variant']:>2}  recall={row['recall_at_5']:.1%}  "
            f"kept={row['answerable_kept_rate']:.1%}  rejected={row['unanswerable_rejected_rate']:.1%}  "
            f"accuracy={row['answer_accuracy']:.1%}  calls={row['generator_calls']}"
        )
    total_calls = sum(int(row["generator_calls"]) for row in summaries)
    total_cost = sum(float(row["estimated_cost_usd"]) for row in summaries)
    print(f"\nNew generator calls: {total_calls}; estimated cost: ${total_cost:.4f}")
    print(f"Wrote {output_dir / 'ablation_summary.csv'} and {output_dir / 'ablation_details.csv'}")


if __name__ == "__main__":
    main()
