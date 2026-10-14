#!/usr/bin/env python3
"""
PROJECT: Nidana
FILE: rag_runtime.py

Offline hybrid retrieval (ChromaDB Vector + SQLite FTS5 Keyword) and
grounded response / question-generation runtime using local Ollama.

Key Guarantees:
- Never rebuilds databases or extracts textbook concepts at runtime.
- Reuses `embeddings.py` singleton and `query: ` prefix formatting.
- Combines semantic and keyword matches via Reciprocal Rank Fusion (RRF)
  with normalized score blending and strict chunk deduplication.
- Enforces metadata filters (`class_level`, `subject`, `chapter`, `language`)
  and a configurable context character budget.
- Clearly separates verbatim retrieved textbook passages from LLM-generated
  explanations or practice questions.
- Reports Ollama unavailability transparently without falsely claiming success.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import unittest
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

import chromadb
from dotenv import load_dotenv

from embeddings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    EVALUATION_DIR,
    SQLITE_DB_PATH,
    embed_query,
    format_query,
    get_embedding_function,
    preprocess_text,
    strip_e5_prefix,
)
import embeddings as _emb_mod

load_dotenv()

# ============================================================================
# ChromaDB v0.5.x Compatibility Patch
# ============================================================================
if hasattr(_emb_mod, "ChromaE5EmbeddingFunction") and not hasattr(_emb_mod.ChromaE5EmbeddingFunction, "name"):
    _emb_mod.ChromaE5EmbeddingFunction.name = staticmethod(lambda: "default")
    _emb_mod.ChromaE5EmbeddingFunction.is_legacy = lambda self: True
    _emb_mod.ChromaE5EmbeddingFunction.get_config = lambda self: {}
    _emb_mod.ChromaE5EmbeddingFunction.build_from_config = staticmethod(lambda cfg: _emb_mod.get_embedding_function())


# ============================================================================
# Runtime Configuration & Constants
# ============================================================================

DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
DEFAULT_OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "45"))

DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 3200
DEFAULT_RRF_K = 60
DEFAULT_MIN_SEMANTIC_SIMILARITY = 0.76

ALLOWED_CLASSES = {9, 10, 12}
ALLOWED_LANGUAGES = {"or", "en", "bilingual"}

# Common conversational stopwords stripped when constructing SQLite FTS5 queries
STOPWORDS_EN_OR = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "describe",
    "define", "do", "does", "explain", "for", "from", "give", "how", "i",
    "in", "is", "it", "list", "me", "name", "of", "on", "or", "short",
    "state", "tell", "that", "the", "their", "them", "these", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "write", "note", "about", "between",
    "କଣ", "କ'ଣ", "କାହିଁକି", "କିପରି", "କେଉଁ", "କେତେ", "ଏବଂ", "ଓ", "କିମ୍ବା",
    "এক", "ଏକ", "ଏହା", "ତାହା", "ବିଷୟରେ", "ଲେଖ", "বুঝাଅ", "ବୁଝାଅ", "କୁହାଯାଏ",
}

# Token extractor preserving English words, numbers, and Odia script + matras
FTS_TOKEN_REGEX = re.compile(r"[A-Za-z0-9\u0B00-\u0B7F]{2,}")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RetrievedChunk:
    """Canonical representation of a deduplicated chunk retrieved at runtime."""
    chunk_id: str
    parent_id: str
    text: str
    metadata: Dict[str, Any]
    citation: str
    semantic_rank: Optional[int] = None
    keyword_rank: Optional[int] = None
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    rrf_score: float = 0.0
    combined_score: float = 0.0
    sources_matched: List[str] = field(default_factory=list)


# ============================================================================
# Filter Validation & Query Sanitization Helpers
# ============================================================================

def validate_runtime_filters(
    class_level: int,
    subject: Optional[str] = None,
    chapter: Optional[Any] = None,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """Validates and normalizes runtime filter parameters."""
    if isinstance(class_level, bool) or not isinstance(class_level, int) or class_level not in ALLOWED_CLASSES:
        raise ValueError(f"class_level must be one of {sorted(ALLOWED_CLASSES)}, got {class_level!r}")

    norm_subject = subject.strip() if isinstance(subject, str) and subject.strip() else None
    norm_chapter = str(chapter).strip() if chapter is not None and str(chapter).strip() else None
    norm_language: Optional[str] = None

    if isinstance(language, str) and language.strip():
        norm_language = language.strip().lower()
        if norm_language not in ALLOWED_LANGUAGES:
            raise ValueError(
                f"language filter must be one of {sorted(ALLOWED_LANGUAGES)}, got {language!r}"
            )

    return {
        "class_level": class_level,
        "subject": norm_subject,
        "chapter": norm_chapter,
        "language": norm_language,
    }


def extract_query_tokens(query: str) -> List[str]:
    """
    Extracts clean alphanumeric and Odia tokens from a student's question,
    filtering out conversational question words while keeping domain keywords.
    """
    cleaned = preprocess_text(strip_e5_prefix(query))
    raw_tokens = FTS_TOKEN_REGEX.findall(cleaned)
    if not raw_tokens:
        return []

    filtered = [t for t in raw_tokens if t.lower() not in STOPWORDS_EN_OR]
    selected = filtered if filtered else raw_tokens

    # Deduplicate while preserving order
    unique_tokens: List[str] = []
    seen: Set[str] = set()
    for tok in selected:
        low = tok.lower()
        if low not in seen:
            seen.add(low)
            unique_tokens.append(tok)
    return unique_tokens


def build_fts5_match_expression(query: str) -> str:
    """
    Builds a syntax-safe SQLite FTS5 MATCH query string using quoted tokens
    joined by OR so BM25 naturally ranks chunks matching the most keywords first.
    Never throws `sqlite3.OperationalError: fts5: syntax error` on Odia/math punctuation.
    """
    tokens = extract_query_tokens(query)
    if not tokens:
        return ""
    safe_quoted = [f'"{tok.replace(chr(34), "")}"' for tok in tokens if tok.replace('"', '')]
    return " OR ".join(safe_quoted)


def format_citation(meta: Dict[str, Any], index: Optional[int] = None) -> str:
    """Formats a textbook source citation from preserved chunk metadata."""
    prefix = f"[{index}] " if index is not None else ""
    source_name = meta.get("source_name") or "Odisha State Board Textbook"
    cls_lvl = meta.get("class_level", "?")
    subject = meta.get("subject", "General")
    ch_id = meta.get("chapter_id", "")
    ch_name = meta.get("chapter_name", "")
    sec_heading = meta.get("section_heading", "")
    pages = meta.get("page_numbers", "")
    chunk_id = meta.get("chunk_id", "")

    parts = [f"{source_name} (Class {cls_lvl} {subject})"]
    if ch_id or ch_name:
        ch_label = f"{ch_id}: {ch_name}".strip(": ")
        parts.append(f"Chapter {ch_label}")
    if sec_heading:
        parts.append(f"Section: {sec_heading}")
    if pages:
        parts.append(f"Pages: {pages}")
    if chunk_id:
        parts.append(f"ID: {chunk_id}")

    return prefix + " | ".join(parts)


def metadata_matches_filters(
    meta: Dict[str, Any],
    class_level: int,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    language: Optional[str] = None,
    allow_remediation: bool = False,
) -> bool:
    """
    Verifies that a candidate chunk's metadata strictly satisfies all requested filters.
    """
    try:
        rec_class = int(meta.get("class_level", -1))
    except (ValueError, TypeError):
        return False

    if allow_remediation:
        if rec_class > class_level or rec_class < 1:
            return False
    else:
        if rec_class != class_level:
            return False

    if subject:
        rec_subject = str(meta.get("subject", "")).strip().lower()
        if rec_subject != subject.strip().lower():
            return False

    if language:
        rec_lang = str(meta.get("language", "")).strip().lower()
        if rec_lang != language.strip().lower():
            return False

    if chapter:
        ch_target = chapter.strip().lower()
        rec_ch_id = str(meta.get("chapter_id", "")).strip().lower()
        rec_ch_name = str(meta.get("chapter_name", "")).strip().lower()
        rec_ch_num = str(meta.get("chapter_num", "")).strip().lower()

        # Match exact chapter_id ("ch03"), chapter_num ("3"), or substring of chapter_name ("water resources")
        matched_chapter = (
            ch_target == rec_ch_id
            or ch_target == rec_ch_num
            or f"ch{ch_target.zfill(2)}" == rec_ch_id
            or ch_target in rec_ch_name
        )
        if not matched_chapter:
            return False

    return True


# ============================================================================
# Stage 1: ChromaDB Semantic Search & SQLite FTS5 Keyword Search
# ============================================================================

def build_chroma_where_clause(
    class_level: int,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    language: Optional[str] = None,
    allow_remediation: bool = False,
) -> Dict[str, Any]:
    """
    Constructs a ChromaDB pre-filter `where` clause so vector search is
    restricted to matching metadata during HNSW index traversal.
    """
    conditions: List[Dict[str, Any]] = []

    if allow_remediation:
        conditions.append({"class_level": {"$lte": int(class_level)}})
    else:
        conditions.append({"class_level": {"$eq": int(class_level)}})

    if language:
        conditions.append({"language": {"$eq": language.strip().lower()}})

    if subject:
        s_clean = subject.strip()
        case_variants = sorted({s_clean, s_clean.lower(), s_clean.title(), s_clean.upper()})
        conditions.append({"subject": {"$in": case_variants}})

    if chapter:
        c_clean = chapter.strip()
        # If chapter looks like a chapter_id (e.g. 'ch03') or integer ('3'), push down to Chroma where
        if re.match(r"^ch\d+$", c_clean, re.IGNORECASE):
            ch_variants = sorted({c_clean.lower(), c_clean.upper()})
            conditions.append({"chapter_id": {"$in": ch_variants}})
        elif c_clean.isdigit():
            ch_int = int(c_clean)
            conditions.append({
                "$or": [
                    {"chapter_num": {"$eq": ch_int}},
                    {"chapter_id": {"$in": [f"ch{ch_int:02d}", f"ch{ch_int}"]}},
                ]
            })

    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def search_chromadb_semantic(
    query: str,
    class_level: int,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    language: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    allow_remediation: bool = False,
    chroma_path: Path = CHROMA_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> List[Tuple[str, str, Dict[str, Any], float]]:
    """
    Embeds `query` using the E5 `"query: "` prefix via `embeddings.py` and searches
    ChromaDB for semantic matches.

    Returns:
        List of `(chunk_id, clean_text, metadata, cosine_similarity)` ordered by similarity.
    """
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        return []

    query_embedding = embed_query(query)
    client = chromadb.PersistentClient(path=str(chroma_path))

    existing_names = [
        c.name if hasattr(c, "name") else str(c) for c in client.list_collections()
    ]
    if collection_name not in existing_names:
        return []

    collection = client.get_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
    )
    total_count = collection.count()
    if total_count == 0:
        return []

    where_clause = build_chroma_where_clause(
        class_level=class_level,
        subject=subject,
        chapter=chapter,
        language=language,
        allow_remediation=allow_remediation,
    )

    fetch_k = min(max(top_k * 3, 10), total_count)
    try:
        res = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    ids = (res.get("ids") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    results: List[Tuple[str, str, Dict[str, Any], float]] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        meta_dict = dict(meta or {})
        meta_dict.setdefault("chunk_id", str(cid))
        if not metadata_matches_filters(
            meta_dict,
            class_level=class_level,
            subject=subject,
            chapter=chapter,
            language=language,
            allow_remediation=allow_remediation,
        ):
            continue

        clean_doc = strip_e5_prefix(str(doc))
        # Cosine distance in ChromaDB is (1 - cosine_similarity)
        similarity = max(0.0, min(1.0, 1.0 - float(dist)))
        results.append((str(cid), clean_doc, meta_dict, similarity))
        if len(results) >= top_k * 2:
            break

    return results


def search_sqlite_fts5_keyword(
    query: str,
    class_level: int,
    subject: Optional[str] = None,
    chapter: Optional[str] = None,
    language: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    allow_remediation: bool = False,
    sqlite_path: Path = SQLITE_DB_PATH,
) -> List[Tuple[str, str, Dict[str, Any], float]]:
    """
    Searches SQLite FTS5 `offline_chunks_fts` using sanitized keyword tokens and
    pre-filtered SQL WHERE clauses.

    Returns:
        List of `(chunk_id, clean_text, metadata, normalized_keyword_score)` ordered by BM25 rank.
    """
    if not sqlite_path.exists():
        return []

    fts_expr = build_fts5_match_expression(query)
    if not fts_expr:
        return []

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='offline_chunks_fts';"
        )
        if not cur.fetchone():
            return []

        where_clauses = ["offline_chunks_fts MATCH ?"]
        params: List[Any] = [fts_expr]

        if allow_remediation:
            where_clauses.append("CAST(class_level AS INTEGER) <= ?")
            params.append(int(class_level))
        else:
            where_clauses.append("CAST(class_level AS INTEGER) = ?")
            params.append(int(class_level))

        if subject:
            where_clauses.append("LOWER(subject) = LOWER(?)")
            params.append(subject.strip())

        if language:
            where_clauses.append("LOWER(language) = LOWER(?)")
            params.append(language.strip().lower())

        if chapter:
            ch_clean = chapter.strip()
            where_clauses.append(
                "(LOWER(chapter_id) = LOWER(?) OR CAST(chapter_num AS TEXT) = ? OR LOWER(chapter_name) LIKE LOWER(?))"
            )
            params.extend([ch_clean, ch_clean, f"%{ch_clean}%"])

        params.append(int(top_k * 2))

        sql = f"""
            SELECT
                chunk_id,
                parent_id,
                class_level,
                subject,
                chapter_id,
                chapter_num,
                chapter_name,
                language,
                source_name,
                page_numbers,
                section_id,
                section_heading,
                source_file,
                concept_tag,
                prerequisite_concept,
                english_tech_terms,
                content_odia,
                content_hash,
                bm25(offline_chunks_fts) AS raw_bm25
            FROM offline_chunks_fts
            WHERE {" AND ".join(where_clauses)}
            ORDER BY raw_bm25 ASC
            LIMIT ?;
        """
        rows = cur.fetchall() if not cur.execute(sql, tuple(params)) else cur.fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()

    results: List[Tuple[str, str, Dict[str, Any], float]] = []
    for row in rows:
        cid = str(row["chunk_id"])
        text = str(row["content_odia"])
        # SQLite FTS5 bm25() returns negative values where more negative = better match
        raw_bm25 = abs(float(row["raw_bm25"] or 0.0))
        norm_kw_score = raw_bm25 / (1.0 + raw_bm25) if raw_bm25 > 0 else 0.5

        meta_dict: Dict[str, Any] = {
            "chunk_id": cid,
            "parent_id": str(row["parent_id"] or ""),
            "class_level": int(row["class_level"]),
            "subject": str(row["subject"] or ""),
            "chapter_id": str(row["chapter_id"] or ""),
            "chapter_num": int(row["chapter_num"] or 1),
            "chapter_name": str(row["chapter_name"] or ""),
            "language": str(row["language"] or ""),
            "source_name": str(row["source_name"] or ""),
            "page_numbers": str(row["page_numbers"] or ""),
            "section_id": str(row["section_id"] or ""),
            "section_heading": str(row["section_heading"] or ""),
            "source_file": str(row["source_file"] or ""),
            "concept_tag": str(row["concept_tag"] or ""),
            "prerequisite_concept": str(row["prerequisite_concept"] or "none"),
            "english_tech_terms": str(row["english_tech_terms"] or ""),
            "content_hash": str(row["content_hash"] or ""),
        }

        if metadata_matches_filters(
            meta_dict,
            class_level=class_level,
            subject=subject,
            chapter=chapter,
            language=language,
            allow_remediation=allow_remediation,
        ):
            results.append((cid, text, meta_dict, norm_kw_score))

    return results


# ============================================================================
# Stage 2: Documented Hybrid Ranking Strategy & Deduplication
# ============================================================================

def combine_and_rank_results(
    query: str,
    semantic_hits: Sequence[Tuple[str, str, Dict[str, Any], float]],
    keyword_hits: Sequence[Tuple[str, str, Dict[str, Any], float]],
    top_k: int = DEFAULT_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    min_semantic_similarity: float = DEFAULT_MIN_SEMANTIC_SIMILARITY,
) -> List[RetrievedChunk]:
    """
    Documented Hybrid Ranking Strategy:
    1. Deduplication by Canonical Identity & Content Hash:
       - Merges hits sharing the same `chunk_id` or identical normalized text
         so duplicate chunks are never counted as separate evidence.
    2. Relevance Gate (Anti-Hallucination Filter):
       - A candidate chunk is retained if it matched via SQLite FTS5 keyword search,
         contains at least one domain query token, OR has a ChromaDB cosine
         similarity >= `min_semantic_similarity`.
    3. Reciprocal Rank Fusion (RRF) + Score Blending:
       - Computes RRF(d) = 1/(k + rank_semantic) + 1/(k + rank_keyword) with k=60.
       - Computes `combined_score = 0.55 * semantic_score + 0.30 * keyword_score + 0.15 * normalized_rrf`
         (with a 1.15x dual-evidence boost when a chunk is retrieved by BOTH ChromaDB and SQLite FTS5).
    """
    query_tokens = [t.lower() for t in extract_query_tokens(query)]
    merged_by_id: Dict[str, RetrievedChunk] = {}
    seen_text_signatures: Dict[str, str] = {}  # text_hash -> canonical chunk_id

    def _resolve_canonical_id(cid: str, text: str) -> str:
        norm_sig = hashlib.sha256(preprocess_text(text).lower().encode("utf-8")).hexdigest()
        if cid in merged_by_id:
            return cid
        if norm_sig in seen_text_signatures:
            return seen_text_signatures[norm_sig]
        seen_text_signatures[norm_sig] = cid
        return cid

    # 1. Incorporate semantic hits from ChromaDB
    for rank_idx, (cid, text, meta, sem_score) in enumerate(semantic_hits, start=1):
        canon_id = _resolve_canonical_id(cid, text)
        if canon_id not in merged_by_id:
            merged_by_id[canon_id] = RetrievedChunk(
                chunk_id=canon_id,
                parent_id=str(meta.get("parent_id", "")),
                text=text,
                metadata=meta,
                citation=format_citation(meta),
                semantic_rank=rank_idx,
                semantic_score=round(float(sem_score), 4),
                sources_matched=["chromadb"],
            )
        else:
            existing = merged_by_id[canon_id]
            if existing.semantic_rank is None or rank_idx < existing.semantic_rank:
                existing.semantic_rank = rank_idx
                existing.semantic_score = round(float(sem_score), 4)
            if "chromadb" not in existing.sources_matched:
                existing.sources_matched.append("chromadb")

    # 2. Incorporate keyword hits from SQLite FTS5
    for rank_idx, (cid, text, meta, kw_score) in enumerate(keyword_hits, start=1):
        canon_id = _resolve_canonical_id(cid, text)
        if canon_id not in merged_by_id:
            merged_by_id[canon_id] = RetrievedChunk(
                chunk_id=canon_id,
                parent_id=str(meta.get("parent_id", "")),
                text=text,
                metadata=meta,
                citation=format_citation(meta),
                keyword_rank=rank_idx,
                keyword_score=round(float(kw_score), 4),
                sources_matched=["sqlite_fts5"],
            )
        else:
            existing = merged_by_id[canon_id]
            if existing.keyword_rank is None or rank_idx < existing.keyword_rank:
                existing.keyword_rank = rank_idx
                existing.keyword_score = round(float(kw_score), 4)
            if "sqlite_fts5" not in existing.sources_matched:
                existing.sources_matched.append("sqlite_fts5")

    # 3. Score, filter out low-relevance noise, and sort
    ranked_chunks: List[RetrievedChunk] = []
    for chunk in merged_by_id.values():
        haystack = (
            f"{chunk.text} {chunk.metadata.get('chapter_name', '')} "
            f"{chunk.metadata.get('section_heading', '')} {chunk.metadata.get('english_tech_terms', '')}"
        ).lower()
        lexical_hit = any(tok in haystack for tok in query_tokens) if query_tokens else False

        # Relevance gate: reject purely semantic hits that fall below similarity threshold AND have 0 lexical overlap
        if (
            chunk.keyword_rank is None
            and not lexical_hit
            and chunk.semantic_score < min_semantic_similarity
        ):
            continue

        rrf_val = 0.0
        if chunk.semantic_rank is not None:
            rrf_val += 1.0 / (rrf_k + chunk.semantic_rank)
        if chunk.keyword_rank is not None:
            rrf_val += 1.0 / (rrf_k + chunk.keyword_rank)

        chunk.rrf_score = round(rrf_val, 6)
        norm_rrf = rrf_val * (rrf_k + 1) / 2.0

        base_combined = (
            0.55 * chunk.semantic_score
            + 0.30 * chunk.keyword_score
            + 0.15 * norm_rrf
        )
        if len(chunk.sources_matched) >= 2:
            base_combined *= 1.15

        chunk.combined_score = round(base_combined, 4)
        ranked_chunks.append(chunk)

    ranked_chunks.sort(
        key=lambda c: (c.combined_score, c.rrf_score, c.semantic_score, c.keyword_score),
        reverse=True,
    )

    selected = ranked_chunks[: max(1, int(top_k))]
    for idx, ch in enumerate(selected, start=1):
        ch.citation = format_citation(ch.metadata, index=idx)

    return selected


def retrieve_hybrid_context(
    question: str,
    class_level: int,
    subject: Optional[str] = None,
    chapter: Optional[Any] = None,
    language: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    allow_remediation: bool = False,
    min_semantic_similarity: float = DEFAULT_MIN_SEMANTIC_SIMILARITY,
    chroma_path: Path = CHROMA_DB_PATH,
    sqlite_path: Path = SQLITE_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Tuple[List[RetrievedChunk], str, List[str]]:
    """
    Executes hybrid retrieval across ChromaDB and SQLite FTS5, deduplicates and
    ranks results, and builds a context block within `max_context_chars`.

    Returns:
        `(selected_chunks, formatted_context_string, citations_list)`
    """
    if not isinstance(question, str) or not preprocess_text(strip_e5_prefix(question)):
        raise ValueError("Question or topic must be a non-empty string.")

    filters = validate_runtime_filters(
        class_level=class_level,
        subject=subject,
        chapter=chapter,
        language=language,
    )

    sem_hits = search_chromadb_semantic(
        query=question,
        class_level=filters["class_level"],
        subject=filters["subject"],
        chapter=filters["chapter"],
        language=filters["language"],
        top_k=top_k,
        allow_remediation=allow_remediation,
        chroma_path=chroma_path,
        collection_name=collection_name,
    )

    kw_hits = search_sqlite_fts5_keyword(
        query=question,
        class_level=filters["class_level"],
        subject=filters["subject"],
        chapter=filters["chapter"],
        language=filters["language"],
        top_k=top_k,
        allow_remediation=allow_remediation,
        sqlite_path=sqlite_path,
    )

    ranked_chunks = combine_and_rank_results(
        query=question,
        semantic_hits=sem_hits,
        keyword_hits=kw_hits,
        top_k=top_k,
        min_semantic_similarity=min_semantic_similarity,
    )

    # Enforce configurable context character budget
    budget = max(200, int(max_context_chars))
    budgeted_chunks: List[RetrievedChunk] = []
    context_blocks: List[str] = []
    citations: List[str] = []
    used_chars = 0

    for chunk in ranked_chunks:
        cite_idx = len(budgeted_chunks) + 1
        chunk.citation = format_citation(chunk.metadata, index=cite_idx)
        header = f"SOURCE [{cite_idx}]: {chunk.citation}\n"
        body = chunk.text.strip()
        block = f"{header}{body}"

        if used_chars + len(block) > budget:
            remaining = budget - used_chars - len(header) - 20
            if remaining >= 120:
                truncated_body = body[:remaining].rsplit(" ", 1)[0] + " [...]"
                block = f"{header}{truncated_body}"
                context_blocks.append(block)
                budgeted_chunks.append(chunk)
                citations.append(chunk.citation)
                used_chars += len(block)
            break

        context_blocks.append(block)
        budgeted_chunks.append(chunk)
        citations.append(chunk.citation)
        used_chars += len(block) + 2

    context_str = "\n\n".join(context_blocks)
    return budgeted_chunks, context_str, citations


# ============================================================================
# Stage 3: Local Ollama Client & Grounded Prompt Execution
# ============================================================================

def call_local_ollama(
    prompt: str,
    system_prompt: str,
    ollama_url: str = DEFAULT_OLLAMA_BASE_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    timeout_sec: int = DEFAULT_OLLAMA_TIMEOUT_SEC,
) -> Tuple[bool, str, Optional[str]]:
    """
    Calls the configured local Ollama `/api/generate` endpoint.
    Never falsely reports success when Ollama is offline, unreachable, or errors out.

    Returns:
        `(succeeded: bool, generated_text: str, error_message: Optional[str])`
    """
    endpoint = f"{ollama_url.rstrip('/')}/api/generate"
    payload = json.dumps({
        "model": ollama_model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 600,
        },
    }).encode("utf-8")

    req = urllib_request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=timeout_sec) as resp:
            raw_body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw_body)
            response_text = str(data.get("response", "")).strip()
            if not response_text:
                return False, "", "Ollama returned an empty response body."
            return True, response_text, None
    except urllib_error.HTTPError as exc:
        return (
            False,
            "",
            f"Ollama HTTP error {exc.code} from {endpoint} (model='{ollama_model}'): {exc.reason}",
        )
    except urllib_error.URLError as exc:
        return (
            False,
            "",
            f"Local Ollama service is unreachable at {endpoint}: {exc.reason}",
        )
    except (TimeoutError, OSError, json.JSONDecodeError) as exc:
        return (
            False,
            "",
            f"Local Ollama request failed ({type(exc).__name__}): {exc}",
        )


def format_dual_section_output(
    mode_title: str,
    retrieved_chunks: Sequence[RetrievedChunk],
    generated_section_text: str,
    llm_status: str,
    error_message: Optional[str] = None,
) -> str:
    """
    Formats the response so retrieved textbook source passages are visually and
    structurally distinct from generated LLM explanations or questions.
    """
    lines = ["=== 1. RETRIEVED TEXTBOOK SOURCE PASSAGES (VERBATIM) ==="]
    if not retrieved_chunks:
        lines.append("No matching textbook passages were found for the specified query and filters.")
    else:
        for ch in retrieved_chunks:
            lines.append(f"\n{ch.citation}")
            lines.append(
                f"[Retrieval Scores: combined={ch.combined_score}, semantic={ch.semantic_score}, "
                f"keyword={ch.keyword_score}, matched_by={','.join(ch.sources_matched)}]"
            )
            lines.append(ch.text)

    lines.append(f"\n=== 2. {mode_title} [LLM Status: {llm_status.upper()}] ===")
    if error_message:
        lines.append(f"[LLM Runtime Notice]: {error_message}")
    lines.append(generated_section_text)
    return "\n".join(lines)


# ============================================================================
# Public Runtime Interface: Explanation Mode & Question-Generation Mode
# ============================================================================

def answer_question(
    question: str,
    class_level: int,
    subject: Optional[str] = None,
    chapter: Optional[Any] = None,
    language: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    allow_remediation: bool = False,
    min_semantic_similarity: float = DEFAULT_MIN_SEMANTIC_SIMILARITY,
    chroma_path: Path = CHROMA_DB_PATH,
    sqlite_path: Path = SQLITE_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
    ollama_url: str = DEFAULT_OLLAMA_BASE_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_caller: Optional[Callable[[str, str, str, str], Tuple[bool, str, Optional[str]]]] = None,
) -> Dict[str, Any]:
    """
    Public Explanation Mode Runtime:
    Retrieves grounded Odisha State Board textbook chunks from ChromaDB and
    SQLite FTS5 and uses the local Ollama model to answer the student's question.
    """
    filters = validate_runtime_filters(class_level, subject, chapter, language)

    chunks, context_str, citations = retrieve_hybrid_context(
        question=question,
        class_level=class_level,
        subject=subject,
        chapter=chapter,
        language=language,
        top_k=top_k,
        max_context_chars=max_context_chars,
        allow_remediation=allow_remediation,
        min_semantic_similarity=min_semantic_similarity,
        chroma_path=chroma_path,
        sqlite_path=sqlite_path,
        collection_name=collection_name,
    )

    serialized_chunks = [asdict(c) for c in chunks]

    # Requirement 16: Acknowledge limitation if retrieved context is insufficient
    if not chunks or not context_str.strip():
        limitation_msg = (
            "Insufficient textbook context: No approved textbook passages matched your "
            f"question under the requested filters (Class={class_level}, Subject={subject or 'Any'}, "
            f"Chapter={chapter or 'Any'}, Language={language or 'Any'}). "
            "I cannot answer this question without inventing facts outside the textbook."
        )
        formatted = format_dual_section_output(
            mode_title="GENERATED EXPLANATION",
            retrieved_chunks=[],
            generated_section_text=limitation_msg,
            llm_status="skipped_insufficient_context",
        )
        return {
            "mode": "explanation",
            "question": question,
            "filters_applied": filters,
            "sufficient_context": False,
            "generation_succeeded": False,
            "llm_status": "skipped_insufficient_context",
            "error": None,
            "retrieved_chunks": [],
            "context_used": "",
            "context_char_count": 0,
            "context_budget_chars": max_context_chars,
            "citations": [],
            "generated_explanation": limitation_msg,
            "answer": limitation_msg,
            "formatted_output": formatted,
        }

    lang_instruction = (
        "Respond in Odia (ଓଡ଼ିଆ) keeping technical terms in brackets, or in English if the question is in English."
        if (language or "").lower() != "en"
        else "Respond clearly in English."
    )

    system_prompt = (
        f"You are Nidana, an offline Odisha State Board tutor for Class {class_level}. "
        "You must answer ONLY using the provided TEXTBOOK CONTEXT. "
        "Never invent facts, formulas, or examples not present in the context. "
        "Cite the source number (e.g., [1], [2]) for every factual claim. "
        "If the provided textbook context does not contain enough information to fully "
        "answer the student's question, explicitly state what is missing from the context.\n"
        f"{lang_instruction}"
    )

    user_prompt = (
        f"TEXTBOOK CONTEXT:\n{context_str}\n\n"
        f"STUDENT QUESTION (Class {class_level}):\n{question}\n\n"
        "Provide a grounded explanation citing [1], [2], etc. based strictly on the textbook context above:"
    )

    caller = ollama_caller or call_local_ollama
    succeeded, llm_text, err_msg = caller(
        user_prompt, system_prompt, ollama_url, ollama_model
    )

    if not succeeded:
        unavailable_msg = (
            "Local Ollama model is currently unavailable, so a generated explanation could not be produced. "
            "Please review the verbatim retrieved textbook passages above."
        )
        formatted = format_dual_section_output(
            mode_title="GENERATED EXPLANATION",
            retrieved_chunks=chunks,
            generated_section_text=unavailable_msg,
            llm_status="unavailable",
            error_message=err_msg,
        )
        return {
            "mode": "explanation",
            "question": question,
            "filters_applied": filters,
            "sufficient_context": True,
            "generation_succeeded": False,
            "llm_status": "unavailable",
            "error": err_msg,
            "retrieved_chunks": serialized_chunks,
            "context_used": context_str,
            "context_char_count": len(context_str),
            "context_budget_chars": max_context_chars,
            "citations": citations,
            "generated_explanation": "",
            "answer": unavailable_msg,
            "formatted_output": formatted,
        }

    formatted = format_dual_section_output(
        mode_title="GENERATED EXPLANATION",
        retrieved_chunks=chunks,
        generated_section_text=llm_text,
        llm_status="ok",
    )

    return {
        "mode": "explanation",
        "question": question,
        "filters_applied": filters,
        "sufficient_context": True,
        "generation_succeeded": True,
        "llm_status": "ok",
        "error": None,
        "retrieved_chunks": serialized_chunks,
        "context_used": context_str,
        "context_char_count": len(context_str),
        "context_budget_chars": max_context_chars,
        "citations": citations,
        "generated_explanation": llm_text,
        "answer": llm_text,
        "formatted_output": formatted,
    }


def generate_practice_questions(
    topic_or_query: str,
    class_level: int,
    subject: Optional[str] = None,
    chapter: Optional[Any] = None,
    language: Optional[str] = None,
    num_mcqs: int = 3,
    num_descriptive: int = 2,
    top_k: int = DEFAULT_TOP_K,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    allow_remediation: bool = False,
    min_semantic_similarity: float = DEFAULT_MIN_SEMANTIC_SIMILARITY,
    save_to_question_bank: Optional[Path] = None,
    chroma_path: Path = CHROMA_DB_PATH,
    sqlite_path: Path = SQLITE_DB_PATH,
    collection_name: str = CHROMA_COLLECTION_NAME,
    ollama_url: str = DEFAULT_OLLAMA_BASE_URL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_caller: Optional[Callable[[str, str, str, str], Tuple[bool, str, Optional[str]]]] = None,
) -> Dict[str, Any]:
    """
    Public Question-Generation Mode Runtime:
    Generates textbook-grounded Multiple-Choice Questions (MCQs) and Descriptive
    Questions strictly from retrieved textbook chunks.

    Requirement 19 Guarantee:
    Generated questions are NEVER inserted into ChromaDB or SQLite `offline_chunks_fts`.
    If `save_to_question_bank` is explicitly provided, they are saved only to that
    separate evaluation file (defaulting under `data/evaluation/`).
    """
    filters = validate_runtime_filters(class_level, subject, chapter, language)

    chunks, context_str, citations = retrieve_hybrid_context(
        question=topic_or_query,
        class_level=class_level,
        subject=subject,
        chapter=chapter,
        language=language,
        top_k=top_k,
        max_context_chars=max_context_chars,
        allow_remediation=allow_remediation,
        min_semantic_similarity=min_semantic_similarity,
        chroma_path=chroma_path,
        sqlite_path=sqlite_path,
        collection_name=collection_name,
    )

    serialized_chunks = [asdict(c) for c in chunks]

    if not chunks or not context_str.strip():
        limitation_msg = (
            "Insufficient textbook context: Cannot generate practice questions because "
            "no textbook passages matched the requested topic and metadata filters."
        )
        formatted = format_dual_section_output(
            mode_title="GENERATED PRACTICE QUESTIONS",
            retrieved_chunks=[],
            generated_section_text=limitation_msg,
            llm_status="skipped_insufficient_context",
        )
        return {
            "mode": "question_generation",
            "topic": topic_or_query,
            "filters_applied": filters,
            "sufficient_context": False,
            "generation_succeeded": False,
            "llm_status": "skipped_insufficient_context",
            "error": None,
            "retrieved_chunks": [],
            "context_used": "",
            "context_char_count": 0,
            "context_budget_chars": max_context_chars,
            "citations": [],
            "generated_questions": "",
            "saved_question_bank_path": None,
            "formatted_output": formatted,
        }

    lang_instruction = (
        "Respond ENTIRELY in Odia (ଓଡ଼ିଆ) keeping technical terms in brackets (e.g., [ISRO])."
        if (language or "").lower() != "en"
        else "Respond clearly in English."
    )

    system_prompt = (
        f"You are a test generator for Class {class_level}. "
        "Your job is to read the provided Odia TEXTBOOK CONTEXT and write practice questions in ENGLISH. "
        "Focus only on historical dates, people, and events."
    )

    user_prompt = (
        f"TEXTBOOK CONTEXT:\n{context_str}\n\n"
        f"TOPIC: {topic_or_query}\n\n"
        "TASK:\n"
        "Write the questions in English based on the context above. "
        "Do not write explanations. Ignore the list of English 'Terms' at the bottom of the context.\n\n"
        "Follow this exact format:\n\n"
        "### MCQs\n"
        "Q1. [Question]\n"
        "A) [Option 1]\n"
        "B) [Option 2]\n"
        "C) [Option 3]\n"
        "D) [Option 4]\n"
        "Correct Answer: [Letter] [1]\n\n"
        "### Descriptive\n"
        "Q1. [Question]\n"
        "Answer: [Answer] [1]\n\n"
        f"Generate {int(num_mcqs)} MCQs and {int(num_descriptive)} Descriptive questions now.\n\n"
        "### MCQs\n"
    )

    
    caller = ollama_caller or call_local_ollama
    succeeded, llm_questions, err_msg = caller(
        user_prompt, system_prompt, ollama_url, ollama_model
    )

    if not succeeded:
        unavailable_msg = (
            "Local Ollama model is unavailable, so practice questions could not be generated."
        )
        formatted = format_dual_section_output(
            mode_title="GENERATED PRACTICE QUESTIONS",
            retrieved_chunks=chunks,
            generated_section_text=unavailable_msg,
            llm_status="unavailable",
            error_message=err_msg,
        )
        return {
            "mode": "question_generation",
            "topic": topic_or_query,
            "filters_applied": filters,
            "sufficient_context": True,
            "generation_succeeded": False,
            "llm_status": "unavailable",
            "error": err_msg,
            "retrieved_chunks": serialized_chunks,
            "context_used": context_str,
            "context_char_count": len(context_str),
            "context_budget_chars": max_context_chars,
            "citations": citations,
            "generated_questions": "",
            "saved_question_bank_path": None,
            "formatted_output": formatted,
        }

    saved_path_str: Optional[str] = None
    if save_to_question_bank is not None:
        bank_path = Path(save_to_question_bank).resolve()
        # Guard against ever saving generated questions into source_text, processed, or extracted_json
        forbidden_dirs = {"source_text", "processed", "extracted_json", "chroma_odisha_kb"}
        if any(part.lower() in forbidden_dirs for part in bank_path.parts):
            raise ValueError(
                f"Refusing to save generated questions inside knowledge base directory: {bank_path}"
            )
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        existing_bank: List[Dict[str, Any]] = []
        if bank_path.exists():
            try:
                existing_bank = json.loads(bank_path.read_text(encoding="utf-8"))
            except Exception:
                existing_bank = []

        existing_bank.append({
            "topic": topic_or_query,
            "filters": filters,
            "num_mcqs": num_mcqs,
            "num_descriptive": num_descriptive,
            "citations": citations,
            "generated_questions": llm_questions,
        })
        bank_path.write_text(
            json.dumps(existing_bank, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        saved_path_str = str(bank_path)

    formatted = format_dual_section_output(
        mode_title="GENERATED PRACTICE QUESTIONS",
        retrieved_chunks=chunks,
        generated_section_text=llm_questions,
        llm_status="ok",
    )

    return {
        "mode": "question_generation",
        "topic": topic_or_query,
        "filters_applied": filters,
        "sufficient_context": True,
        "generation_succeeded": True,
        "llm_status": "ok",
        "error": None,
        "retrieved_chunks": serialized_chunks,
        "context_used": context_str,
        "context_char_count": len(context_str),
        "context_budget_chars": max_context_chars,
        "citations": citations,
        "generated_questions": llm_questions,
        "saved_question_bank_path": saved_path_str,
        "formatted_output": formatted,
    }


# ============================================================================
# Automated Unit Tests
# ============================================================================

class TestRagRuntime(unittest.TestCase):
    """
    Tests verifying:
    - Relevant content is retrieved (hybrid ChromaDB + SQLite FTS5).
    - Metadata filters work (class_level, subject, chapter, language).
    - Empty / insufficient retrieval is handled without hallucinating facts.
    - Ollama failure is reported honestly (generation_succeeded=False).
    - Generated responses are grounded in retrieved context and clearly separated.
    - Generated questions are never inserted into the source knowledge base.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="nidana_rag_runtime_test_"))
        cls.chroma_dir = cls.temp_dir / "chroma_kb"
        cls.sqlite_file = cls.temp_dir / "offline_curriculum.db"
        cls.backup_dir = cls.temp_dir / "backups"
        cls.collection_name = "test_runtime_kb"

        # Dynamically import build_knowledge_base from 3_build_databases.py
        import importlib.util
        builder_path = Path(__file__).resolve().parent / "3_build_databases.py"
        spec = importlib.util.spec_from_file_location("build_db_mod", builder_path)
        build_mod = importlib.util.module_from_spec(spec)
        sys.modules["build_db_mod"] = build_mod
        spec.loader.exec_module(build_mod)

        cls.sample_records = [
            {
                "id": "c10_geography_ch03_s001",
                "text": (
                    "Water resources in Odisha include major river basins like the Mahanadi, "
                    "Brahmani, and Baitarani. The Hirakud Dam across the Mahanadi River is the "
                    "longest earthen dam in the world and supports irrigation and hydroelectricity."
                ),
                "metadata": {
                    "class_level": 10,
                    "subject": "Geography",
                    "chapter_id": "ch03",
                    "chapter_num": 3,
                    "chapter_name": "Water Resources",
                    "language": "en",
                    "source_name": "BSE Odisha Class 10 Geography",
                    "page_numbers": "25-30",
                    "section_id": "s001",
                    "section_heading": "Mahanadi and Hirakud Project",
                    "source_file": "data/source_text/10th Water resources.txt",
                    "concept_tag": "hirakud_dam_c10",
                    "prerequisite_concept": "none",
                    "english_tech_terms": "water resources, hirakud dam, mahanadi",
                    "diagnostic_question_odia": "",
                },
            },
            {
                "id": "c9_physics_ch01_s001",
                "text": (
                    "ନିଉଟନ୍‌ଙ୍କ ପ୍ରଥମ ଗତି ନିୟମ (Newton's First Law of Motion): ବାହ୍ୟ ଅସନ୍ତୁଳିତ ବଳ "
                    "(External Unbalanced Force) ପ୍ରୟୋଗ ନ ହେଲେ ସ୍ଥିର ବସ୍ତୁ ସ୍ଥିର ରହେ ଏବଂ ଗତିଶୀଳ ବସ୍ତୁ "
                    "ସମାନ ବେଗରେ সরଳରେଖାରେ ଗତି କରେ। ଏହାକୁ ଜଡ଼ତା ନିୟମ (Law of Inertia) କୁହାଯାଏ।"
                ),
                "metadata": {
                    "class_level": 9,
                    "subject": "Physics",
                    "chapter_id": "ch01",
                    "chapter_num": 1,
                    "chapter_name": "ଗତି (Motion)",
                    "language": "or",
                    "source_name": "BSE Odisha Class 9 Vigyan",
                    "page_numbers": "1-8",
                    "section_id": "s001",
                    "section_heading": "ଜଡ଼ତା ଏବଂ ବଳ (Inertia and Force)",
                    "source_file": "data/source_text/c9_motion.txt",
                    "concept_tag": "inertia_law_c9",
                    "prerequisite_concept": "none",
                    "english_tech_terms": "newton's first law of motion, law of inertia",
                    "diagnostic_question_odia": "",
                },
            },
            {
                "id": "c12_physics_ch03_s001",
                "text": (
                    "Kirchhoff's Current Law (KCL) states that the algebraic sum of electric "
                    "currents meeting at any junction in an electrical network is zero: ΣI = 0."
                ),
                "metadata": {
                    "class_level": 12,
                    "subject": "Physics",
                    "chapter_id": "ch03",
                    "chapter_num": 3,
                    "chapter_name": "Current Electricity",
                    "language": "en",
                    "source_name": "CHSE Odisha Class 12 Physics",
                    "page_numbers": "92-98",
                    "section_id": "s001",
                    "section_heading": "Kirchhoff's Laws",
                    "source_file": "data/source_text/c12_electricity.txt",
                    "concept_tag": "kirchhoff_laws_c12",
                    "prerequisite_concept": "none",
                    "english_tech_terms": "kirchhoff's current law, junction",
                    "diagnostic_question_odia": "",
                },
            },
        ]

        build_mod.build_knowledge_base(
            approved_records=cls.sample_records,
            chroma_path=cls.chroma_dir,
            sqlite_path=cls.sqlite_file,
            backup_dir=cls.backup_dir,
            collection_name=cls.collection_name,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_1_relevant_content_retrieved_and_deduplicated(self) -> None:
        """Verifies hybrid retrieval finds relevant content from both ChromaDB and SQLite without duplicates."""
        chunks, context_str, citations = retrieve_hybrid_context(
            question="Which river is the Hirakud Dam built across in Odisha?",
            class_level=10,
            subject="Geography",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "c10_geography_ch03_s001_c001")
        # Both ChromaDB and SQLite FTS5 should match this chunk and merge into 1 deduplicated evidence item
        self.assertIn("chromadb", chunks[0].sources_matched)
        self.assertIn("sqlite_fts5", chunks[0].sources_matched)
        self.assertIn("Hirakud Dam", context_str)
        self.assertEqual(len(citations), 1)

    def test_2_metadata_filters_enforce_class_subject_chapter_language(self) -> None:
        """Verifies class_level, subject, chapter, and language filters strictly bound retrieval."""
        # Class 9 Odia Physics query should retrieve only the Class 9 Odia chunk
        chunks_c9, _, _ = retrieve_hybrid_context(
            question="ଜଡ଼ତା ନିୟମ କ'ଣ (What is Law of Inertia)?",
            class_level=9,
            subject="Physics",
            chapter="ch01",
            language="or",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
        )
        self.assertEqual(len(chunks_c9), 1)
        self.assertEqual(chunks_c9[0].metadata["class_level"], 9)
        self.assertEqual(chunks_c9[0].metadata["language"], "or")

        # Grade-ceiling isolation: A Class 9 student asking about Class 12 Kirchhoff's Law must NOT receive Class 12 chunks
        chunks_leak_check, _, _ = retrieve_hybrid_context(
            question="Explain Kirchhoff's Current Law KCL at a junction",
            class_level=9,
            subject="Physics",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
        )
        for ch in chunks_leak_check:
            self.assertNotEqual(ch.metadata["class_level"], 12)

    def test_3_empty_retrieval_acknowledged_without_inventing_facts(self) -> None:
        """Verifies that empty retrieval sets sufficient_context=False and refuses to hallucinate."""
        res = answer_question(
            question="Explain mitosis and meiosis in plant cells.",
            class_level=10,
            subject="Biology",  # No Biology records exist in the KB
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
        )
        self.assertFalse(res["sufficient_context"])
        self.assertFalse(res["generation_succeeded"])
        self.assertEqual(res["llm_status"], "skipped_insufficient_context")
        self.assertEqual(len(res["retrieved_chunks"]), 0)
        self.assertIn("Insufficient textbook context", res["answer"])

    def test_4_ollama_failure_reported_without_false_success(self) -> None:
        """Verifies that when Ollama is offline/unreachable, generation_succeeded is False and error is reported."""
        res = answer_question(
            question="What is the importance of Hirakud Dam?",
            class_level=10,
            subject="Geography",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
            ollama_url="http://127.0.0.1:65530",  # Unreachable port
        )
        self.assertTrue(res["sufficient_context"])
        self.assertEqual(len(res["retrieved_chunks"]), 1)
        self.assertFalse(res["generation_succeeded"])
        self.assertEqual(res["llm_status"], "unavailable")
        self.assertIsNotNone(res["error"])
        # Even when Ollama fails, the verbatim textbook passages are still preserved in formatted_output
        self.assertIn("RETRIEVED TEXTBOOK SOURCE PASSAGES", res["formatted_output"])
        self.assertIn("Hirakud Dam", res["formatted_output"])

    def test_5_grounded_response_and_question_generation_separation(self) -> None:
        """Verifies grounded explanation and question generation never modify ChromaDB or SQLite KB."""
        def mock_ollama_ok(prompt: str, system: str, url: str, model: str) -> Tuple[bool, str, Optional[str]]:
            # Verify the prompt sent to Ollama contains the retrieved textbook context
            self.assertIn("Hirakud Dam", prompt)
            self.assertIn("SOURCE [1]:", prompt)
            return (
                True,
                "According to the textbook [1], Hirakud Dam is built across the Mahanadi River and supports irrigation.",
                None,
            )

        ans = answer_question(
            question="Which river is Hirakud Dam built across?",
            class_level=10,
            subject="Geography",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
            ollama_caller=mock_ollama_ok,
        )
        self.assertTrue(ans["sufficient_context"])
        self.assertTrue(ans["generation_succeeded"])
        self.assertEqual(ans["llm_status"], "ok")
        self.assertIn("[1]", ans["generated_explanation"])
        self.assertIn("1. RETRIEVED TEXTBOOK SOURCE PASSAGES", ans["formatted_output"])
        self.assertIn("2. GENERATED EXPLANATION", ans["formatted_output"])

        # Now test Question Generation Mode and verify KB record counts remain untouched
        q_bank_file = self.temp_dir / "evaluation" / "generated_bank.json"

        def mock_ollama_qgen(prompt: str, system: str, url: str, model: str) -> Tuple[bool, str, Optional[str]]:
            return (
                True,
                "MCQ 1 [1]: Hirakud Dam is built on which river?\nA) Ganga B) Mahanadi C) Godavari D) Krishna\nAnswer: B",
                None,
            )

        q_res = generate_practice_questions(
            topic_or_query="Hirakud Dam and Mahanadi River",
            class_level=10,
            subject="Geography",
            num_mcqs=1,
            num_descriptive=1,
            save_to_question_bank=q_bank_file,
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
            ollama_caller=mock_ollama_qgen,
        )
        self.assertTrue(q_res["generation_succeeded"])
        self.assertIn("MCQ 1 [1]", q_res["generated_questions"])
        self.assertTrue(q_bank_file.exists())

        # Verify source KB still has exactly 3 records in ChromaDB and 3 rows in SQLite FTS5
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        col = client.get_collection(self.collection_name)
        self.assertEqual(col.count(), 3)

        conn = sqlite3.connect(str(self.sqlite_file))
        fts_count = conn.execute("SELECT COUNT(*) FROM offline_chunks_fts;").fetchone()[0]
        conn.close()
        self.assertEqual(fts_count, 3)


# ============================================================================
# Command-Line Interface
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nidana Offline RAG Runtime: Hybrid Retrieval + Grounded Ollama Explanations & Questions."
    )
    parser.add_argument(
        "--question",
        type=str,
        default="What are the major water resources and river projects in Odisha?",
        help="Student question or topic to retrieve and answer.",
    )
    parser.add_argument(
        "--class-level",
        type=int,
        default=10,
        help="Student class level: 9, 10, or 12 (default: 10).",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optional subject filter (e.g., Geography, Physics).",
    )
    parser.add_argument(
        "--chapter",
        type=str,
        default=None,
        help="Optional chapter filter (e.g., ch03, 3, or 'Water Resources').",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Optional language filter: 'or' or 'en'.",
    )
    parser.add_argument(
        "--mode",
        choices=["explain", "questions"],
        default="explain",
        help="Runtime mode: 'explain' for concept explanation or 'questions' for MCQ/descriptive generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Maximum number of deduplicated chunks to retrieve (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=DEFAULT_MAX_CONTEXT_CHARS,
        help=f"Maximum character budget for retrieved textbook context (default: {DEFAULT_MAX_CONTEXT_CHARS}).",
    )
    parser.add_argument(
        "--save-questions",
        type=Path,
        default=None,
        help="Optional path under data/evaluation/ to save generated questions separately from the KB.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the automated unit test suite and exit.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.run_tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRagRuntime)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    try:
        if args.mode == "explain":
            result_payload = answer_question(
                question=args.question,
                class_level=args.class_level,
                subject=args.subject,
                chapter=args.chapter,
                language=args.language,
                top_k=args.top_k,
                max_context_chars=args.context_budget,
            )
        else:
            result_payload = generate_practice_questions(
                topic_or_query=args.question,
                class_level=args.class_level,
                subject=args.subject,
                chapter=args.chapter,
                language=args.language,
                top_k=args.top_k,
                max_context_chars=args.context_budget,
                save_to_question_bank=args.save_questions,
            )

        print(result_payload["formatted_output"])
        return 0
    except Exception as exc:
        print(f"\n[RUNTIME ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())