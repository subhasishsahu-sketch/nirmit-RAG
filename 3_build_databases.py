#!/usr/bin/env python3
"""
PROJECT: Nidana
FILE: 3_build_databases.py

Stage 3: One-time and incremental knowledge-base builder.
Consumes only approved textbook records validated by `2_validate_links.py`,
splits long sections into context-preserving chunks, generates embeddings
strictly through `embeddings.py` (with "passage: " prefixes), and populates
both ChromaDB (`db/chroma_odisha_kb/`) and SQLite FTS5 (`db/offline_curriculum.db`)
from a single canonical chunk representation.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import chromadb

from embeddings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    DB_DIR,
    EXTRACTED_JSON_DIR,
    PROJECT_ROOT,
    SQLITE_DB_PATH,
    embed_passages,
    ensure_directories,
    format_passage,
    get_embedding_function,
)
import embeddings as _emb_mod

# ============================================================================
# ChromaDB v0.5.x Compatibility Patch
# ============================================================================
if hasattr(_emb_mod, "ChromaE5EmbeddingFunction") and not hasattr(_emb_mod.ChromaE5EmbeddingFunction, "name"):
    _emb_mod.ChromaE5EmbeddingFunction.name = staticmethod(lambda: "default")
    _emb_mod.ChromaE5EmbeddingFunction.is_legacy = lambda self: True
    _emb_mod.ChromaE5EmbeddingFunction.get_config = lambda self: {}
    _emb_mod.ChromaE5EmbeddingFunction.build_from_config = staticmethod(lambda cfg: _emb_mod.get_embedding_function())


# ============================================================================
# Configuration & Constants
# ============================================================================

DEFAULT_CHUNK_SIZE_WORDS = 180
DEFAULT_CHUNK_OVERLAP_WORDS = 30
BACKUP_DIR = DB_DIR / "backups"

FORMULA_OR_DEF_REGEX = re.compile(
    r"(?:"
    r"[A-Za-z0-9_]+\s*=\s*[A-Za-z0-9_(\-+*/^²³√]+"
    r"|^\s*(?:definition|formula|example|note|ସଂଜ୍ଞା|ସୂତ୍ର|ଉଦାହରଣ|ଅର୍ଥ)\s*[:\-–]"
    r"|^\s*(?:\d+[\.\)]|\([a-z0-9]+\)|[•\-*])\s+"
    r")",
    re.IGNORECASE,
)

HEADING_LINE_REGEX = re.compile(r"^\s*(?:#{1,4}\s+.+|\[SECTION:\s*.+\])\s*$")
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?।॥])\s+")


# ============================================================================
# Exceptions & Data Structures
# ============================================================================

class DatabaseBuildError(Exception):
    """Base exception for knowledge-base build failures."""


class DuplicateChunkError(DatabaseBuildError):
    """Raised when duplicate source or chunk identifiers are detected."""


class DatabaseConsistencyError(DatabaseBuildError):
    """Raised when ChromaDB and SQLite FTS5 fail post-build consistency verification."""


@dataclass
class CanonicalChunk:
    """Single canonical chunk representation shared by ChromaDB and SQLite FTS5."""
    chunk_id: str
    parent_id: str
    text: str
    formatted_passage: str
    content_hash: str
    metadata: Dict[str, Any]


@dataclass
class DatabaseBuildReport:
    dry_run: bool = False
    source_records_processed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    chunks_skipped_unchanged: int = 0
    chunks_removed_stale: int = 0
    chromadb_record_count: int = 0
    sqlite_record_count: int = 0
    backup_path: Optional[Path] = None
    verified_consistent: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Stage 2 Approved Record Loader
# ============================================================================

def load_approved_records_from_validator(
    input_dir: Optional[Path] = None,
    strict_links: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """
    Invokes `2_validate_links.py` so only structurally valid, approved records
    enter the database builder.
    """
    validator_path = PROJECT_ROOT / "2_validate_links.py"
    if not validator_path.exists():
        raise DatabaseBuildError(f"Validator module not found at: {validator_path}")

    spec = importlib.util.spec_from_file_location("validate_links_stage", validator_path)
    if spec is None or spec.loader is None:
        raise DatabaseBuildError("Failed to load 2_validate_links.py module spec.")

    val_mod = importlib.util.module_from_spec(spec)
    sys.modules["validate_links_stage"] = val_mod
    spec.loader.exec_module(val_mod)

    val_report = val_mod.run_validation_stage(
        input_dir=input_dir,
        approved_output_dir=EXTRACTED_JSON_DIR,
        strict_links=strict_links,
        write_approved=False,
    )

    err_msgs = [
        f"{iss.file_name} ({iss.record_id}): {iss.message}"
        for iss in val_report.errors
    ]
    warn_msgs = [
        f"{iss.file_name} ({iss.record_id}): {iss.message}"
        for iss in val_report.warnings
        if iss.category != "human_review"
    ]
    return val_report.approved_records, err_msgs, warn_msgs


# ============================================================================
# Context-Preserving Chunker (Preserves Definitions, Formulas & Headings)
# ============================================================================

def _word_count(text: str) -> int:
    return len(text.split())


def _is_intro_or_heading(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    if HEADING_LINE_REGEX.match(stripped):
        return True
    if stripped.endswith(":") or stripped.endswith(":-") or stripped.endswith("—"):
        return True
    return False


def _starts_with_formula_or_definition(block: str) -> bool:
    stripped = block.strip()
    if not stripped:
        return False
    first_line = stripped.split("\n", 1)[0].strip()
    return bool(FORMULA_OR_DEF_REGEX.search(first_line))


def _split_oversized_block(block: str, max_words: int) -> List[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT_REGEX.split(block.strip()) if s.strip()]
    if len(sentences) <= 1:
        words = block.split()
        if len(words) <= max_words:
            return [block.strip()]
        step = max(1, max_words)
        return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]

    fused_sentences: List[str] = []
    for sent in sentences:
        if fused_sentences and (
            _starts_with_formula_or_definition(sent) or _is_intro_or_heading(fused_sentences[-1])
        ):
            combined = f"{fused_sentences[-1]} {sent}"
            if _word_count(combined) <= int(max_words * 1.35):
                fused_sentences[-1] = combined
                continue
        fused_sentences.append(sent)

    sub_blocks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sent in fused_sentences:
        s_words = _word_count(sent)
        if current and (current_words + s_words > max_words):
            sub_blocks.append(" ".join(current))
            current = [sent]
            current_words = s_words
        else:
            current.append(sent)
            current_words += s_words

    if current:
        sub_blocks.append(" ".join(current))

    return sub_blocks


def split_text_preserving_context(
    text: str,
    section_heading: str = "",
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """
    Splits a section's text into self-contained chunks.
    - If the section already fits within `chunk_size_words`, returns it intact as 1 chunk.
    - Keeps headings, definitions, and mathematical formulas fused with their
      immediate introductory or explanatory context.
    - Applies configurable word overlap across chunk boundaries.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    effective_size = max(20, int(chunk_size_words))
    effective_overlap = max(0, min(int(chunk_overlap_words), effective_size // 2))

    if _word_count(clean_text) <= effective_size:
        return [clean_text]

    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", clean_text) if p.strip()]
    if len(raw_paragraphs) == 1:
        raw_paragraphs = [ln.strip() for ln in clean_text.split("\n") if ln.strip()]

    atomic_blocks: List[str] = []
    for para in raw_paragraphs:
        if _word_count(para) > effective_size:
            atomic_blocks.extend(_split_oversized_block(para, effective_size))
        else:
            atomic_blocks.append(para)

    cohesive_units: List[str] = []
    for block in atomic_blocks:
        if cohesive_units and (
            _is_intro_or_heading(cohesive_units[-1])
            or _starts_with_formula_or_definition(block)
        ):
            merged = f"{cohesive_units[-1]}\n{block}"
            if _word_count(merged) <= int(effective_size * 1.3):
                cohesive_units[-1] = merged
                continue
        cohesive_units.append(block)

    chunks: List[str] = []
    current_units: List[str] = []
    current_words = 0

    for unit in cohesive_units:
        u_words = _word_count(unit)
        if current_units and (current_words + u_words > effective_size):
            chunks.append("\n\n".join(current_units).strip())

            overlap_units: List[str] = []
            overlap_count = 0
            if effective_overlap > 0:
                for prev_u in reversed(current_units):
                    pw = _word_count(prev_u)
                    if overlap_count + pw <= effective_overlap or not overlap_units:
                        overlap_units.insert(0, prev_u)
                        overlap_count += pw
                        if overlap_count >= effective_overlap:
                            break
                if len(overlap_units) == 1 and overlap_count > effective_overlap * 2:
                    tail_words = overlap_units[0].split()[-effective_overlap:]
                    overlap_units = [" ".join(tail_words)]
                    overlap_count = len(tail_words)

            current_units = overlap_units + [unit]
            current_words = sum(_word_count(x) for x in current_units)
        else:
            current_units.append(unit)
            current_words += u_words

    if current_units:
        final_chunk = "\n\n".join(current_units).strip()
        if not chunks or final_chunk != chunks[-1]:
            chunks.append(final_chunk)

    clean_heading = section_heading.strip()
    if len(chunks) > 1 and clean_heading and clean_heading.lower() != "introduction":
        for idx in range(1, len(chunks)):
            if clean_heading.lower() not in chunks[idx].lower():
                chunks[idx] = f"[{clean_heading}]\n{chunks[idx]}"

    return chunks


def build_canonical_chunks(
    approved_records: Sequence[Dict[str, Any]],
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> List[CanonicalChunk]:
    """
    Converts validated source records into a single canonical list of `CanonicalChunk`
    objects. Enforces stable IDs, complete metadata preservation, and excludes
    generated evaluation questions from entering the knowledge base.
    """
    canonical_chunks: List[CanonicalChunk] = []
    seen_parent_ids: Set[str] = set()
    seen_chunk_ids: Set[str] = set()

    for rec in approved_records:
        if not isinstance(rec, dict):
            raise DatabaseBuildError("Approved record must be a dictionary.")

        parent_id = str(rec.get("id", "")).strip()
        if not parent_id:
            raise DatabaseBuildError("Approved record is missing 'id'.")

        if parent_id in seen_parent_ids:
            raise DuplicateChunkError(f"Duplicate parent record ID detected: '{parent_id}'")
        seen_parent_ids.add(parent_id)

        raw_text = str(rec.get("text", "")).strip()
        if not raw_text:
            raise DatabaseBuildError(f"Approved record '{parent_id}' has empty text.")

        meta = rec.get("metadata") or {}
        section_heading = str(meta.get("section_heading", "Introduction")).strip()

        text_splits = split_text_preserving_context(
            raw_text,
            section_heading=section_heading,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
        )
        if not text_splits:
            raise DatabaseBuildError(f"Chunker produced 0 chunks for record '{parent_id}'.")

        total_splits = len(text_splits)
        base_concept_tag = str(meta.get("concept_tag", f"{parent_id}_concept")).strip()
        english_terms = str(meta.get("english_tech_terms", "")).strip()

        for idx, chunk_text in enumerate(text_splits, start=1):
            chunk_id = f"{parent_id}_c{idx:03d}"
            if chunk_id in seen_chunk_ids:
                raise DuplicateChunkError(f"Duplicate chunk ID generated: '{chunk_id}'")
            seen_chunk_ids.add(chunk_id)

            chunk_concept_tag = (
                base_concept_tag if total_splits == 1 else f"{base_concept_tag}_c{idx:03d}"
            )
            prereq = (
                str(meta.get("prerequisite_concept", "none")).strip()
                if idx == 1
                else f"{base_concept_tag}_c{idx - 1:03d}" if total_splits > 1 else "none"
            )

            formatted_doc = format_passage(chunk_text, english_terms=english_terms)

            chunk_meta: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "class_level": int(meta["class_level"]),
                "subject": str(meta["subject"]).strip(),
                "chapter_id": str(meta.get("chapter_id", f"ch{int(meta['chapter_num']):02d}")).strip(),
                "chapter_num": int(meta["chapter_num"]),
                "chapter_name": str(meta["chapter_name"]).strip(),
                "language": str(meta["language"]).strip().lower(),
                "source_name": str(meta.get("source_name", "")).strip(),
                "page_numbers": str(meta.get("page_numbers", "")).strip(),
                "section_id": str(meta.get("section_id", "s001")).strip(),
                "section_heading": section_heading,
                "source_file": str(meta.get("source_file", "")).strip(),
                "concept_tag": chunk_concept_tag,
                "prerequisite_concept": prereq,
                "english_tech_terms": english_terms,
                "chunk_index": idx,
                "total_chunks_in_section": total_splits,
            }

            hash_payload = json.dumps(
                {"id": chunk_id, "passage": formatted_doc, "meta": chunk_meta},
                sort_keys=True,
                ensure_ascii=False,
            )
            content_hash = hashlib.sha256(hash_payload.encode("utf-8")).hexdigest()
            chunk_meta["content_hash"] = content_hash

            canonical_chunks.append(
                CanonicalChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    text=chunk_text,
                    formatted_passage=formatted_doc,
                    content_hash=content_hash,
                    metadata=chunk_meta,
                )
            )

    return canonical_chunks


# ============================================================================
# Incremental State Inspection & Backup Management
# ============================================================================

def inspect_existing_chroma_hashes(
    chroma_path: Path,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> Dict[str, str]:
    """Returns `{chunk_id: content_hash}` currently stored in ChromaDB, or `{}` if absent."""
    if not chroma_path.exists() or not any(chroma_path.iterdir()):
        return {}
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        existing_names = [
            c.name if hasattr(c, "name") else str(c) for c in client.list_collections()
        ]
        if collection_name not in existing_names:
            return {}
        col = client.get_collection(name=collection_name)
        data = col.get(include=["metadatas"])
        result: Dict[str, str] = {}
        for cid, meta in zip(data.get("ids", []), data.get("metadatas", [])):
            if isinstance(meta, dict) and meta.get("content_hash"):
                result[str(cid)] = str(meta["content_hash"])
            else:
                result[str(cid)] = ""
        return result
    except Exception:
        return {}


def inspect_existing_sqlite_hashes(sqlite_path: Path) -> Dict[str, str]:
    """Returns `{chunk_id: content_hash}` currently stored in SQLite FTS5, or `{}` if absent."""
    if not sqlite_path.exists():
        return {}
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='offline_chunks_fts';"
        )
        if not cur.fetchone():
            return {}
        cur.execute("PRAGMA table_info(offline_chunks_fts);")
        cols = {row[1] for row in cur.fetchall()}
        if "content_hash" not in cols:
            cur.execute("SELECT chunk_id FROM offline_chunks_fts;")
            return {str(r[0]): "" for r in cur.fetchall()}
        cur.execute("SELECT chunk_id, content_hash FROM offline_chunks_fts;")
        return {str(r[0]): str(r[1] or "") for r in cur.fetchall()}
    except Exception:
        return {}
    finally:
        conn.close()


def create_database_backup(
    chroma_path: Path,
    sqlite_path: Path,
    backup_root: Path = BACKUP_DIR,
) -> Optional[Path]:
    """
    Creates a timestamped backup of existing ChromaDB and SQLite databases
    before any production modification occurs.
    """
    has_chroma = chroma_path.exists() and any(chroma_path.iterdir())
    has_sqlite = sqlite_path.exists() and sqlite_path.stat().st_size > 0
    if not has_chroma and not has_sqlite:
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    snap_dir = backup_root / f"backup_{timestamp}_{int((time.time() % 1) * 1000):03d}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    if has_sqlite:
        shutil.copy2(sqlite_path, snap_dir / sqlite_path.name)
    if has_chroma:
        shutil.copytree(chroma_path, snap_dir / chroma_path.name, dirs_exist_ok=True)

    return snap_dir


def restore_database_backup(
    snap_dir: Optional[Path],
    chroma_path: Path,
    sqlite_path: Path,
) -> None:
    """Restores databases from a backup snapshot if a build or verification fails."""
    if snap_dir is None or not snap_dir.exists():
        return
    bak_sqlite = snap_dir / sqlite_path.name
    if bak_sqlite.exists():
        shutil.copy2(bak_sqlite, sqlite_path)
    bak_chroma = snap_dir / chroma_path.name
    if bak_chroma.exists():
        if chroma_path.exists():
            shutil.rmtree(chroma_path, ignore_errors=True)
        shutil.copytree(bak_chroma, chroma_path, dirs_exist_ok=True)


# ============================================================================
# SQLite FTS5 & ChromaDB Upsert/Delete Operations
# ============================================================================

def initialize_sqlite_schema(conn: sqlite3.Connection) -> None:
    """
    Initializes non-destructive runtime tables (`student_mastery_local`,
    `offline_sync_queue`) and the curriculum FTS5 table (`offline_chunks_fts`).
    """
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS student_mastery_local (
            concept_tag TEXT PRIMARY KEY,
            class_level INTEGER,
            mastery_score REAL DEFAULT 0.0,
            needs_remediation INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS offline_sync_queue (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            action_type TEXT,
            payload_json TEXT,
            synced INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    required_fts_cols = {
        "chunk_id",
        "parent_id",
        "class_level",
        "subject",
        "chapter_id",
        "chapter_num",
        "chapter_name",
        "language",
        "source_name",
        "page_numbers",
        "section_id",
        "section_heading",
        "source_file",
        "concept_tag",
        "prerequisite_concept",
        "english_tech_terms",
        "content_odia",
        "content_hash",
    }

    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='offline_chunks_fts';"
    )
    if cur.fetchone():
        cur.execute("PRAGMA table_info(offline_chunks_fts);")
        existing_cols = {row[1] for row in cur.fetchall()}
        if existing_cols != required_fts_cols:
            cur.execute("DROP TABLE offline_chunks_fts;")

    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS offline_chunks_fts USING fts5(
            chunk_id UNINDEXED,
            parent_id UNINDEXED,
            class_level UNINDEXED,
            subject UNINDEXED,
            chapter_id UNINDEXED,
            chapter_num UNINDEXED,
            chapter_name,
            language UNINDEXED,
            source_name UNINDEXED,
            page_numbers UNINDEXED,
            section_id UNINDEXED,
            section_heading,
            source_file UNINDEXED,
            concept_tag,
            prerequisite_concept UNINDEXED,
            english_tech_terms,
            content_odia,
            content_hash UNINDEXED
        );
    """)
    conn.commit()


def sync_sqlite_database(
    sqlite_path: Path,
    chunks_to_upsert: Sequence[CanonicalChunk],
    chunk_ids_to_delete: Sequence[str],
) -> int:
    """
    Updates SQLite via a staging copy first, preserving student state tables,
    applying incremental deletes/upserts, optimizing FTS5, and atomically
    swapping the staging file into place.
    """
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path = sqlite_path.with_suffix(".staging.db")
    if staging_path.exists():
        staging_path.unlink()

    if sqlite_path.exists():
        shutil.copy2(sqlite_path, staging_path)

    conn = sqlite3.connect(str(staging_path))
    try:
        initialize_sqlite_schema(conn)
        cur = conn.cursor()

        ids_to_remove = list(chunk_ids_to_delete) + [c.chunk_id for c in chunks_to_upsert]
        for cid in ids_to_remove:
            cur.execute("DELETE FROM offline_chunks_fts WHERE chunk_id = ?;", (cid,))

        for chunk in chunks_to_upsert:
            m = chunk.metadata
            cur.execute(
                """
                INSERT INTO offline_chunks_fts (
                    chunk_id, parent_id, class_level, subject, chapter_id,
                    chapter_num, chapter_name, language, source_name,
                    page_numbers, section_id, section_heading, source_file,
                    concept_tag, prerequisite_concept, english_tech_terms,
                    content_odia, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.parent_id,
                    int(m["class_level"]),
                    m["subject"],
                    m["chapter_id"],
                    int(m["chapter_num"]),
                    m["chapter_name"],
                    m["language"],
                    m["source_name"],
                    m["page_numbers"],
                    m["section_id"],
                    m["section_heading"],
                    m["source_file"],
                    m["concept_tag"],
                    m["prerequisite_concept"],
                    m["english_tech_terms"],
                    chunk.text,
                    chunk.content_hash,
                ),
            )

        cur.execute("INSERT INTO offline_chunks_fts(offline_chunks_fts) VALUES('optimize');")
        conn.commit()
        cur.execute("SELECT COUNT(*) FROM offline_chunks_fts;")
        total_rows = int(cur.fetchone()[0])
    finally:
        conn.close()

    shutil.move(str(staging_path), str(sqlite_path))
    return total_rows


def sync_chroma_database(
    chroma_path: Path,
    chunks_to_upsert: Sequence[CanonicalChunk],
    embeddings_to_upsert: Sequence[List[float]],
    chunk_ids_to_delete: Sequence[str],
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> int:
    """
    Applies incremental deletes and batch upserts (with precomputed E5 embeddings
    from `embeddings.py`) to ChromaDB.
    """
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    if chunk_ids_to_delete:
        collection.delete(ids=list(chunk_ids_to_delete))

    if chunks_to_upsert:
        if len(chunks_to_upsert) != len(embeddings_to_upsert):
            raise DatabaseBuildError(
                f"Chunk count ({len(chunks_to_upsert)}) does not match embedding count ({len(embeddings_to_upsert)})."
            )

        batch_size = 100
        for start in range(0, len(chunks_to_upsert), batch_size):
            batch_chunks = chunks_to_upsert[start : start + batch_size]
            batch_vecs = list(embeddings_to_upsert[start : start + batch_size])
            collection.upsert(
                ids=[c.chunk_id for c in batch_chunks],
                documents=[c.formatted_passage for c in batch_chunks],
                embeddings=batch_vecs,
                metadatas=[c.metadata for c in batch_chunks],
            )

    return int(collection.count())


# ============================================================================
# Post-Build Consistency Verifier
# ============================================================================

def verify_databases_consistency(
    expected_chunks: Sequence[CanonicalChunk],
    chroma_path: Path,
    sqlite_path: Path,
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> None:
    """
    Verifies that both ChromaDB and SQLite FTS5 contain the exact expected chunk IDs,
    matching record counts, E5 'passage: ' prefixes in ChromaDB, and identical
    content hashes and metadata across both stores.
    """
    expected_map = {c.chunk_id: c for c in expected_chunks}
    expected_ids = set(expected_map.keys())

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=collection_name)
    chroma_data = collection.get(include=["documents", "metadatas"])
    chroma_ids = set(chroma_data.get("ids", []))

    if chroma_ids != expected_ids:
        missing = expected_ids - chroma_ids
        extra = chroma_ids - expected_ids
        raise DatabaseConsistencyError(
            f"ChromaDB ID mismatch! Missing={sorted(missing)[:5]}, Extra={sorted(extra)[:5]}"
        )

    for cid, doc, meta in zip(
        chroma_data["ids"], chroma_data["documents"], chroma_data["metadatas"]
    ):
        exp = expected_map[cid]
        if not str(doc).startswith("passage: "):
            raise DatabaseConsistencyError(
                f"ChromaDB document '{cid}' is missing required 'passage: ' prefix."
            )
        if meta.get("content_hash") != exp.content_hash:
            raise DatabaseConsistencyError(
                f"ChromaDB content_hash mismatch for chunk '{cid}'."
            )
        if int(meta.get("class_level", -1)) != exp.metadata["class_level"]:
            raise DatabaseConsistencyError(
                f"ChromaDB class_level mismatch for chunk '{cid}'."
            )

    if not sqlite_path.exists():
        raise DatabaseConsistencyError(f"SQLite database file missing at {sqlite_path}")

    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT chunk_id, parent_id, class_level, subject, chapter_id,
                   language, source_name, content_odia, content_hash
            FROM offline_chunks_fts;
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    sqlite_ids = {str(r[0]) for r in rows}
    if len(rows) != len(expected_chunks) or sqlite_ids != expected_ids:
        missing = expected_ids - sqlite_ids
        extra = sqlite_ids - expected_ids
        raise DatabaseConsistencyError(
            f"SQLite FTS5 ID mismatch! Expected {len(expected_chunks)} rows, got {len(rows)}. "
            f"Missing={sorted(missing)[:5]}, Extra={sorted(extra)[:5]}"
        )

    for r in rows:
        cid, pid, cls_lvl, subj, ch_id, lang, src_name, content_odia, c_hash = r
        exp = expected_map[cid]
        if content_odia != exp.text:
            raise DatabaseConsistencyError(
                f"SQLite text content mismatch for chunk '{cid}'."
            )
        if c_hash != exp.content_hash:
            raise DatabaseConsistencyError(
                f"SQLite content_hash mismatch for chunk '{cid}'."
            )
        if int(cls_lvl) != exp.metadata["class_level"] or pid != exp.parent_id:
            raise DatabaseConsistencyError(
                f"SQLite metadata mismatch for chunk '{cid}'."
            )


# ============================================================================
# Main Orchestrator Function
# ============================================================================

def build_knowledge_base(
    approved_records: Optional[Sequence[Dict[str, Any]]] = None,
    input_dir: Optional[Path] = None,
    chroma_path: Path = CHROMA_DB_PATH,
    sqlite_path: Path = SQLITE_DB_PATH,
    backup_dir: Path = BACKUP_DIR,
    collection_name: str = CHROMA_COLLECTION_NAME,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    dry_run: bool = False,
    force_rebuild: bool = False,
) -> DatabaseBuildReport:
    """
    Executes the full Stage 3 database build:
    1. Loads approved records (if not passed directly).
    2. Splits records into context-preserving CanonicalChunks.
    3. Diffs against existing ChromaDB & SQLite hashes to skip unchanged chunks.
    4. Embeds only new/modified chunks via `embeddings.py`.
    5. Backs up existing databases, applies changes, and verifies 1:1 consistency.
    """
    report = DatabaseBuildReport(dry_run=dry_run)

    if approved_records is None:
        records, val_errs, val_warns = load_approved_records_from_validator(input_dir=input_dir)
        report.errors.extend(val_errs)
        report.warnings.extend(val_warns)
        if val_errs:
            raise DatabaseBuildError(
                f"Validation stage reported {len(val_errs)} blocking error(s). Refusing to build KB."
            )
    else:
        records = list(approved_records)

    if not records:
        raise DatabaseBuildError("No approved records available to build the knowledge base.")

    report.source_records_processed = len(records)

    canonical_chunks = build_canonical_chunks(
        records,
        chunk_size_words=chunk_size_words,
        chunk_overlap_words=chunk_overlap_words,
    )
    report.chunks_created = len(canonical_chunks)

    chroma_hashes = {} if force_rebuild else inspect_existing_chroma_hashes(chroma_path, collection_name)
    sqlite_hashes = {} if force_rebuild else inspect_existing_sqlite_hashes(sqlite_path)

    if sqlite_hashes and sqlite_path.exists():
        conn = sqlite3.connect(str(sqlite_path))
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(offline_chunks_fts);")
            existing_cols = {row[1] for row in cur.fetchall()}
            if "parent_id" not in existing_cols or "content_hash" not in existing_cols:
                sqlite_hashes = {}
        finally:
            conn.close()

    chunks_to_upsert: List[CanonicalChunk] = []
    for chunk in canonical_chunks:
        cid = chunk.chunk_id
        in_chroma = chroma_hashes.get(cid) == chunk.content_hash
        in_sqlite = sqlite_hashes.get(cid) == chunk.content_hash
        if in_chroma and in_sqlite:
            report.chunks_skipped_unchanged += 1
        else:
            chunks_to_upsert.append(chunk)

    current_id_set = {c.chunk_id for c in canonical_chunks}
    existing_all_ids = set(chroma_hashes.keys()) | set(sqlite_hashes.keys())
    stale_ids_to_delete = sorted(existing_all_ids - current_id_set)
    report.chunks_removed_stale = len(stale_ids_to_delete)

    if dry_run:
        report.embeddings_generated = 0
        report.chromadb_record_count = len(chroma_hashes)
        report.sqlite_record_count = len(sqlite_hashes)
        report.verified_consistent = True
        return report

    report.backup_path = create_database_backup(
        chroma_path=chroma_path,
        sqlite_path=sqlite_path,
        backup_root=backup_dir,
    )

    try:
        new_embeddings: List[List[float]] = []
        if chunks_to_upsert:
            new_embeddings = embed_passages(
                texts=[c.text for c in chunks_to_upsert],
                english_terms_list=[c.metadata["english_tech_terms"] for c in chunks_to_upsert],
            )
            report.embeddings_generated = len(new_embeddings)

        if force_rebuild and chroma_path.exists():
            client = chromadb.PersistentClient(path=str(chroma_path))
            existing_names = [
                c.name if hasattr(c, "name") else str(c) for c in client.list_collections()
            ]
            if collection_name in existing_names:
                client.delete_collection(collection_name)

        report.chromadb_record_count = sync_chroma_database(
            chroma_path=chroma_path,
            chunks_to_upsert=chunks_to_upsert,
            embeddings_to_upsert=new_embeddings,
            chunk_ids_to_delete=stale_ids_to_delete,
            collection_name=collection_name,
        )

        report.sqlite_record_count = sync_sqlite_database(
            sqlite_path=sqlite_path,
            chunks_to_upsert=chunks_to_upsert if not force_rebuild else canonical_chunks,
            chunk_ids_to_delete=stale_ids_to_delete if not force_rebuild else list(existing_all_ids),
        )

        verify_databases_consistency(
            expected_chunks=canonical_chunks,
            chroma_path=chroma_path,
            sqlite_path=sqlite_path,
            collection_name=collection_name,
        )
        report.verified_consistent = True

    except Exception as exc:
        report.verified_consistent = False
        report.errors.append(str(exc))
        restore_database_backup(report.backup_path, chroma_path, sqlite_path)
        raise DatabaseBuildError(
            f"Database build failed and was rolled back to backup: {exc}"
        ) from exc

    return report


def format_build_report(report: DatabaseBuildReport) -> str:
    mode = "DRY RUN (No Databases Modified)" if report.dry_run else "PRODUCTION BUILD"
    status = "READY (Verified Consistent)" if report.verified_consistent else "FAILED (Not Ready)"
    lines = [
        f"=== NIDANA STAGE 3: DATABASE BUILD REPORT [{mode}] ===",
        f"Knowledge Base Status     : {status}",
        f"Source Records Processed  : {report.source_records_processed}",
        f"Canonical Chunks Created  : {report.chunks_created}",
        f"Embeddings Generated      : {report.embeddings_generated}",
        f"Chunks Skipped (Unchanged): {report.chunks_skipped_unchanged}",
        f"Stale Chunks Removed      : {report.chunks_removed_stale}",
        "---------------------------------------------------------------",
        f"ChromaDB Record Count     : {report.chromadb_record_count}",
        f"SQLite FTS5 Record Count  : {report.sqlite_record_count}",
        f"Backup Snapshot           : {report.backup_path if report.backup_path else 'None (Initial Build)'}",
        "---------------------------------------------------------------",
        f"Errors                    : {len(report.errors)}",
        f"Warnings                  : {len(report.warnings)}",
    ]
    if report.errors:
        lines.append("\n--- ERRORS ---")
        for err in report.errors:
            lines.append(f"  [ERROR] {err}")
    if report.warnings:
        lines.append("\n--- WARNINGS ---")
        for warn in report.warnings:
            lines.append(f"  [WARN]  {warn}")
    return "\n".join(lines)


# ============================================================================
# Automated Unit Tests
# ============================================================================

class TestBuildDatabases(unittest.TestCase):
    """Tests chunking, metadata preservation, duplicate IDs, repeated builds, and consistency."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="nidana_db_test_"))
        self.chroma_dir = self.temp_dir / "chroma_test_kb"
        self.sqlite_file = self.temp_dir / "offline_test.db"
        self.backup_dir = self.temp_dir / "backups"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _sample_record(
        self,
        rec_id: str = "c10_geography_ch03_s001",
        text: str = "Water resources (ଜଳ ସମ୍ପଦ) are vital for irrigation and multipurpose river projects in Odisha.",
        class_level: int = 10,
        chapter_id: str = "ch03",
        concept_tag: str = "water_resources_intro",
    ) -> Dict[str, Any]:
        return {
            "id": rec_id,
            "text": text,
            "metadata": {
                "class_level": class_level,
                "subject": "Geography",
                "chapter_id": chapter_id,
                "chapter_num": 3,
                "chapter_name": "Water Resources",
                "language": "en",
                "source_name": "BSE Odisha Class 10 Geography",
                "page_numbers": "25-30",
                "section_id": "s001",
                "section_heading": "Water Conservation",
                "source_file": "data/source_text/10th Water resources.txt",
                "concept_tag": concept_tag,
                "prerequisite_concept": "none",
                "english_tech_terms": "water resources, irrigation",
                "diagnostic_question_odia": "ଏହି ପ୍ରଶ୍ନଟି ଡାଟାବେସରେ ରହିବ ନାହିଁ।",
            },
        }

    def test_1_chunking_preserves_definitions_formulas_and_boundaries(self) -> None:
        """Verifies long sections split cleanly while keeping definitions/formulas attached to intro lines."""
        para_1 = " ".join(["Motion in a straight line describes displacement over time."] * 6)
        formula_intro = "Newton's Second Law of Motion states that force equals mass times acceleration:"
        formula_line = "Formula: F = m × a and momentum p = m × v."
        para_2 = " ".join(["In Odisha board physics, SI units of force are measured in Newtons (N)."] * 6)

        long_text = f"{para_1}\n\n{formula_intro}\n\n{formula_line}\n\n{para_2}"
        chunks = split_text_preserving_context(
            long_text,
            section_heading="Newton's Laws",
            chunk_size_words=75,
            chunk_overlap_words=15,
        )
        self.assertGreaterEqual(len(chunks), 2)

        formula_chunks = [c for c in chunks if "F = m × a" in c]
        self.assertTrue(len(formula_chunks) >= 1)
        self.assertIn("Newton's Second Law of Motion states", formula_chunks[0])

    def test_2_metadata_preservation_and_eval_question_exclusion(self) -> None:
        """Verifies all required metadata fields are preserved on every chunk and eval questions are excluded."""
        rec = self._sample_record()
        chunks = build_canonical_chunks([rec], chunk_size_words=180, chunk_overlap_words=30)
        self.assertEqual(len(chunks), 1)
        c = chunks[0]

        self.assertEqual(c.chunk_id, "c10_geography_ch03_s001_c001")
        self.assertEqual(c.parent_id, "c10_geography_ch03_s001")
        self.assertEqual(c.metadata["class_level"], 10)
        self.assertEqual(c.metadata["subject"], "Geography")
        self.assertEqual(c.metadata["chapter_id"], "ch03")
        self.assertEqual(c.metadata["chapter_name"], "Water Resources")
        self.assertEqual(c.metadata["language"], "en")
        self.assertEqual(c.metadata["source_name"], "BSE Odisha Class 10 Geography")
        self.assertEqual(c.metadata["page_numbers"], "25-30")
        self.assertTrue(c.formatted_passage.startswith("passage: "))
        self.assertNotIn("diagnostic_question_odia", c.metadata)

    def test_3_duplicate_identifiers_rejected(self) -> None:
        """Verifies duplicate parent record IDs raise DuplicateChunkError."""
        r1 = self._sample_record(rec_id="c10_geography_ch03_s001")
        r2 = self._sample_record(rec_id="c10_geography_ch03_s001")
        with self.assertRaises(DuplicateChunkError):
            build_canonical_chunks([r1, r2])

    def test_4_repeated_builds_skip_unchanged_and_handle_mutations(self) -> None:
        """Verifies incremental build skips unchanged chunks and safely handles added/modified/removed records."""
        r1 = self._sample_record(rec_id="c10_geography_ch03_s001", concept_tag="concept_1")
        r2 = self._sample_record(
            rec_id="c10_geography_ch03_s002",
            text="Hirakud Dam is built across the Mahanadi River in Sambalpur.",
            concept_tag="concept_2",
        )

        rep1 = build_knowledge_base(
            approved_records=[r1, r2],
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name="test_col_inc",
        )
        self.assertEqual(rep1.chunks_created, 2)
        self.assertEqual(rep1.embeddings_generated, 2)
        self.assertEqual(rep1.chunks_skipped_unchanged, 0)
        self.assertTrue(rep1.verified_consistent)

        rep2 = build_knowledge_base(
            approved_records=[r1, r2],
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name="test_col_inc",
        )
        self.assertEqual(rep2.embeddings_generated, 0)
        self.assertEqual(rep2.chunks_skipped_unchanged, 2)
        self.assertEqual(rep2.chromadb_record_count, 2)
        self.assertEqual(rep2.sqlite_record_count, 2)
        self.assertIsNotNone(rep2.backup_path)

        r3 = self._sample_record(
            rec_id="c9_physics_ch01_s001",
            class_level=9,
            chapter_id="ch01",
            text="ଗତି ଏବଂ ବିଶ୍ରାମ (Motion and Rest): F = m × a।",
            concept_tag="concept_3",
        )
        rep3 = build_knowledge_base(
            approved_records=[r1, r3],
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name="test_col_inc",
        )
        self.assertEqual(rep3.chunks_skipped_unchanged, 1)
        self.assertEqual(rep3.embeddings_generated, 1)
        self.assertEqual(rep3.chunks_removed_stale, 1)
        self.assertEqual(rep3.chromadb_record_count, 2)
        self.assertEqual(rep3.sqlite_record_count, 2)
        self.assertTrue(rep3.verified_consistent)

    def test_5_database_consistency_verifier_detects_tampering(self) -> None:
        """Verifies that verify_databases_consistency catches discrepancies between ChromaDB and SQLite."""
        r1 = self._sample_record(rec_id="c10_geography_ch03_s001")
        build_knowledge_base(
            approved_records=[r1],
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name="test_col_tamper",
        )

        conn = sqlite3.connect(str(self.sqlite_file))
        conn.execute("DELETE FROM offline_chunks_fts;")
        conn.commit()
        conn.close()

        expected_chunks = build_canonical_chunks([r1])
        with self.assertRaises(DatabaseConsistencyError):
            verify_databases_consistency(
                expected_chunks=expected_chunks,
                chroma_path=self.chroma_dir,
                sqlite_path=self.sqlite_file,
                collection_name="test_col_tamper",
            )


# ============================================================================
# Command-Line Interface
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nidana Stage 3: Build ChromaDB and SQLite FTS5 databases from approved records."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input directory of records to validate & index (default: data/processed or data/extracted_json).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_WORDS,
        help=f"Target maximum words per chunk (default: {DEFAULT_CHUNK_SIZE_WORDS}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_WORDS,
        help=f"Target overlap words between split chunks (default: {DEFAULT_CHUNK_OVERLAP_WORDS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate chunking and incremental diffing without generating embeddings or modifying databases.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Re-embed and rebuild all chunks after backing up existing databases.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the built-in unit test suite and exit.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.run_tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestBuildDatabases)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    ensure_directories()

    try:
        report = build_knowledge_base(
            input_dir=args.input,
            chunk_size_words=args.chunk_size,
            chunk_overlap_words=args.chunk_overlap,
            dry_run=args.dry_run,
            force_rebuild=args.force_rebuild,
        )
        print("\n" + format_build_report(report))
        return 0 if report.verified_consistent else 1
    except Exception as exc:
        print(f"\n[FATAL BUILD FAILURE] {exc}", file=sys.stderr)
        print("Knowledge base is NOT ready.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())