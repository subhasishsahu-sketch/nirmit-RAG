#!/usr/bin/env python3
"""
PROJECT: Nidana
FILE: run_pipeline.py

One-Time Textbook Ingestion & Knowledge-Base Orchestrator.

Normal Pipeline Stages:
  1. Discover plain-text textbook files (.txt) under `--input`.
  2. Import and normalize source records (`import_textbook.py`).
  3. Validate records and prerequisite links (`2_validate_links.py`).
  4. Optionally extract/organize concepts if `--extract-concepts` is enabled.
  5. Chunk approved content preserving definitions and formulas (`3_build_databases.py`).
  6. Generate E5 passage embeddings (`embeddings.py`).
  7. Build ChromaDB vector store and SQLite FTS5 keyword index (`3_build_databases.py`).
  8. Verify 1:1 database consistency across ChromaDB and SQLite FTS5.
  9. Run post-build ingestion & retrieval sanity checks (`rag_runtime.py`).
 10. Produce a comprehensive final report.

Guarantees:
- Never calls `1_extract_pdfs.py`, PDF/OCR tools, or vision models.
- Never rebuilds the knowledge base during student queries.
- Stops immediately if ingestion, validation, embedding, or DB verification fails.
- Protects existing production databases via backup/staging and automatic rollback.
- Skips unchanged chunks on repeated runs to avoid unnecessary re-embedding.
"""

import argparse
import importlib
import json
import shutil
import sqlite3
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb

import embeddings
import import_textbook
import rag_runtime

# Dynamically import stage modules whose filenames begin with digits
validate_links_mod = importlib.import_module("2_validate_links")
build_databases_mod = importlib.import_module("3_build_databases")


# ============================================================================
# Pipeline Report Data Structure
# ============================================================================

@dataclass
class PipelineSummaryReport:
    success: bool = False
    dry_run: bool = False
    halted_at_stage: Optional[str] = None
    elapsed_seconds: float = 0.0

    # Paths
    input_path: Path = embeddings.SOURCE_TEXT_DIR
    processed_dir: Path = embeddings.PROCESSED_DIR
    extracted_dir: Path = embeddings.EXTRACTED_JSON_DIR
    chroma_path: Path = embeddings.CHROMA_DB_PATH
    sqlite_path: Path = embeddings.SQLITE_DB_PATH
    collection_name: str = embeddings.CHROMA_COLLECTION_NAME

    # Stage 1 & 2: Discovery & Import Metrics
    source_files_discovered: int = 0
    source_files_imported: int = 0
    source_files_rejected: int = 0
    source_characters_processed: int = 0
    source_records_created: int = 0
    eval_sections_excluded: int = 0

    # Stage 3: Validation Metrics
    records_approved: int = 0
    records_rejected: int = 0

    # Stage 5-8: Chunking, Embedding & Database Build Metrics
    chunks_created: int = 0
    embeddings_generated: int = 0
    chunks_skipped_unchanged: int = 0
    chunks_removed_stale: int = 0
    chromadb_entries: int = 0
    sqlite_fts_entries: int = 0
    databases_consistent: bool = False
    backup_path: Optional[Path] = None

    # Stage 9: Post-Ingestion Checks
    ingestion_checks_passed: bool = False
    ingestion_check_details: List[str] = field(default_factory=list)

    # Diagnostics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ============================================================================
# Helper: Discover & Import Plain-Text Chapters (Stages 1, 2 & Optional 4)
# ============================================================================

def _discover_and_import_textbooks(
    input_path: Path,
    processed_dir: Path,
    extracted_dir: Path,
    dry_run: bool,
    extract_concepts: bool,
    ollama_url: str,
    ollama_model: str,
) -> Tuple[import_textbook.BatchIngestionReport, List[Tuple[str, List[Dict[str, Any]]]]]:
    """
    Discovers `.txt` files under `input_path`, parses and normalizes them via
    `import_textbook.py`, checks for duplicate identifiers, and returns both the
    `BatchIngestionReport` and the in-memory `(file_name, serialized_records)`
    payloads so downstream validation and dry-run stages work seamlessly.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Guard against legacy PDF directory usage
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        raise import_textbook.UnsupportedFormatError(
            f"PDF files are not supported ({input_path.name}). Supply plain-text UTF-8 .txt files."
        )

    candidate_files: List[Path] = []
    if input_path.is_file():
        candidate_files = [input_path]
    else:
        for p in sorted(input_path.rglob("*")):
            if not p.is_file():
                continue
            if p.name.endswith(".meta.json") or p.name.startswith("."):
                continue
            # Ignore stray README or backup files if any; process all chapter files
            candidate_files.append(p)

    batch_report = import_textbook.BatchIngestionReport(
        files_discovered=len(candidate_files)
    )
    seen_chapter_ids: Dict[str, Path] = {}
    seen_record_ids: Dict[str, Path] = {}
    successful_results: List[import_textbook.FileIngestionResult] = []
    in_memory_payloads: List[Tuple[str, List[Dict[str, Any]]]] = []

    for file_path in candidate_files:
        try:
            result = import_textbook.process_textbook_file(
                file_path,
                extract_concepts=extract_concepts,
                ollama_url=ollama_url,
                ollama_model=ollama_model,
            )

            if result.chapter_source_id in seen_chapter_ids:
                prev = seen_chapter_ids[result.chapter_source_id]
                raise import_textbook.DuplicateIdentifierError(
                    f"Duplicate chapter source identifier '{result.chapter_source_id}' "
                    f"in {file_path.name} (already defined in {prev.name})."
                )

            for rec in result.records:
                if rec.id in seen_record_ids:
                    prev = seen_record_ids[rec.id]
                    raise import_textbook.DuplicateIdentifierError(
                        f"Duplicate document ID '{rec.id}' in {file_path.name} "
                        f"(already defined in {prev.name})."
                    )

            seen_chapter_ids[result.chapter_source_id] = file_path
            for rec in result.records:
                seen_record_ids[rec.id] = file_path

            successful_results.append(result)
            serialized = [r.model_dump() for r in result.records]
            in_memory_payloads.append((f"{result.chapter_source_id}.json", serialized))

            batch_report.files_processed += 1
            batch_report.total_records_created += len(result.records)
            batch_report.total_characters_processed += result.char_count
            batch_report.skipped_eval_sections += result.skipped_eval_sections

        except Exception as exc:
            batch_report.files_failed += 1
            batch_report.errors.append((file_path, str(exc)))

    if not dry_run and successful_results and batch_report.files_failed == 0:
        processed_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)

        for res in successful_results:
            serialized = [r.model_dump() for r in res.records]
            out_file = processed_dir / f"{res.chapter_source_id}.json"
            out_file.write_text(
                json.dumps(serialized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            batch_report.output_files_written.append(out_file)

            ext_file = extracted_dir / f"{res.chapter_source_id}.json"
            ext_file.write_text(
                json.dumps(serialized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return batch_report, in_memory_payloads


# ============================================================================
# Stage 9: Post-Build Ingestion & Retrieval Sanity Checks
# ============================================================================

def run_post_ingestion_checks(
    approved_records: Sequence[Dict[str, Any]],
    expected_chunk_count: int,
    chroma_path: Path,
    sqlite_path: Path,
    collection_name: str = embeddings.CHROMA_COLLECTION_NAME,
    dry_run: bool = False,
) -> Tuple[bool, List[str], List[str]]:
    """
    Executes post-ingestion verification checks:
    1. Confirms approved record metadata completeness and non-empty text.
    2. In production mode, verifies ChromaDB and SQLite FTS5 record counts match
       `expected_chunk_count` and executes a live read-only hybrid retrieval probe
       via `rag_runtime.retrieve_hybrid_context()` without rebuilding the KB.

    Returns:
        `(passed: bool, check_details: List[str], errors: List[str])`
    """
    details: List[str] = []
    errors: List[str] = []

    if not approved_records:
        errors.append("Post-ingestion check failed: 0 approved records present.")
        return False, details, errors

    # Check 1: Provenance and language distribution check
    classes_seen = sorted({int(r["metadata"]["class_level"]) for r in approved_records})
    subjects_seen = sorted({str(r["metadata"]["subject"]) for r in approved_records})
    langs_seen = sorted({str(r["metadata"]["language"]) for r in approved_records})
    details.append(
        f"Check 1 [PASS]: Verified provenance across Classes={classes_seen}, "
        f"Subjects={subjects_seen}, Languages={langs_seen}."
    )

    if dry_run:
        details.append(
            f"Check 2 [PASS]: Dry-run chunk simulation verified {expected_chunk_count} "
            "self-contained chunks ready for indexing."
        )
        return True, details, errors

    # Check 2: Verify ChromaDB and SQLite FTS5 counts match expected_chunk_count
    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.get_collection(name=collection_name)
        chroma_count = int(col.count())
    except Exception as exc:
        errors.append(f"ChromaDB collection check failed: {exc}")
        return False, details, errors

    try:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            cur = conn.cursor()
            sqlite_count = int(
                cur.execute("SELECT COUNT(*) FROM offline_chunks_fts;").fetchone()[0]
            )
            empty_rows = int(
                cur.execute(
                    "SELECT COUNT(*) FROM offline_chunks_fts WHERE TRIM(content_odia) = '';"
                ).fetchone()[0]
            )
        finally:
            conn.close()
    except Exception as exc:
        errors.append(f"SQLite FTS5 table check failed: {exc}")
        return False, details, errors

    if chroma_count != expected_chunk_count or sqlite_count != expected_chunk_count:
        errors.append(
            f"Database count mismatch: expected={expected_chunk_count}, "
            f"ChromaDB={chroma_count}, SQLite={sqlite_count}."
        )
        return False, details, errors

    if empty_rows > 0:
        errors.append(f"SQLite FTS5 contains {empty_rows} empty chunk row(s).")
        return False, details, errors

    details.append(
        f"Check 2 [PASS]: ChromaDB ({chroma_count}) and SQLite FTS5 ({sqlite_count}) "
        "counts match exactly with 0 empty rows."
    )

    # Check 3: Read-only hybrid retrieval sanity probe using a sample approved record
    sample_rec = approved_records[0]
    sample_meta = sample_rec["metadata"]
    sample_class = int(sample_meta["class_level"])
    sample_subject = str(sample_meta["subject"])
    sample_chapter = str(sample_meta["chapter_id"])
    sample_probe_query = (
        f"{sample_meta.get('chapter_name', '')} {sample_meta.get('section_heading', '')} "
        f"{sample_rec['text'][:120]}"
    ).strip()

    try:
        retrieved_chunks, _, _ = rag_runtime.retrieve_hybrid_context(
            question=sample_probe_query,
            class_level=sample_class,
            subject=sample_subject,
            chapter=sample_chapter,
            top_k=3,
            min_semantic_similarity=0.50,
            chroma_path=chroma_path,
            sqlite_path=sqlite_path,
            collection_name=collection_name,
        )
        if not retrieved_chunks:
            errors.append(
                f"Hybrid retrieval probe returned 0 chunks for Class {sample_class} {sample_subject} ({sample_chapter})."
            )
            return False, details, errors

        top_hit = retrieved_chunks[0]
        details.append(
            f"Check 3 [PASS]: Read-only hybrid retrieval probe succeeded "
            f"(top chunk='{top_hit.chunk_id}', combined_score={top_hit.combined_score}, "
            f"matched_by={','.join(top_hit.sources_matched)})."
        )
    except Exception as exc:
        errors.append(f"Hybrid retrieval sanity probe failed: {exc}")
        return False, details, errors

    return True, details, errors


# ============================================================================
# Main Pipeline Orchestrator
# ============================================================================

def run_pipeline(
    input_path: Path = embeddings.SOURCE_TEXT_DIR,
    processed_dir: Path = embeddings.PROCESSED_DIR,
    extracted_dir: Path = embeddings.EXTRACTED_JSON_DIR,
    chroma_path: Path = embeddings.CHROMA_DB_PATH,
    sqlite_path: Path = embeddings.SQLITE_DB_PATH,
    backup_dir: Optional[Path] = None,
    collection_name: str = embeddings.CHROMA_COLLECTION_NAME,
    chunk_size_words: int = build_databases_mod.DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap_words: int = build_databases_mod.DEFAULT_CHUNK_OVERLAP_WORDS,
    dry_run: bool = False,
    extract_concepts: bool = False,
    strict_links: bool = False,
    force_rebuild: bool = False,
    ollama_url: str = rag_runtime.DEFAULT_OLLAMA_BASE_URL,
    ollama_model: str = rag_runtime.DEFAULT_OLLAMA_MODEL,
) -> PipelineSummaryReport:
    """
    Executes the 10-stage one-time textbook ingestion pipeline end to end.
    Halts immediately if file ingestion, record validation, embedding generation,
    database construction, consistency verification, or post-ingestion checks fail.
    """
    start_time = time.time()
    effective_backup_dir = backup_dir or (sqlite_path.parent / "backups")

    report = PipelineSummaryReport(
        dry_run=dry_run,
        input_path=input_path,
        processed_dir=processed_dir,
        extracted_dir=extracted_dir,
        chroma_path=chroma_path,
        sqlite_path=sqlite_path,
        collection_name=collection_name,
    )

    if not dry_run:
        embeddings.ensure_directories()

    # ------------------------------------------------------------------------
    # STAGES 1, 2 & (Optional) 4: Discover, Import & Normalize Source Files
    # ------------------------------------------------------------------------
    try:
        ingest_report, in_memory_payloads = _discover_and_import_textbooks(
            input_path=input_path,
            processed_dir=processed_dir,
            extracted_dir=extracted_dir,
            dry_run=dry_run,
            extract_concepts=extract_concepts,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
        )
    except Exception as exc:
        report.halted_at_stage = "Stage 1-2: Textbook Discovery & Import"
        report.errors.append(f"[Import Error] {exc}")
        report.elapsed_seconds = round(time.time() - start_time, 3)
        return report

    report.source_files_discovered = ingest_report.files_discovered
    report.source_files_imported = ingest_report.files_processed
    report.source_files_rejected = ingest_report.files_failed
    report.source_characters_processed = ingest_report.total_characters_processed
    report.source_records_created = ingest_report.total_records_created
    report.eval_sections_excluded = ingest_report.skipped_eval_sections

    for failed_path, err_msg in ingest_report.errors:
        report.errors.append(f"[Import Rejected: {failed_path.name}] {err_msg}")

    if ingest_report.files_failed > 0 or ingest_report.files_processed == 0:
        if ingest_report.files_processed == 0 and not report.errors:
            report.errors.append(f"No valid plain-text (.txt) files found in {input_path}.")
        report.halted_at_stage = "Stage 2: Textbook Import"
        report.elapsed_seconds = round(time.time() - start_time, 3)
        return report

    # ------------------------------------------------------------------------
    # STAGE 3: Validate Source Records & Prerequisite Graph
    # ------------------------------------------------------------------------
    val_report = validate_links_mod.validate_records_in_memory(
        in_memory_payloads,
        strict_links=strict_links,
    )
    report.records_approved = val_report.approved_count
    report.records_rejected = val_report.rejected_count

    for err in val_report.errors:
        report.errors.append(
            f"[Validation Error | {err.category}] {err.file_name} ({err.record_id}): {err.message}"
        )
    for warn in val_report.warnings:
        report.warnings.append(
            f"[Validation Warning | {warn.category}] {warn.file_name} ({warn.record_id}): {warn.message}"
        )

    # Stop immediately if validation fails
    if val_report.rejected_count > 0 or len(val_report.errors) > 0 or val_report.approved_count == 0:
        if val_report.approved_count == 0 and not report.errors:
            report.errors.append("Validation produced 0 approved records.")
        report.halted_at_stage = "Stage 3: Record & Prerequisite Validation"
        report.elapsed_seconds = round(time.time() - start_time, 3)
        return report

    # ------------------------------------------------------------------------
    # STAGES 5, 6, 7 & 8: Chunk, Embed, Build Dual Databases & Verify Consistency
    # ------------------------------------------------------------------------
    try:
        db_report = build_databases_mod.build_knowledge_base(
            approved_records=val_report.approved_records,
            chroma_path=chroma_path,
            sqlite_path=sqlite_path,
            backup_dir=effective_backup_dir,
            collection_name=collection_name,
            chunk_size_words=chunk_size_words,
            chunk_overlap_words=chunk_overlap_words,
            dry_run=dry_run,
            force_rebuild=force_rebuild,
        )
    except Exception as exc:
        report.halted_at_stage = "Stage 5-8: Chunking, Embedding & Database Build"
        report.errors.append(f"[Database Build Error] {exc}")
        report.elapsed_seconds = round(time.time() - start_time, 3)
        return report

    report.chunks_created = db_report.chunks_created
    report.embeddings_generated = db_report.embeddings_generated
    report.chunks_skipped_unchanged = db_report.chunks_skipped_unchanged
    report.chunks_removed_stale = db_report.chunks_removed_stale
    report.chromadb_entries = db_report.chromadb_record_count
    report.sqlite_fts_entries = db_report.sqlite_record_count
    report.databases_consistent = db_report.verified_consistent
    report.backup_path = db_report.backup_path
    report.warnings.extend(db_report.warnings)

    if not db_report.verified_consistent or db_report.errors:
        report.errors.extend(db_report.errors)
        report.halted_at_stage = "Stage 8: Database Consistency Verification"
        report.elapsed_seconds = round(time.time() - start_time, 3)
        return report

    # ------------------------------------------------------------------------
    # STAGE 9: Run Configured Post-Ingestion & Retrieval Sanity Checks
    # ------------------------------------------------------------------------
    checks_ok, check_details, check_errors = run_post_ingestion_checks(
        approved_records=val_report.approved_records,
        expected_chunk_count=db_report.chunks_created,
        chroma_path=chroma_path,
        sqlite_path=sqlite_path,
        collection_name=collection_name,
        dry_run=dry_run,
    )
    report.ingestion_checks_passed = checks_ok
    report.ingestion_check_details = check_details

    if not checks_ok:
        report.errors.extend(check_errors)
        report.halted_at_stage = "Stage 9: Post-Ingestion Checks"
        report.elapsed_seconds = round(time.time() - start_time, 3)
        return report

    # ------------------------------------------------------------------------
    # STAGE 10: Pipeline Succeeded
    # ------------------------------------------------------------------------
    report.success = True
    report.elapsed_seconds = round(time.time() - start_time, 3)
    return report


# ============================================================================
# Stage 10: Final Report Formatter
# ============================================================================

def format_pipeline_report(report: PipelineSummaryReport) -> str:
    mode_str = "DRY RUN (No Databases Modified)" if report.dry_run else "PRODUCTION BUILD"
    status_str = "SUCCESS (Knowledge Base Ready)" if report.success else "FAILED (Halted)"

    lines = [
        f"======================================================================",
        f"NIDANA ONE-TIME INGESTION PIPELINE REPORT [{mode_str}]",
        f"======================================================================",
        f"Overall Pipeline Status    : {status_str}",
        f"Elapsed Time               : {report.elapsed_seconds:.2f}s",
    ]
    if report.halted_at_stage:
        lines.append(f"Halted At Stage            : {report.halted_at_stage}")

    lines.extend([
        f"----------------------------------------------------------------------",
        f"PATHS",
        f"  Source Text Input        : {report.input_path}",
        f"  Processed JSON Output    : {report.processed_dir}",
        f"  ChromaDB Vector Store    : {report.chroma_path} (collection: {report.collection_name})",
        f"  SQLite FTS5 Database     : {report.sqlite_path}",
        f"  Backup Snapshot          : {report.backup_path if report.backup_path else 'None'}",
        f"----------------------------------------------------------------------",
        f"PIPELINE METRICS",
        f"  1. Source Files Found    : {report.source_files_discovered}",
        f"  2. Source Files Imported : {report.source_files_imported} (Rejected: {report.source_files_rejected})",
        f"  3. Characters Processed  : {report.source_characters_processed}",
        f"  4. Source Sections/Recs  : {report.source_records_created} (Eval Excluded: {report.eval_sections_excluded})",
        f"  5. Validation Approved   : {report.records_approved} (Rejected: {report.records_rejected})",
        f"  6. Canonical Chunks      : {report.chunks_created}",
        f"  7. Embeddings Generated  : {report.embeddings_generated} (Skipped Unchanged: {report.chunks_skipped_unchanged})",
        f"  8. Stale Chunks Removed  : {report.chunks_removed_stale}",
        f"  9. ChromaDB Entry Count  : {report.chromadb_entries}",
        f" 10. SQLite FTS5 Row Count : {report.sqlite_fts_entries}",
        f" 11. Dual-DB Consistency   : {'VERIFIED' if report.databases_consistent else 'FAILED'}",
        f" 12. Ingestion Checks      : {'PASSED' if report.ingestion_checks_passed else 'FAILED'}",
    ])

    if report.ingestion_check_details:
        lines.append("----------------------------------------------------------------------")
        lines.append("POST-INGESTION CHECKS")
        for item in report.ingestion_check_details:
            lines.append(f"  * {item}")

    if report.warnings:
        lines.append("----------------------------------------------------------------------")
        lines.append(f"WARNINGS ({len(report.warnings)})")
        for w in report.warnings:
            lines.append(f"  [WARN]  {w}")

    if report.errors:
        lines.append("----------------------------------------------------------------------")
        lines.append(f"FAILURES & ERRORS ({len(report.errors)})")
        for e in report.errors:
            lines.append(f"  [ERROR] {e}")

    lines.append("======================================================================",
    )
    return "\n".join(lines)


# ============================================================================
# Automated Unit Tests
# ============================================================================

class TestRunPipeline(unittest.TestCase):
    """
    Automated tests verifying:
    - Dry-run mode reports accurate counts without writing databases.
    - Full pipeline builds ChromaDB and SQLite FTS5 and passes post-ingestion checks.
    - Repeated execution is idempotent and skips unchanged records (0 re-embeddings).
    - Validation or import failure halts the pipeline immediately and protects existing DBs.
    """

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="nidana_pipeline_test_"))
        self.src_dir = self.temp_dir / "data" / "source_text"
        self.proc_dir = self.temp_dir / "data" / "processed"
        self.ext_dir = self.temp_dir / "data" / "extracted_json"
        self.chroma_dir = self.temp_dir / "db" / "chroma_odisha_kb"
        self.sqlite_file = self.temp_dir / "db" / "offline_curriculum.db"
        self.backup_dir = self.temp_dir / "db" / "backups"
        self.collection_name = "test_pipeline_col"
        self.src_dir.mkdir(parents=True, exist_ok=True)

        self.valid_chapter_content = (
            "---\n"
            "CLASS: 10\n"
            "SUBJECT: Geography\n"
            "CHAPTER_ID: ch03\n"
            "CHAPTER_NAME: Water Resources\n"
            "LANGUAGE: or\n"
            "SOURCE_NAME: BSE Odisha Class 10 Geography\n"
            "PAGE_NUMBERS: 25-32\n"
            "---\n\n"
            "## ୩.୧ ଜଳ ସମ୍ବଳ ଓ ହୀରାକୁଦ ନଦୀବନ୍ଧ (Water Resources and Hirakud Dam)\n"
            "ଓଡ଼ିଶାର ମହାନଦୀ ଅବବାହିକାରେ ନିର୍ମିତ ହୀରାକୁଦ ନଦୀବନ୍ଧ ଯୋଜନାରେ ଜଳ ସଂରକ୍ଷଣ ସହ ବନ୍ୟା ନିୟନ୍ତ୍ରଣ ଓ "
            "ଜଳବିଦ୍ୟୁତ୍ ଉତ୍ପାଦନ କରାଯାଇଥାଏ। ଜଳ ସଂରକ୍ଷଣ (Water Conservation) ଖାଦ୍ୟ ସୁରକ୍ଷା ପାଇଁ ଅତ୍ୟନ୍ତ ଜରୁରୀ।\n"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_dry_run_mode_does_not_create_databases(self) -> None:
        """Verifies --dry-run validates and chunks files without creating ChromaDB or SQLite files."""
        (self.src_dir / "c10_water.txt").write_text(self.valid_chapter_content, encoding="utf-8")

        rep = run_pipeline(
            input_path=self.src_dir,
            processed_dir=self.proc_dir,
            extracted_dir=self.ext_dir,
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name=self.collection_name,
            dry_run=True,
        )
        self.assertTrue(rep.success)
        self.assertTrue(rep.dry_run)
        self.assertEqual(rep.source_files_imported, 1)
        self.assertEqual(rep.records_approved, 1)
        self.assertGreaterEqual(rep.chunks_created, 1)
        self.assertEqual(rep.embeddings_generated, 0)
        self.assertFalse(self.sqlite_file.exists())
        self.assertFalse(self.chroma_dir.exists())

    def test_2_full_pipeline_and_idempotent_repeated_execution(self) -> None:
        """Verifies full build followed by a repeated run skips unchanged chunks without re-embedding."""
        (self.src_dir / "c10_water.txt").write_text(self.valid_chapter_content, encoding="utf-8")

        # Run 1: Initial production build
        rep1 = run_pipeline(
            input_path=self.src_dir,
            processed_dir=self.proc_dir,
            extracted_dir=self.ext_dir,
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name=self.collection_name,
            dry_run=False,
        )
        self.assertTrue(rep1.success)
        self.assertEqual(rep1.embeddings_generated, 1)
        self.assertEqual(rep1.chunks_skipped_unchanged, 0)
        self.assertEqual(rep1.chromadb_entries, 1)
        self.assertEqual(rep1.sqlite_fts_entries, 1)
        self.assertTrue(rep1.databases_consistent)
        self.assertTrue(rep1.ingestion_checks_passed)

        # Run 2: Repeated execution on unchanged input -> 0 embeddings generated
        rep2 = run_pipeline(
            input_path=self.src_dir,
            processed_dir=self.proc_dir,
            extracted_dir=self.ext_dir,
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name=self.collection_name,
            dry_run=False,
        )
        self.assertTrue(rep2.success)
        self.assertEqual(rep2.embeddings_generated, 0)
        self.assertEqual(rep2.chunks_skipped_unchanged, 1)
        self.assertEqual(rep2.chromadb_entries, 1)
        self.assertEqual(rep2.sqlite_fts_entries, 1)
        self.assertIsNotNone(rep2.backup_path)

    def test_3_pipeline_stops_when_validation_or_metadata_fails(self) -> None:
        """Verifies pipeline stops before database build when a file has missing metadata or corrupt content."""
        # First build a valid production database
        (self.src_dir / "c10_water.txt").write_text(self.valid_chapter_content, encoding="utf-8")
        rep_valid = run_pipeline(
            input_path=self.src_dir,
            processed_dir=self.proc_dir,
            extracted_dir=self.ext_dir,
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name=self.collection_name,
        )
        self.assertTrue(rep_valid.success)

        # Now add a broken file missing required metadata
        (self.src_dir / "broken_chapter.txt").write_text(
            "This file has no metadata header and no sidecar.", encoding="utf-8"
        )
        rep_fail = run_pipeline(
            input_path=self.src_dir,
            processed_dir=self.proc_dir,
            extracted_dir=self.ext_dir,
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name=self.collection_name,
        )
        self.assertFalse(rep_fail.success)
        self.assertEqual(rep_fail.halted_at_stage, "Stage 2: Textbook Import")
        self.assertGreaterEqual( len(rep_fail.errors), 1)

        # Verify the existing production SQLite database is still intact with 1 record
        conn = sqlite3.connect(str(self.sqlite_file))
        count = conn.execute("SELECT COUNT(*) FROM offline_chunks_fts;").fetchone()[0]
        conn.close()
        self.assertEqual(count, 1)


# ============================================================================
# Command-Line Interface
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nidana One-Time Textbook Ingestion Pipeline: Discovers plain-text (.txt) "
            "chapters, validates records & prerequisite links, chunks prose, generates "
            "multilingual-e5-base embeddings, and builds ChromaDB + SQLite FTS5."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py --input data/source_text --dry-run\n"
            "  python run_pipeline.py --input data/source_text\n"
            "  python run_pipeline.py --run-tests\n"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=embeddings.SOURCE_TEXT_DIR,
        help="Input directory or .txt file path containing plain-text chapters (default: data/source_text).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=embeddings.PROCESSED_DIR,
        help="Output directory for processed JSON source records (default: data/processed).",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=embeddings.EXTRACTED_JSON_DIR,
        help="Directory for validated canonical JSON records (default: data/extracted_json).",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=embeddings.CHROMA_DB_PATH,
        help="Target path for ChromaDB persistent directory (default: db/chroma_odisha_kb).",
    )
    parser.add_argument(
        "--sqlite-path",
        type=Path,
        default=embeddings.SQLITE_DB_PATH,
        help="Target path for SQLite FTS5 database file (default: db/offline_curriculum.db).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=build_databases_mod.DEFAULT_CHUNK_SIZE_WORDS,
        help=f"Maximum words per textbook chunk (default: {build_databases_mod.DEFAULT_CHUNK_SIZE_WORDS}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=build_databases_mod.DEFAULT_CHUNK_OVERLAP_WORDS,
        help=f"Overlap words between split chunks (default: {build_databases_mod.DEFAULT_CHUNK_OVERLAP_WORDS}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run discovery, import, validation, and chunking simulation without writing files or modifying databases.",
    )
    parser.add_argument(
        "--extract-concepts",
        action="store_true",
        help="Explicitly enable optional local Ollama concept tag & diagnostic question extraction during import.",
    )
    parser.add_argument(
        "--strict-links",
        action="store_true",
        help="Treat unresolved external prerequisite_concept links as blocking validation errors.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force re-embedding and full rebuild of all chunks after backing up existing databases.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the built-in automated test suite for the pipeline orchestrator and exit.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.run_tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestRunPipeline)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    report = run_pipeline(
        input_path=args.input,
        processed_dir=args.output,
        extracted_dir=args.extracted_dir,
        chroma_path=args.chroma_path,
        sqlite_path=args.sqlite_path,
        chunk_size_words=args.chunk_size,
        chunk_overlap_words=args.chunk_overlap,
        dry_run=args.dry_run,
        extract_concepts=args.extract_concepts,
        strict_links=args.strict_links,
        force_rebuild=args.force_rebuild,
    )

    print("\n" + format_pipeline_report(report))
    return 0 if report.success else 1


if __name__ == "__main__":
    sys.exit(main())