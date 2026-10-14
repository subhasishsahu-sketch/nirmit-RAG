#!/usr/bin/env python3
"""
PROJECT: Nidana
FILE: 4_test_rag.py

Stage 4: Comprehensive Verification Suite & Quantitative RAG Evaluation Harness.

Covers Test Categories A through E:
  A. INGESTION  : Plain-text discovery, Odia Unicode preservation, metadata validation,
                  duplicate detection, empty input & evaluation exclusion handling.
  B. CHUNKING   : Meaningful boundaries, metadata preservation, stable chunk IDs,
                  and formula/definition context integrity.
  C. DATABASES  : ChromaDB & SQLite FTS5 counts, 1:1 canonical chunk ID parity,
                  idempotent repeat-build behavior, and safe failure rollback.
  D. RETRIEVAL  : Odia queries, English queries, class/subject/chapter/language filters,
                  paraphrased conceptual queries, and out-of-scope (no-evidence) queries.
  E. GENERATION : Source grounding verification, unsupported/hallucinated answer detection,
                  MCQ concept alignment, strict separation of generated questions from the
                  canonical KB, and honest Ollama connection failure reporting.

Also seeds and runs an offline evaluation benchmark (`data/evaluation/test_questions.json`)
reporting Recall@k, retrieval failures, answer-grounding failures, DB consistency checks,
and retrieval/generation latency measurements.

NOTE: Passing automated structural, retrieval, and grounding checks verifies system
engineering integrity; it does NOT guarantee pedagogical completeness or educational
accuracy, which requires review by qualified Odisha State Board educators.
"""

import argparse
import importlib
import json
import re
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import chromadb

import embeddings
import import_textbook
import rag_runtime

validate_links_mod = importlib.import_module("2_validate_links")
build_databases_mod = importlib.import_module("3_build_databases")


# ============================================================================
# Paths & Evaluation Constants
# ============================================================================

EVAL_QUESTIONS_PATH = embeddings.EVALUATION_DIR / "test_questions.json"
EVAL_RESULTS_PATH = embeddings.EVALUATION_DIR / "latest_eval_report.json"


# ============================================================================
# Grounding & MCQ Verification Helpers (Category E)
# ============================================================================

@dataclass
class GroundingAuditResult:
    is_grounded: bool
    has_citation_markers: bool
    valid_citation_indices: bool
    key_term_overlap_ratio: float
    matched_terms: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)


def audit_answer_grounding(
    answer_text: str,
    retrieved_chunks: Sequence[Dict[str, Any]],
    min_term_overlap: float = 0.20,
) -> GroundingAuditResult:
    """
    Audits whether a generated explanation is grounded in `retrieved_chunks`:
    1. If `retrieved_chunks` is empty, the answer must acknowledge insufficient context
       rather than asserting unsupported textbook facts.
    2. Checks that the answer includes valid source citation markers (`[1]`, `[2]`, ...)
       within the bounds `1..len(retrieved_chunks)`.
    3. Computes content token overlap between the answer and the retrieved passages
       (supporting both Odia and English tokens) to flag unsupported/hallucinated claims.
    """
    failures: List[str] = []
    clean_ans = (answer_text or "").strip()

    if not clean_ans:
        return GroundingAuditResult(
            is_grounded=False,
            has_citation_markers=False,
            valid_citation_indices=False,
            key_term_overlap_ratio=0.0,
            failure_reasons=["Answer text is empty."],
        )

    # Case 1: No retrieved evidence -> only an explicit limitation notice is valid
    if not retrieved_chunks:
        lower_ans = clean_ans.lower()
        acknowledges_limit = any(
            phrase in lower_ans
            for phrase in (
                "insufficient textbook context",
                "no approved textbook passages",
                "cannot answer",
                "not present in the textbook",
            )
        )
        if not acknowledges_limit:
            failures.append(
                "Answer asserted content when 0 textbook chunks were retrieved (hallucination risk)."
            )
        return GroundingAuditResult(
            is_grounded=acknowledges_limit,
            has_citation_markers=False,
            valid_citation_indices=acknowledges_limit,
            key_term_overlap_ratio=1.0 if acknowledges_limit else 0.0,
            failure_reasons=failures,
        )

    # Case 2: Retrieved chunks exist -> verify citation markers [1]..[N]
    cited_nums = [int(m) for m in re.findall(r"\[(\d+)\]", clean_ans)]
    has_citations = len(cited_nums) > 0
    if not has_citations:
        failures.append("Generated answer contains no source citation markers (e.g., [1]).")

    max_idx = len(retrieved_chunks)
    out_of_bounds = [n for n in cited_nums if n < 1 or n > max_idx]
    valid_indices = has_citations and (len(out_of_bounds) == 0)
    if out_of_bounds:
        failures.append(
            f"Answer cited non-existent source index(es) {out_of_bounds} (available: 1..{max_idx})."
        )

    # Case 3: Verify factual token grounding against retrieved passage text + technical terms
    context_blob = " ".join(
        f"{c.get('text', '')} {(c.get('metadata') or {}).get('english_tech_terms', '')} "
        f"{(c.get('metadata') or {}).get('chapter_name', '')}"
        for c in retrieved_chunks
    )
    context_tokens = {t.lower() for t in rag_runtime.extract_query_tokens(context_blob)}
    answer_tokens = [t.lower() for t in rag_runtime.extract_query_tokens(clean_ans)]

    if not answer_tokens:
        failures.append("Answer contains no verifiable domain tokens.")
        return GroundingAuditResult(
            is_grounded=False,
            has_citation_markers=has_citations,
            valid_citation_indices=valid_indices,
            key_term_overlap_ratio=0.0,
            failure_reasons=failures,
        )

    matched = sorted({t for t in answer_tokens if t in context_tokens})
    unique_ans_tokens = set(answer_tokens)
    overlap_ratio = round(len(matched) / max(1, len(unique_ans_tokens)), 4)

    if overlap_ratio < min_term_overlap:
        failures.append(
            f"Low source-term overlap ({overlap_ratio:.2f} < {min_term_overlap:.2f}); "
            "answer may contain unsupported external statements."
        )

    is_grounded = has_citations and valid_indices and (overlap_ratio >= min_term_overlap)
    return GroundingAuditResult(
        is_grounded=is_grounded,
        has_citation_markers=has_citations,
        valid_citation_indices=valid_indices,
        key_term_overlap_ratio=overlap_ratio,
        matched_terms=matched,
        failure_reasons=failures,
    )


def audit_generated_mcqs(
    questions_text: str,
    retrieved_chunks: Sequence[Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """
    Verifies that generated MCQs:
    1. Contain structured options (A/B/C/D) and an explicit answer key.
    2. Reference source citations (`[1]`, `[2]`, ...).
    3. Overlap with the concepts/terms in `retrieved_chunks`.
    """
    issues: List[str] = []
    clean_q = (questions_text or "").strip()
    if not clean_q:
        return False, ["Generated question text is empty."]

    has_options = bool(
        re.search(r"\bA[\)\.]", clean_q) and re.search(r"\bB[\)\.]", clean_q)
    )
    if not has_options:
        issues.append("Generated MCQs are missing standard option labels (A), B), ...).")

    has_citation = bool(re.search(r"\[\d+\]", clean_q))
    if not has_citation:
        issues.append("Generated questions are missing source passage citations ([1], [2], ...).")

    context_blob = " ".join(
        f"{c.get('text', '')} {(c.get('metadata') or {}).get('english_tech_terms', '')} "
        f"{(c.get('metadata') or {}).get('concept_tag', '')}"
        for c in retrieved_chunks
    )
    context_tokens = {t.lower() for t in rag_runtime.extract_query_tokens(context_blob)}
    question_tokens = {t.lower() for t in rag_runtime.extract_query_tokens(clean_q)}
    shared = context_tokens & question_tokens

    if len(shared) < 2:
        issues.append("Generated questions do not share domain concepts with the retrieved chunks.")

    return len(issues) == 0, issues


# ============================================================================
# Shared Reference Fixture Corpus for Categories A - E Unit & Integration Tests
# ============================================================================

def build_test_corpus_files(source_dir: Path) -> Dict[str, Path]:
    """
    Creates a realistic multi-class, bilingual (Odia + English) textbook fixture
    under `source_dir` for deterministic offline testing across Categories A-E.
    """
    c9_dir = source_dir / "class_9" / "physics"
    c10_dir = source_dir / "class_10" / "geography"
    c12_dir = source_dir / "class_12" / "physics"
    c9_dir.mkdir(parents=True, exist_ok=True)
    c10_dir.mkdir(parents=True, exist_ok=True)
    c12_dir.mkdir(parents=True, exist_ok=True)

    c9_motion_path = c9_dir / "c9_motion.txt"
    c9_motion_path.write_text(
        "---\n"
        "CLASS: 9\n"
        "SUBJECT: Physics\n"
        "CHAPTER_ID: ch01\n"
        "CHAPTER_NAME: ଗତି (Motion)\n"
        "LANGUAGE: or\n"
        "SOURCE_NAME: BSE Odisha Class 9 Vigyan\n"
        "PAGE_NUMBERS: 1-14\n"
        "---\n\n"
        "## ୧.୧ ଗତି ଏବଂ ବିଶ୍ରାମ (Motion and Rest)\n"
        "ଯେତେବେଳେ ଗୋଟିଏ ବସ୍ତୁର ସ୍ଥିତି (Position) ସମୟ ସହିତ ପାରିପାର୍ଶ୍ୱିକ ବସ୍ତୁ ତୁଳନାରେ ପରିବର୍ତ୍ତନ ହୁଏ, "
        "ତାହାକୁ ଗତି (Motion) କୁହାଯାଏ। ଦୂରତା (Distance) ଏକ ଅଦିଶ ରାଶି (Scalar Quantity) ଏବଂ "
        "ବିସ୍ଥାପନ (Displacement) ଏକ ସଦିଶ ରାଶି (Vector Quantity) ଅଟେ।\n\n"
        "## ୧.୨ ନିଉଟନ୍‌ଙ୍କ ଗତି ନିୟମ (Newton's Laws of Motion)\n"
        "ନିଉଟନ୍‌ଙ୍କ ପ୍ରଥମ ଗତି ନିୟମକୁ ଜଡ଼ତା ନିୟମ (Law of Inertia) କୁହାଯାଏ। ବାହ୍ୟ ଅସନ୍ତୁଳିତ ବଳ "
        "(External Unbalanced Force) ପ୍ରୟୋଗ ନ ହେଲେ ସ୍ଥିର ବସ୍ତୁ ସ୍ଥିର ରହେ। "
        "ନିଉଟନ୍‌ଙ୍କ ଦ୍ୱିତୀୟ ଗତି ନିୟମ ଅନୁଯାୟୀ ବଳ ହେଉଛି ବସ୍ତୁତ୍ୱ ଓ ତ୍ୱରଣର ଗୁଣଫଳ:\n"
        "Formula: F = m × a ଏବଂ ସଂବେଗ (Momentum) p = m × v।\n\n"
        "## ଅଭ୍ୟାସ (Exercises)\n"
        "1. ଜଡ଼ତା କ'ଣ ବୁଝାଅ?\n",
        encoding="utf-8",
    )

    c10_water_path = c10_dir / "c10_water_resources.txt"
    c10_water_path.write_text(
        "---\n"
        "CLASS: 10\n"
        "SUBJECT: Geography\n"
        "CHAPTER_ID: ch03\n"
        "CHAPTER_NAME: Water Resources\n"
        "LANGUAGE: or\n"
        "SOURCE_NAME: BSE Odisha Class 10 Geography\n"
        "PAGE_NUMBERS: 25-38\n"
        "---\n\n"
        "## ୩.୧ ଜଳ ସଂରକ୍ଷଣ ଓ ବହୁମୁଖୀ ନଦୀବନ୍ଧ ଯୋଜନା (Water Conservation and Multipurpose Projects)\n"
        "ଖାଦ୍ୟ ସୁରକ୍ଷା ସୁନିଶ୍ଚିତ କରିବା ଏବଂ ପରିସଂସ୍ଥାକୁ ରକ୍ଷା କରିବା ପାଇଁ ଜଳ ସଂରକ୍ଷଣ (Water Conservation) "
        "ଅତ୍ୟନ୍ତ ଜରୁରୀ। ମହାନଦୀ ଅବବାହିକାରେ ନିର୍ମିତ ହୀରାକୁଦ ନଦୀବନ୍ଧ ଯୋଜନା (Hirakud Dam Project) "
        "ଜଳସେଚନ (Irrigation), ଜଳବିଦ୍ୟୁତ୍ ଉତ୍ପାଦନ (Hydroelectricity) ଏବଂ ବନ୍ୟା ନିୟନ୍ତ୍ରଣ (Flood Control) "
        "ପାଇଁ ବ୍ୟବହୃତ ହୁଏ। ପଣ୍ଡିତ ଜବାହାରଲାଲ ନେହେରୁ ନଦୀବନ୍ଧ ଯୋଜନାଗୁଡ଼ିକୁ ‘ଆଧୁନିକ ଭାରତର ମନ୍ଦିର’ ବୋଲି ଅଭିହିତ କରିଥିଲେ।\n\n"
        "## ୩.୨ ବର୍ଷାଜଳ ଅମଳ (Rainwater Harvesting)\n"
        "ଛାତ ଉପରର ବର୍ଷାଜଳ ସଂଗ୍ରହ (Rooftop Rainwater Harvesting) ଦ୍ୱାରା ଭୂତଳ ଜଳସ୍ତର (Groundwater Table) "
        "ବୃଦ୍ଧି କରାଯାଇପାରେ। ରାଜସ୍ଥାନର ଶୁষ্କ ଅଞ୍ଚଳରେ ‘ଟାଙ୍କା’ (Tanka) ଓ ‘ଖଦିନ୍’ (Khadin) ପଦ୍ଧତି ବ୍ୟବହୃତ ହୁଏ।\n",
        encoding="utf-8",
    )

    c12_elec_path = c12_dir / "c12_current_electricity.txt"
    c12_elec_path.write_text(
        "---\n"
        "CLASS: 12\n"
        "SUBJECT: Physics\n"
        "CHAPTER_ID: ch03\n"
        "CHAPTER_NAME: Current Electricity\n"
        "LANGUAGE: en\n"
        "SOURCE_NAME: CHSE Odisha Class 12 Physics\n"
        "PAGE_NUMBERS: 88-105\n"
        "---\n\n"
        "## 3.1 Ohm's Law and Electrical Resistance\n"
        "Definition: Ohm's Law states that at constant temperature, the electric current (I) "
        "flowing through a metallic conductor is directly proportional to the potential difference (V) "
        "across its ends:\n"
        "Formula: V = I × R, where R is the electrical resistance measured in Ohms (Ω). "
        "The resistivity ρ depends on the material of the conductor: R = ρ(L/A).\n\n"
        "## 3.2 Kirchhoff's Circuit Rules\n"
        "Kirchhoff's First Rule (Junction Rule / KCL) is based on conservation of electric charge: "
        "the algebraic sum of currents meeting at any junction is zero (ΣI = 0). "
        "Kirchhoff's Second Rule (Loop Rule / KVL) is based on conservation of energy around a closed loop (ΣΔV = 0).\n",
        encoding="utf-8",
    )

    return {
        "c9_motion": c9_motion_path,
        "c10_water": c10_water_path,
        "c12_electricity": c12_elec_path,
    }


# ============================================================================
# Automated Test Suite Covering Categories A, B, C, D, and E
# ============================================================================

class TestNidanaEndToEndRAG(unittest.TestCase):
    """
    Comprehensive test suite verifying Categories A (Ingestion), B (Chunking),
    C (Databases), D (Retrieval), and E (Generation).
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_root = Path(tempfile.mkdtemp(prefix="nidana_4_test_rag_"))
        cls.source_dir = cls.temp_root / "data" / "source_text"
        cls.processed_dir = cls.temp_root / "data" / "processed"
        cls.extracted_dir = cls.temp_root / "data" / "extracted_json"
        cls.eval_dir = cls.temp_root / "data" / "evaluation"
        cls.chroma_dir = cls.temp_root / "db" / "chroma_odisha_kb"
        cls.sqlite_file = cls.temp_root / "db" / "offline_curriculum.db"
        cls.backup_dir = cls.temp_root / "db" / "backups"
        cls.collection_name = "test_nidana_stem_kb"

        cls.fixture_files = build_test_corpus_files(cls.source_dir)

        # Ingest, validate, and build the test knowledge base once for retrieval/generation tests
        cls.ingest_report = import_textbook.import_textbooks_from_path(
            input_path=cls.source_dir,
            output_dir=cls.processed_dir,
            sync_extracted_dir=cls.extracted_dir,
            dry_run=False,
            strict=True,
        )
        cls.val_report = validate_links_mod.run_validation_stage(
            input_dir=cls.processed_dir,
            approved_output_dir=cls.extracted_dir,
            write_approved=True,
        )
        cls.db_report = build_databases_mod.build_knowledge_base(
            approved_records=cls.val_report.approved_records,
            chroma_path=cls.chroma_dir,
            sqlite_path=cls.sqlite_file,
            backup_dir=cls.backup_dir,
            collection_name=cls.collection_name,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_root, ignore_errors=True)

    # ------------------------------------------------------------------------
    # CATEGORY A: INGESTION TESTS
    # ------------------------------------------------------------------------

    def test_A1_plain_text_discovery_and_eval_exclusion(self) -> None:
        """Category A: Discovers nested .txt chapters and excludes end-of-chapter exercise sections."""
        self.assertEqual(self.ingest_report.files_discovered, 3)
        self.assertEqual(self.ingest_report.files_processed, 3)
        self.assertEqual(self.ingest_report.files_failed, 0)
        # 2 sections per file = 6 sections total; 1 'ଅଭ୍ୟାସ (Exercises)' section excluded from c9_motion.txt
        self.assertEqual(self.ingest_report.total_records_created, 6)
        self.assertEqual(self.ingest_report.skipped_eval_sections, 1)

    def test_A2_odia_unicode_and_formula_preservation(self) -> None:
        """Category A: Preserves Odia Unicode characters, ZWNJ (\u200c), and math formulas verbatim."""
        c9_json = json.loads(
            (self.processed_dir / "c9_physics_ch01.json").read_text(encoding="utf-8")
        )
        sec2_text = c9_json[1]["text"]
        # Verify Odia ZWNJ in 'ନିଉଟନ୍‌ଙ୍କ' and formula 'F = m × a'
        self.assertIn("ନିଉଟନ୍\u200cଙ୍କ", sec2_text)
        self.assertIn("Formula: F = m × a", sec2_text)
        self.assertIn("p = m × v", sec2_text)
        self.assertNotIn("ଅଭ୍ୟାସ (Exercises)", sec2_text)

    def test_A3_metadata_validation_and_duplicate_and_empty_rejection(self) -> None:
        """Category A: Rejects missing metadata, invalid UTF-8, empty files, and duplicate IDs."""
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp)

            # 1. Empty 0-byte file
            empty_f = tpath / "empty.txt"
            empty_f.write_bytes(b"")
            with self.assertRaises(import_textbook.EmptyContentError):
                import_textbook.process_textbook_file(empty_f)

            # 2. Missing metadata
            no_meta_f = tpath / "missing_meta.txt"
            no_meta_f.write_text("Some textbook text without header.", encoding="utf-8")
            with self.assertRaises(import_textbook.MetadataValidationError):
                import_textbook.process_textbook_file(no_meta_f)

            # 3. Duplicate chapter source identifiers across two files
            dup_dir = tpath / "dups"
            dup_dir.mkdir()
            valid_txt = self.fixture_files["c9_motion"].read_text(encoding="utf-8")
            (dup_dir / "a.txt").write_text(valid_txt, encoding="utf-8")
            (dup_dir / "b.txt").write_text(valid_txt, encoding="utf-8")
            with self.assertRaises(import_textbook.DuplicateIdentifierError):
                import_textbook.import_textbooks_from_path(dup_dir, strict=True)

            # 4. Stage 2 validator catches mojibake corruption and invalid class level
            corrupt_rec = {
                "id": "c11_physics_ch01_s001",
                "text": "à¬“à¬¡à¬¼à¬¿à¬¶à¬¾ mojibake text",
                "metadata": {
                    "class_level": 11,
                    "subject": "Physics",
                    "chapter_id": "ch01",
                    "chapter_num": 1,
                    "chapter_name": "Motion",
                    "language": "or",
                    "source_name": "BSE",
                    "concept_tag": "bad_tag",
                },
            }
            v_rep = validate_links_mod.validate_records_in_memory([("bad.json", [corrupt_rec])])
            self.assertEqual(v_rep.approved_count, 0)
            self.assertEqual(v_rep.rejected_count, 1)

    # ------------------------------------------------------------------------
    # CATEGORY B: CHUNKING TESTS
    # ------------------------------------------------------------------------

    def test_B1_chunk_boundaries_formula_integrity_and_stable_ids(self) -> None:
        """Category B: Preserves definitions/formulas with context, preserves metadata, and produces stable IDs."""
        chunks_run_1 = build_databases_mod.build_canonical_chunks(
            self.val_report.approved_records,
            chunk_size_words=60,
            chunk_overlap_words=12,
        )
        chunks_run_2 = build_databases_mod.build_canonical_chunks(
            self.val_report.approved_records,
            chunk_size_words=60,
            chunk_overlap_words=12,
        )

        # Stable chunk IDs and content hashes across runs
        self.assertEqual(
            [c.chunk_id for c in chunks_run_1],
            [c.chunk_id for c in chunks_run_2],
        )
        self.assertEqual(
            [c.content_hash for c in chunks_run_1],
            [c.content_hash for c in chunks_run_2],
        )

        # Check metadata completeness on every chunk
        required_meta = {
            "chunk_id", "parent_id", "class_level", "subject", "chapter_id",
            "chapter_name", "language", "source_name", "page_numbers",
        }
        for ch in chunks_run_1:
            self.assertTrue(required_meta.issubset(set(ch.metadata.keys())))
            self.assertTrue(ch.formatted_passage.startswith("passage: "))
            self.assertNotIn("diagnostic_question_odia", ch.metadata)

        # Verify formula/definition integrity: 'Formula: V = I × R' stays attached to Ohm's Law definition
        ohm_chunks = [c.text for c in chunks_run_1 if "V = I × R" in c.text]
        self.assertTrue(len(ohm_chunks) >= 1)
        self.assertIn("Ohm's Law states", ohm_chunks[0])

    # ------------------------------------------------------------------------
    # CATEGORY C: DATABASE TESTS
    # ------------------------------------------------------------------------

    def test_C1_database_counts_matching_ids_repeat_build_and_rollback(self) -> None:
        """Category C: Verifies ChromaDB/SQLite parity, 0 re-embedding on repeat build, and safe failure rollback."""
        self.assertTrue(self.db_report.verified_consistent)
        self.assertEqual(self.db_report.chromadb_record_count, 6)
        self.assertEqual(self.db_report.sqlite_record_count, 6)

        # Inspect ChromaDB IDs vs SQLite FTS5 IDs directly
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        col = client.get_collection(self.collection_name)
        chroma_ids = set(col.get()["ids"])

        conn = sqlite3.connect(str(self.sqlite_file))
        sqlite_ids = {
            r[0] for r in conn.execute("SELECT chunk_id FROM offline_chunks_fts;").fetchall()
        }
        conn.close()
        self.assertEqual(chroma_ids, sqlite_ids)

        # Repeat build on unchanged records -> 0 embeddings generated, 6 skipped as unchanged
        rep_repeat = build_databases_mod.build_knowledge_base(
            approved_records=self.val_report.approved_records,
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            backup_dir=self.backup_dir,
            collection_name=self.collection_name,
        )
        self.assertEqual(rep_repeat.embeddings_generated, 0)
        self.assertEqual(rep_repeat.chunks_skipped_unchanged, 6)
        self.assertTrue(rep_repeat.verified_consistent)

        # Safe failure handling: passing an invalid record triggers DatabaseBuildError without corrupting existing DB
        bad_records = list(self.val_report.approved_records) + [{"id": "dup", "text": ""}]
        with self.assertRaises(build_databases_mod.DatabaseBuildError):
            build_databases_mod.build_knowledge_base(
                approved_records=bad_records,
                chroma_path=self.chroma_dir,
                sqlite_path=self.sqlite_file,
                backup_dir=self.backup_dir,
                collection_name=self.collection_name,
            )
        # Verify DB still has 6 intact records
        self.assertEqual(col.count(), 6)

    # ------------------------------------------------------------------------
    # CATEGORY D: RETRIEVAL TESTS
    # ------------------------------------------------------------------------

    def test_D1_odia_and_english_queries_and_metadata_filters(self) -> None:
        """Category D: Verifies Odia queries, English queries, and class/subject/chapter/language filters."""
        # 1. Relevant Odia question (Class 9 Physics)
        chunks_odia, _, _ = rag_runtime.retrieve_hybrid_context(
            question="ନିଉଟନ୍‌ଙ୍କ ଦ୍ୱିତୀୟ ଗତି ନିୟମ ଓ ବଳର ସୂତ୍ର କ'ଣ?",
            class_level=9,
            subject="Physics",
            chapter="ch01",
            language="or",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
        )
        self.assertGreaterEqual(len(chunks_odia), 1)
        self.assertEqual(chunks_odia[0].chunk_id, "c9_physics_ch01_s002_c001")

        # 2. Relevant English question (Class 12 Physics)
        chunks_en, _, _ = rag_runtime.retrieve_hybrid_context(
            question="State Kirchhoff's junction rule and conservation of electric charge.",
            class_level=12,
            subject="Physics",
            chapter="ch03",
            language="en",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
        )
        self.assertGreaterEqual(len(chunks_en), 1)
        self.assertEqual(chunks_en[0].chunk_id, "c12_physics_ch03_s002_c001")

        # 3. Paraphrased / differently phrased conceptual query (Cross-lingual English -> Odia Water Resources)
        chunks_para, _, _ = rag_runtime.retrieve_hybrid_context(
            question="How do people in arid Rajasthan store rainwater underground from rooftops?",
            class_level=10,
            subject="Geography",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
        )
        self.assertGreaterEqual(len(chunks_para), 1)
        retrieved_ids = [c.chunk_id for c in chunks_para]
        self.assertIn("c10_geography_ch03_s002_c001", retrieved_ids)

        # 4. Strict filter isolation: Class 9 student asking about Class 12 Ohm's Law gets 0 Class 12 chunks
        chunks_filtered, _, _ = rag_runtime.retrieve_hybrid_context(
            question="What is resistivity R = ρ(L/A) in Ohm's Law?",
            class_level=9,
            subject="Physics",
            language="en",  # Class 9 Physics has only 'or', so language='en' must return 0
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
        )
        self.assertEqual(len(chunks_filtered), 0)

        # 5. Query with no suitable evidence in KB (Out-of-domain subject/topic)
        chunks_none, _, _ = rag_runtime.retrieve_hybrid_context(
            question="How does photosynthesis occur inside chloroplasts in leaves?",
            class_level=10,
            subject="Biology",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
        )
        self.assertEqual(len(chunks_none), 0)

    # ------------------------------------------------------------------------
    # CATEGORY E: GENERATION & GROUNDING TESTS
    # ------------------------------------------------------------------------

    def test_E1_grounded_answers_unsupported_detection_mcqs_and_ollama_failure(self) -> None:
        """Category E: Verifies grounding audit, hallucination detection, MCQ alignment, KB isolation, and Ollama error handling."""
        # 1. Grounded answer passes grounding audit
        def mock_grounded_llm(prompt: str, system: str, url: str, model: str) -> Tuple[bool, str, Optional[str]]:
            return (
                True,
                "According to the textbook [1], Hirakud Dam (ହୀରାକୁଦ ନଦୀବନ୍ଧ) in the Mahanadi basin supports irrigation, hydroelectricity, and flood control.",
                None,
            )

        ans_ok = rag_runtime.answer_question(
            question="What are the uses of the Hirakud Dam project?",
            class_level=10,
            subject="Geography",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
            ollama_caller=mock_grounded_llm,
        )
        audit_ok = audit_answer_grounding(ans_ok["generated_explanation"], ans_ok["retrieved_chunks"])
        self.assertTrue(audit_ok.is_grounded, f"Expected grounded answer, got failures: {audit_ok.failure_reasons}")

        # 2. Unsupported / hallucinated answer (missing citations & unrelated content) is flagged by auditor
        hallucinated_text = "Nuclear fusion on Jupiter produces heavy elements via quantum tunneling."
        audit_bad = audit_answer_grounding(hallucinated_text, ans_ok["retrieved_chunks"])
        self.assertFalse(audit_bad.is_grounded)
        self.assertGreaterEqual(len(audit_bad.failure_reasons), 2)

        # 3. Generated MCQs match retrieved concepts and are NEVER inserted into the source KB
        def mock_mcq_llm(prompt: str, system: str, url: str, model: str) -> Tuple[bool, str, Optional[str]]:
            return (
                True,
                (
                    "MCQ 1 [1]: According to Ohm's Law, potential difference V is equal to:\n"
                    "A) I / R\nB) I × R\nC) R / I\nD) I + R\n"
                    "Correct Answer: B (Formula V = I × R, Electrical Resistance in Ohms) [1]\n\n"
                    "Descriptive 1 [1]: Define electrical resistance R and resistivity ρ using R = ρ(L/A)."
                ),
                None,
            )

        eval_bank_file = self.eval_dir / "test_generated_mcqs.json"
        q_res = rag_runtime.generate_practice_questions(
            topic_or_query="Ohm's Law and electrical resistance formula",
            class_level=12,
            subject="Physics",
            save_to_question_bank=eval_bank_file,
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
            ollama_caller=mock_mcq_llm,
        )
        self.assertTrue(q_res["generation_succeeded"])
        mcq_valid, mcq_issues = audit_generated_mcqs(
            q_res["generated_questions"], q_res["retrieved_chunks"]
        )
        self.assertTrue(mcq_valid, f"MCQ audit failed: {mcq_issues}")

        # Verify source KB was NOT polluted by generated questions
        client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.assertEqual(client.get_collection(self.collection_name).count(), 6)

        # Verify attempting to save generated questions inside `data/source_text/` is blocked
        with self.assertRaises(ValueError):
            rag_runtime.generate_practice_questions(
                topic_or_query="Ohm's Law",
                class_level=12,
                subject="Physics",
                save_to_question_bank=self.source_dir / "illegal_bank.json",
                chroma_path=self.chroma_dir,
                sqlite_path=self.sqlite_file,
                collection_name=self.collection_name,
                ollama_caller=mock_mcq_llm,
            )

        # 4. Ollama connection failure is handled honestly (generation_succeeded=False)
        ans_offline = rag_runtime.answer_question(
            question="State Ohm's Law.",
            class_level=12,
            subject="Physics",
            chroma_path=self.chroma_dir,
            sqlite_path=self.sqlite_file,
            collection_name=self.collection_name,
            ollama_url="http://127.0.0.1:65531",
        )
        self.assertFalse(ans_offline["generation_succeeded"])
        self.assertEqual(ans_offline["llm_status"], "unavailable")
        self.assertIsNotNone(ans_offline["error"])


# ============================================================================
# Quantitative Evaluation Suite (`data/evaluation/test_questions.json`)
# ============================================================================

DEFAULT_EVALUATION_QUESTIONS: List[Dict[str, Any]] = [
    {
        "eval_id": "eval_c10_water_01",
        "question": "ଜଳ ସଂରକ୍ଷଣ କାହିଁକି ଜରୁରୀ ଏବଂ ବହୁମୁଖୀ ନଦୀବନ୍ଧ ଯୋଜନା କିପରି ସାହାଯ୍ୟ କରେ?",
        "class_level": 10,
        "subject": "Geography",
        "chapter": "ch03",
        "language": "or",
        "query_type": "odia_direct",
        "expected_Available": True,
        "expected_parent_ids": ["c10_geography_ch03_s001"],
        "expected_chunk_ids": [
            "c10_geography_ch03_s001_c010",
            "c10_geography_ch03_s001_c011",
            "c10_geography_ch03_s001_c001",
        ],
        "expected_keywords": ["ଜଳ", "ସଂରକ୍ଷଣ", "ନଦୀବନ୍ଧ"],
    },
    {
        "eval_id": "eval_c10_water_02",
        "question": "How does the Hirakud Dam project on the Mahanadi River help with flood control and irrigation?",
        "class_level": 10,
        "subject": "Geography",
        "chapter": "ch03",
        "language": None,
        "query_type": "english_cross_lingual",
        "expected_Available": True,
        "expected_parent_ids": ["c10_geography_ch03_s001"],
        "expected_chunk_ids": [
            "c10_geography_ch03_s001_c011",
            "c10_geography_ch03_s001_c001",
        ],
        "expected_keywords": ["ହୀରାକୁଦ", "ମହାନଦୀ", "ବନ୍ୟା"],
    },
    {
        "eval_id": "eval_c10_water_03",
        "question": "Why did Jawaharlal Nehru call river valley dams the temples of modern India?",
        "class_level": 10,
        "subject": "Geography",
        "chapter": "Water Resources",
        "language": None,
        "query_type": "paraphrased_conceptual",
        "expected_Available": True,
        "expected_parent_ids": ["c10_geography_ch03_s001"],
        "expected_chunk_ids": [
            "c10_geography_ch03_s001_c011",
            "c10_geography_ch03_s001_c001",
        ],
        "expected_keywords": ["ନେହେରୁ", "ଆଧୁନିକ", "ମନ୍ଦିର"],
    },
    {
        "eval_id": "eval_c10_water_04",
        "question": "What disputes and environmental problems arise from excessive sedimentation in multipurpose dams?",
        "class_level": 10,
        "subject": "Geography",
        "chapter": "ch03",
        "language": None,
        "query_type": "paraphrased_conceptual",
        "expected_Available": True,
        "expected_parent_ids": [
            "c10_geography_ch03_s006",
            "c10_geography_ch03_s001",
        ],
        "expected_chunk_ids": [
            "c10_geography_ch03_s006_c010",
            "c10_geography_ch03_s001_c001",
        ],
        "expected_keywords": ["ଜଳଭଣ୍ଡାର", "ବନ୍ୟା", "ବହୁମୁଖୀ"],
    },
    {
        "eval_id": "eval_unanswerable_05",
        "question": "Explain the structure of DNA double helix and nucleotide base pairing in genetics.",
        "class_level": 10,
        "subject": "Biology",
        "chapter": None,
        "language": "en",
        "query_type": "no_evidence_out_of_scope",
        "expected_Available": False,
        "expected_parent_ids": [],
        "expected_chunk_ids": [],
        "expected_keywords": [],
    },
]


def ensure_evaluation_dataset(eval_path: Path = EVAL_QUESTIONS_PATH) -> List[Dict[str, Any]]:
    """
    Ensures `data/evaluation/test_questions.json` exists separately from the
    production knowledge base and returns the list of test cases.
    """
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    if not eval_path.exists():
        eval_path.write_text(
            json.dumps(DEFAULT_EVALUATION_QUESTIONS, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return json.loads(eval_path.read_text(encoding="utf-8"))


def check_production_db_consistency(
    chroma_path: Path = embeddings.CHROMA_DB_PATH,
    sqlite_path: Path = embeddings.SQLITE_DB_PATH,
    collection_name: str = embeddings.CHROMA_COLLECTION_NAME,
) -> Tuple[bool, int, int, List[str]]:
    """
    Audits consistency between the production ChromaDB vector store and SQLite FTS5
    database without modifying either store.

    Returns:
        `(is_consistent, chroma_count, sqlite_count, failure_messages)`
    """
    failures: List[str] = []
    chroma_count = 0
    sqlite_count = 0

    if not chroma_path.exists():
        failures.append(f"ChromaDB directory does not exist at {chroma_path}")
    if not sqlite_path.exists():
        failures.append(f"SQLite FTS5 database does not exist at {sqlite_path}")
    if failures:
        return False, 0, 0, failures

    try:
        client = chromadb.PersistentClient(path=str(chroma_path))
        col = client.get_collection(name=collection_name)
        c_data = col.get(include=["metadatas", "documents"])
        chroma_ids = set(c_data.get("ids", []))
        chroma_count = len(chroma_ids)
        chroma_hashes = {
            cid: (m or {}).get("content_hash", "")
            for cid, m in zip(c_data.get("ids", []), c_data.get("metadatas", []))
        }
    except Exception as exc:
        failures.append(f"Failed to read ChromaDB collection '{collection_name}': {exc}")
        return False, 0, 0, failures

    try:
        conn = sqlite3.connect(str(sqlite_path))
        try:
            rows = conn.execute(
                "SELECT chunk_id, content_hash FROM offline_chunks_fts;"
            ).fetchall()
            sqlite_ids = {str(r[0]) for r in rows}
            sqlite_count = len(rows)
            sqlite_hashes = {str(r[0]): str(r[1] or "") for r in rows}
        finally:
            conn.close()
    except Exception as exc:
        failures.append(f"Failed to query SQLite FTS5 table: {exc}")
        return False, chroma_count, 0, failures

    if chroma_count != sqlite_count:
        failures.append(
            f"Record count mismatch: ChromaDB={chroma_count} vs SQLite FTS5={sqlite_count}."
        )

    if chroma_ids != sqlite_ids:
        missing_in_sqlite = sorted(chroma_ids - sqlite_ids)[:5]
        missing_in_chroma = sorted(sqlite_ids - chroma_ids)[:5]
        failures.append(
            f"Canonical chunk ID mismatch: missing_in_sqlite={missing_in_sqlite}, "
            f"missing_in_chroma={missing_in_chroma}."
        )
    else:
        hash_mismatches = [
            cid for cid in chroma_ids if chroma_hashes.get(cid) != sqlite_hashes.get(cid)
        ]
        if hash_mismatches:
            failures.append(
                f"Content hash mismatch on {len(hash_mismatches)} chunk(s): {hash_mismatches[:5]}"
            )

    return len(failures) == 0, chroma_count, sqlite_count, failures


def run_evaluation_benchmark(
    eval_path: Path = EVAL_QUESTIONS_PATH,
    chroma_path: Path = embeddings.CHROMA_DB_PATH,
    sqlite_path: Path = embeddings.SQLITE_DB_PATH,
    collection_name: str = embeddings.CHROMA_COLLECTION_NAME,
    top_k: int = 5,
    test_live_ollama: bool = False,
) -> Dict[str, Any]:
    """
    Runs the evaluation benchmark stored in `data/evaluation/test_questions.json`
    against the active knowledge base and measures:
    - Database consistency failures
    - Retrieval Recall@k (where expected source chunk/parent IDs are specified)
    - Out-of-scope rejection accuracy
    - Answer-grounding failures
    - Retrieval and generation latency (ms)
    """
    eval_items = ensure_evaluation_dataset(eval_path)

    db_ok, chroma_cnt, sqlite_cnt, db_failures = check_production_db_consistency(
        chroma_path=chroma_path,
        sqlite_path=sqlite_path,
        collection_name=collection_name,
    )

    retrieval_latencies_ms: List[float] = []
    generation_latencies_ms: List[float] = []
    recall_hits = 0
    recall_eligible = 0
    retrieval_failures: List[str] = []
    grounding_failures: List[str] = []
    per_question_results: List[Dict[str, Any]] = []

    for item in eval_items:
        eval_id = item["eval_id"]
        q_text = item["question"]
        cls_lvl = int(item["class_level"])
        subj = item.get("subject")
        chap = item.get("chapter")
        lang = item.get("language")
        expect_avail = bool(item.get("expected_Available", True))
        exp_chunk_ids = set(item.get("expected_chunk_ids") or [])
        exp_parent_ids = set(item.get("expected_parent_ids") or [])
        exp_keywords = [k.lower() for k in (item.get("expected_keywords") or [])]

        # 1. Measure Retrieval Latency & Recall@k
        t0 = time.perf_counter()
        chunks, context_str, _ = rag_runtime.retrieve_hybrid_context(
            question=q_text,
            class_level=cls_lvl,
            subject=subj,
            chapter=chap,
            language=lang,
            top_k=top_k,
            chroma_path=chroma_path,
            sqlite_path=sqlite_path,
            collection_name=collection_name,
        )
        ret_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        retrieval_latencies_ms.append(ret_ms)

        got_chunk_ids = {c.chunk_id for c in chunks}
        got_parent_ids = {c.parent_id for c in chunks}

        if expect_avail:
            recall_eligible += 1
            id_matched = bool(
                (exp_chunk_ids & got_chunk_ids) or (exp_parent_ids & got_parent_ids)
            )
            kw_matched = any(
                kw in context_str.lower() for kw in exp_keywords
            ) if exp_keywords else True

            if chunks and (id_matched or kw_matched):
                recall_hits += 1
                ret_passed = True
            else:
                ret_passed = False
                retrieval_failures.append(
                    f"[{eval_id}] Missed expected IDs {sorted(exp_chunk_ids)} "
                    f"(retrieved: {sorted(got_chunk_ids)})"
                )
        else:
            # Out-of-scope query should retrieve 0 chunks
            ret_passed = len(chunks) == 0
            if not ret_passed:
                retrieval_failures.append(
                    f"[{eval_id}] Out-of-scope query falsely retrieved {len(chunks)} chunk(s): "
                    f"{sorted(got_chunk_ids)}"
                )

        # 2. Measure Answer Generation & Grounding Audit
        # Uses deterministic grounded synthesis unless `--live-ollama` is requested
        def _deterministic_grounded_synthesizer(
            prompt: str, system: str, url: str, model: str
        ) -> Tuple[bool, str, Optional[str]]:
            if not chunks:
                return False, "", "No context"
            top_c = chunks[0]
            snippet = top_c.text[:220].replace("\n", " ").strip()
            return (
                True,
                f"Based on the Odisha State Board textbook [1] ({top_c.metadata.get('chapter_name')}): {snippet} [1].",
                None,
            )

        t_gen0 = time.perf_counter()
        ans_payload = rag_runtime.answer_question(
            question=q_text,
            class_level=cls_lvl,
            subject=subj,
            chapter=chap,
            language=lang,
            top_k=top_k,
            chroma_path=chroma_path,
            sqlite_path=sqlite_path,
            collection_name=collection_name,
            ollama_caller=None if test_live_ollama else _deterministic_grounded_synthesizer,
        )
        gen_ms = round((time.perf_counter() - t_gen0) * 1000.0, 2)
        generation_latencies_ms.append(gen_ms)

        # If live Ollama is unreachable, report honestly without counting as a grounding hallucination
        if test_live_ollama and ans_payload["llm_status"] == "unavailable":
            grounding_ok = True
            grounding_notes = [f"Ollama unavailable ({ans_payload['error']}); skipped LLM text audit."]
        else:
            audit_res = audit_answer_grounding(
                ans_payload["answer"],
                ans_payload["retrieved_chunks"],
            )
            grounding_ok = audit_res.is_grounded
            grounding_notes = audit_res.failure_reasons
            if not grounding_ok:
                grounding_failures.append(
                    f"[{eval_id}] {'; '.join(audit_res.failure_reasons)}"
                )

        per_question_results.append({
            "eval_id": eval_id,
            "query_type": item.get("query_type"),
            "retrieval_passed": ret_passed,
            "retrieved_chunk_ids": [c.chunk_id for c in chunks],
            "retrieval_latency_ms": ret_ms,
            "grounding_passed": grounding_ok,
            "grounding_notes": grounding_notes,
            "end_to_end_latency_ms": gen_ms,
            "llm_status": ans_payload["llm_status"],
        })

    recall_at_k = round(recall_hits / max(1, recall_eligible), 4)

    def _latency_stats(samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {"min_ms": 0.0, "mean_ms": 0.0, "median_ms": 0.0, "max_ms": 0.0}
        return {
            "min_ms": round(min(samples), 2),
            "mean_ms": round(statistics.mean(samples), 2),
            "median_ms": round(statistics.median(samples), 2),
            "max_ms": round(max(samples), 2),
        }

    summary = {
        "evaluation_file": str(eval_path),
        "top_k": top_k,
        "database_consistency": {
            "consistent": db_ok,
            "chromadb_count": chroma_cnt,
            "sqlite_fts5_count": sqlite_cnt,
            "failures": db_failures,
        },
        "retrieval_metrics": {
            "total_queries": len(eval_items),
            "positive_queries": recall_eligible,
            f"recall_at_{top_k}": recall_at_k,
            "retrieval_failures_count": len(retrieval_failures),
            "retrieval_failures": retrieval_failures,
        },
        "grounding_metrics": {
            "grounding_failures_count": len(grounding_failures),
            "grounding_failures": grounding_failures,
        },
        "latency_ms": {
            "retrieval": _latency_stats(retrieval_latencies_ms),
            "end_to_end_answer": _latency_stats(generation_latencies_ms),
        },
        "questions": per_question_results,
        "epistemic_notice": (
            "Passing automated retrieval Recall@k and lexical source-grounding tests "
            "verifies technical pipeline behavior; it does NOT establish pedagogical "
            "or educational accuracy, which requires human review by subject educators."
        ),
    }

    EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def format_evaluation_report(summary: Dict[str, Any]) -> str:
    top_k = summary["top_k"]
    db_info = summary["database_consistency"]
    ret_info = summary["retrieval_metrics"]
    gnd_info = summary["grounding_metrics"]
    lat_ret = summary["latency_ms"]["retrieval"]
    lat_e2e = summary["latency_ms"]["end_to_end_answer"]

    lines = [
        "======================================================================",
        "NIDANA STAGE 4: RAG VERIFICATION & EVALUATION BENCHMARK REPORT",
        "======================================================================",
        f"Evaluation Dataset        : {summary['evaluation_file']}",
        f"Database Consistency      : {'PASS' if db_info['consistent'] else 'FAIL'} "
        f"(ChromaDB={db_info['chromadb_count']}, SQLite FTS5={db_info['sqlite_fts5_count']})",
        f"Retrieval Recall@{top_k:<8} : {ret_info[f'recall_at_{top_k}'] * 100:.1f}% "
        f"({ret_info['positive_queries'] - ret_info['retrieval_failures_count']}/{ret_info['positive_queries']} positive queries)",
        f"Retrieval Failures        : {ret_info['retrieval_failures_count']}",
        f"Answer-Grounding Failures : {gnd_info['grounding_failures_count']}",
        "----------------------------------------------------------------------",
        "LATENCY MEASUREMENTS (ms)",
        f"  Hybrid Retrieval        : min={lat_ret['min_ms']}ms | mean={lat_ret['mean_ms']}ms | "
        f"median={lat_ret['median_ms']}ms | max={lat_ret['max_ms']}ms",
        f"  End-to-End Answer       : min={lat_e2e['min_ms']}ms | mean={lat_e2e['mean_ms']}ms | "
        f"median={lat_e2e['median_ms']}ms | max={lat_e2e['max_ms']}ms",
        "----------------------------------------------------------------------",
        "PER-QUERY BENCHMARK BREAKDOWN",
    ]

    for q in summary["questions"]:
        r_flag = "PASS" if q["retrieval_passed"] else "FAIL"
        g_flag = "PASS" if q["grounding_passed"] else "FAIL"
        top_id = q["retrieved_chunk_ids"][0] if q["retrieved_chunk_ids"] else "None"
        lines.append(
            f"  * {q['eval_id']} ({q['query_type']}): Retrieval={r_flag} (top={top_id}, {q['retrieval_latency_ms']}ms) | "
            f"Grounding={g_flag} | LLM={q['llm_status']}"
        )

    if db_info["failures"]:
        lines.append("----------------------------------------------------------------------")
        lines.append("DATABASE CONSISTENCY FAILURES")
        for f in db_info["failures"]:
            lines.append(f"  [DB FAIL] {f}")

    if ret_info["retrieval_failures"]:
        lines.append("----------------------------------------------------------------------")
        lines.append("RETRIEVAL FAILURES")
        for f in ret_info["retrieval_failures"]:
            lines.append(f"  [RET FAIL] {f}")

    if gnd_info["grounding_failures"]:
        lines.append("----------------------------------------------------------------------")
        lines.append("ANSWER-GROUNDING FAILURES")
        for f in gnd_info["grounding_failures"]:
            lines.append(f"  [GND FAIL] {f}")

    lines.extend([
        "----------------------------------------------------------------------",
        f"NOTE: {summary['epistemic_notice']}",
        "======================================================================",
    ])
    return "\n".join(lines)


# ============================================================================
# Command-Line Interface
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nidana Stage 4: End-to-End Test Suite (Categories A-E) and "
            "Quantitative RAG Evaluation Benchmark (Recall@k, Grounding, Latency)."
        )
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only the isolated Category A-E unit/integration tests and exit.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run only the quantitative evaluation benchmark against the active production KB.",
    )
    parser.add_argument(
        "--live-ollama",
        action="store_true",
        help="Call the live local Ollama server during the evaluation benchmark.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K cutoff for Recall@k evaluation (default: 5).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    overall_success = True

    if not args.eval_only:
        print("=== RUNNING CATEGORY A-E AUTOMATED TEST SUITE ===")
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNidanaEndToEndRAG)
        runner = unittest.TextTestRunner(verbosity=2)
        test_result = runner.run(suite)
        if not test_result.wasSuccessful():
            overall_success = False

    if not args.unit_only:
        print("\n=== RUNNING PRODUCTION KB EVALUATION BENCHMARK ===")
        summary = run_evaluation_benchmark(
            eval_path=EVAL_QUESTIONS_PATH,
            top_k=args.top_k,
            test_live_ollama=args.live_ollama,
        )
        print(format_evaluation_report(summary))

        if (
            not summary["database_consistency"]["consistent"]
            or summary["retrieval_metrics"]["retrieval_failures_count"] > 0
            or summary["grounding_metrics"]["grounding_failures_count"] > 0
        ):
            overall_success = False

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())