#!/usr/bin/env python3
"""
PROJECT: Nidana
FILE: 2_validate_links.py

Stage 2: Pre-database validation and prerequisite-graph auditor.
Validates structured textbook records from `data/processed/` (or `data/extracted_json/`)
before database construction in `3_build_databases.py`.

Guarantees:
- Never modifies source textbooks, educational facts, or record prose.
- Never invents missing metadata or concepts.
- Never writes to ChromaDB or SQLite.
- Separates blocking errors (which reject a record) from warnings (such as
  unlinked external prerequisites or LLM-enriched fields requiring human review).
"""

import argparse
import json
import re
import sys
import tempfile
import unicodedata
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# ============================================================================
# Canonical Paths & Validation Constants
# ============================================================================

try:
    from embeddings import (
        EXTRACTED_JSON_DIR,
        PROCESSED_DIR,
        PROJECT_ROOT,
    )
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    EXTRACTED_JSON_DIR = PROJECT_ROOT / "data" / "extracted_json"

ALLOWED_CLASSES = {9, 10, 12}
ALLOWED_LANGUAGES = {"or", "en", "bilingual"}
MIN_CONTENT_LENGTH = 10

REQUIRED_METADATA_FIELDS = (
    "class_level",
    "subject",
    "chapter_id",
    "chapter_num",
    "chapter_name",
    "language",
    "source_name",
    "concept_tag",
)

PROVENANCE_FIELDS = (
    "source_name",
    "source_file",
    "section_id",
    "page_numbers",
)

# Stable ID pattern: e.g., c10_geography_ch03_s001
STABLE_ID_REGEX = re.compile(r"^c(9|10|12)_[a-z0-9_]+_[a-z0-9_]+$")
CONCEPT_TAG_REGEX = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,80}$")

# Unicode / Encoding Corruption Patterns
ODIA_CHAR_REGEX = re.compile(r"[\u0B00-\u0B7F]")
CONTROL_CHAR_REGEX = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Common UTF-8 Odia bytes (0xE0 0xAC / 0xE0 0xAD) mis-decoded as Latin-1/cp1252
MOJIBAKE_REGEX = re.compile(r"(?:à¬|à­|Ã¢|â€|ï¿½)")
# Legacy non-Unicode Odia font sequences (Akruti/ShreeLipi ASCII symbols inside words)
LEGACY_FONT_GARBAGE_REGEX = re.compile(r"[a-zA-Z0-9][÷êòùûüýþ]{2,}[a-zA-Z0-9]")


# ============================================================================
# Structured Validation Report Data Classes
# ============================================================================

@dataclass
class ValidationIssue:
    severity: str  # "ERROR" or "WARNING"
    category: str  # "duplicate_id", "missing_metadata", "empty_or_suspicious", "invalid_link", "human_review", "structure"
    file_name: str
    record_id: str
    message: str


@dataclass
class ValidationReport:
    source_files_count: int = 0
    total_records_inspected: int = 0
    approved_records: List[Dict[str, Any]] = field(default_factory=list)
    rejected_records: List[Dict[str, Any]] = field(default_factory=list)

    # Categorized tracking lists for reporting
    duplicate_identifiers: List[str] = field(default_factory=list)
    missing_metadata_issues: List[str] = field(default_factory=list)
    empty_or_suspicious_issues: List[str] = field(default_factory=list)
    invalid_links: List[str] = field(default_factory=list)
    human_review_flags: List[str] = field(default_factory=list)

    # Separated Errors (blocking) and Warnings (non-blocking / advisory)
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)

    @property
    def approved_count(self) -> int:
        return len(self.approved_records)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected_records)

    def add_error(self, category: str, file_name: str, record_id: str, message: str) -> None:
        issue = ValidationIssue("ERROR", category, file_name, record_id, message)
        self.errors.append(issue)
        formatted = f"[{file_name} | id={record_id}] {message}"
        if category == "duplicate_id":
            self.duplicate_identifiers.append(formatted)
        elif category == "missing_metadata":
            self.missing_metadata_issues.append(formatted)
        elif category == "empty_or_suspicious":
            self.empty_or_suspicious_issues.append(formatted)
        elif category == "invalid_link":
            self.invalid_links.append(formatted)

    def add_warning(self, category: str, file_name: str, record_id: str, message: str) -> None:
        issue = ValidationIssue("WARNING", category, file_name, record_id, message)
        self.warnings.append(issue)
        formatted = f"[{file_name} | id={record_id}] {message}"
        if category == "invalid_link":
            self.invalid_links.append(formatted)
        elif category == "human_review":
            self.human_review_flags.append(formatted)
        elif category == "empty_or_suspicious":
            self.empty_or_suspicious_issues.append(formatted)


# ============================================================================
# Core Content, Provenance & Metadata Validators
# ============================================================================

def check_unicode_and_content_integrity(
    text: Any,
    declared_language: Optional[str],
) -> Tuple[List[str], List[str]]:
    """
    Inspects textbook text for empty content, encoding corruption, mojibake,
    or script mismatch without modifying the text.

    Returns:
        (blocking_errors, non_blocking_warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(text, str):
        errors.append(f"Record 'text' must be a string, got {type(text).__name__}.")
        return errors, warnings

    stripped = text.strip()
    if not stripped:
        errors.append("Record 'text' is empty or whitespace-only.")
        return errors, warnings

    if stripped.lower() in ("passage:", "query:"):
        errors.append("Record 'text' contains only an embedding prefix without content.")
        return errors, warnings

    if len(stripped) < MIN_CONTENT_LENGTH:
        errors.append(
            f"Record 'text' is suspiciously short ({len(stripped)} chars < minimum {MIN_CONTENT_LENGTH})."
        )

    # 1. Check for UTF-8 replacement character (\ufffd)
    if "\ufffd" in text:
        errors.append("Detected Unicode replacement character (U+FFFD ''); text has encoding corruption.")

    # 2. Check for illegal binary/control characters
    if CONTROL_CHAR_REGEX.search(text):
        errors.append("Detected non-printable ASCII control characters in text.")

    # 3. Check for UTF-8 -> Latin1/Windows-1252 Mojibake corruption
    if MOJIBAKE_REGEX.search(text):
        errors.append("Detected UTF-8 mojibake corruption pattern (e.g., mis-decoded Odia byte sequence).")

    # 4. Check for legacy non-Unicode font garbage (e.g., Akruti/ShreeLipi)
    if LEGACY_FONT_GARBAGE_REGEX.search(text):
        errors.append("Detected suspected legacy non-Unicode ASCII font sequence in text.")

    # 5. Check NFC normalization consistency (warn if not NFC normalized, do not silently mutate)
    if unicodedata.normalize("NFC", text) != text:
        warnings.append("Text is not in Unicode NFC canonical composition form.")

    # 6. Check language vs script alignment
    if declared_language == "or" and not ODIA_CHAR_REGEX.search(text):
        errors.append("Metadata declares language='or' (Odia), but text contains zero Odia Unicode characters.")

    return errors, warnings


def validate_single_record_structure(
    record: Any,
    file_name: str,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Validates a single record's top-level keys, stable ID, required metadata,
    provenance preservation, and LLM human-review flags.

    Returns:
        (list of (category, error_message), list of (category, warning_message))
    """
    errors: List[Tuple[str, str]] = []
    warnings: List[Tuple[str, str]] = []

    if not isinstance(record, dict):
        errors.append(("missing_metadata", "Record entry is not a JSON object."))
        return errors, warnings

    rec_id = record.get("id")
    if not isinstance(rec_id, str) or not rec_id.strip():
        errors.append(("missing_metadata", "Missing or empty record 'id'."))
        rec_id_str = "<missing_id>"
    else:
        rec_id_str = rec_id.strip()
        if not STABLE_ID_REGEX.match(rec_id_str):
            errors.append((
                "missing_metadata",
                f"Invalid stable identifier format '{rec_id_str}'. Expected pattern 'c<9|10|12>_<subject>_<chapter>_<section>'.",
            ))

    meta = record.get("metadata")
    if not isinstance(meta, dict):
        errors.append(("missing_metadata", "Missing or invalid 'metadata' dictionary."))
        return errors, warnings

    # 1. Check required metadata keys (never invent missing metadata)
    for req_key in REQUIRED_METADATA_FIELDS:
        if req_key not in meta or meta[req_key] is None or str(meta[req_key]).strip() == "":
            # Allow backward compatibility if chapter_id is missing but chapter_num exists?
            # Requirement 3 states: Validate required class, subject, chapter, and language metadata.
            errors.append(("missing_metadata", f"Missing required metadata field: '{req_key}'."))

    # 2. Validate class_level
    raw_class = meta.get("class_level")
    if isinstance(raw_class, bool) or not isinstance(raw_class, int) or raw_class not in ALLOWED_CLASSES:
        errors.append((
            "missing_metadata",
            f"Invalid class_level {raw_class!r}; must be an integer in {sorted(ALLOWED_CLASSES)}.",
        ))
    elif rec_id_str.startswith("c") and not rec_id_str.startswith(f"c{raw_class}_"):
        errors.append((
            "missing_metadata",
            f"Record ID '{rec_id_str}' prefix does not match metadata class_level={raw_class}.",
        ))

    # 3. Validate chapter_num
    raw_ch_num = meta.get("chapter_num")
    if isinstance(raw_ch_num, bool) or not isinstance(raw_ch_num, int) or raw_ch_num < 1:
        errors.append((
            "missing_metadata",
            f"Invalid chapter_num {raw_ch_num!r}; must be a positive integer.",
        ))

    # 4. Validate language
    raw_lang = meta.get("language")
    declared_lang = str(raw_lang).strip().lower() if raw_lang is not None else None
    if declared_lang not in ALLOWED_LANGUAGES:
        errors.append((
            "missing_metadata",
            f"Invalid language {raw_lang!r}; must be one of {sorted(ALLOWED_LANGUAGES)}.",
        ))

    # 5. Validate concept_tag format
    raw_concept = str(meta.get("concept_tag", "")).strip()
    if raw_concept and not CONCEPT_TAG_REGEX.match(raw_concept):
        errors.append((
            "missing_metadata",
            f"Invalid concept_tag '{raw_concept}'; must be lowercase alphanumeric/underscore.",
        ))

    # 6. Check provenance preservation
    if not str(meta.get("source_name", "")).strip():
        errors.append(("missing_metadata", "Missing source provenance 'source_name'."))
    if not str(meta.get("source_file", "")).strip() and not str(meta.get("section_id", "")).strip():
        warnings.append((
            "missing_metadata",
            "Record metadata is missing 'source_file' or 'section_id' provenance fields.",
        ))

    # 7. Validate content and Unicode integrity
    content_errs, content_warns = check_unicode_and_content_integrity(
        record.get("text"), declared_lang
    )
    for ce in content_errs:
        errors.append(("empty_or_suspicious", ce))
    for cw in content_warns:
        warnings.append(("empty_or_suspicious", cw))

    # 8. Flag LLM-transformed / LLM-enriched fields for human review
    is_llm_transformed = bool(
        meta.get("llm_transformed")
        or meta.get("generated_by_llm")
        or meta.get("llm_extracted")
        or str(meta.get("diagnostic_question_odia", "")).strip()
    )
    if is_llm_transformed:
        warnings.append((
            "human_review",
            "Record contains LLM-generated or LLM-enriched content (e.g., diagnostic_question_odia / concept extraction); flagged for human educator review.",
        ))

    return errors, warnings


# ============================================================================
# Prerequisite Graph Validator (Cycles, Self-Links, Grade Inversion, Missing Targets)
# ============================================================================

def validate_prerequisite_graph(
    records: Sequence[Dict[str, Any]],
    record_file_map: Dict[str, str],
    report: ValidationReport,
    strict_links: bool = False,
) -> Set[str]:
    """
    Validates `prerequisite_concept` links across all structurally valid records.
    Returns a set of record IDs that must be rejected due to hard graph violations
    (self-dependencies, circular dependencies, or higher-grade prerequisites).
    """
    rejected_ids: Set[str] = set()

    # Map concept_tag -> list of records defining that concept
    concept_registry: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        tag = str(rec["metadata"].get("concept_tag", "")).strip()
        if tag:
            concept_registry.setdefault(tag, []).append(rec)

    # 1. Check direct link validity for each record
    adj_list: Dict[str, str] = {}  # concept_tag -> prerequisite_concept
    for rec in records:
        rec_id = rec["id"]
        fname = record_file_map.get(rec_id, "<memory>")
        meta = rec["metadata"]
        concept_tag = str(meta.get("concept_tag", "")).strip()
        prereq = str(meta.get("prerequisite_concept", "none")).strip()

        if not prereq or prereq.lower() == "none":
            continue

        # Check A: Self-referential prerequisite
        if prereq == concept_tag:
            report.add_error(
                "invalid_link",
                fname,
                rec_id,
                f"Self-referential prerequisite link: concept '{concept_tag}' cannot require itself.",
            )
            rejected_ids.add(rec_id)
            continue

        # Check B: Prerequisite concept not present in corpus
        if prereq not in concept_registry:
            msg = (
                f"Unresolved prerequisite_concept '{prereq}' referenced by '{concept_tag}' "
                "(target concept_tag not found in current corpus)."
            )
            if strict_links:
                report.add_error("invalid_link", fname, rec_id, msg)
                rejected_ids.add(rec_id)
            else:
                report.add_warning("invalid_link", fname, rec_id, msg)
            continue

        # Check C: Grade-ceiling violation (lower class requiring a higher class prerequisite)
        current_class = int(meta["class_level"])
        target_classes = [int(t["metadata"]["class_level"]) for t in concept_registry[prereq]]
        min_target_class = min(target_classes)
        if min_target_class > current_class:
            report.add_error(
                "invalid_link",
                fname,
                rec_id,
                f"Grade-level inversion: Class {current_class} concept '{concept_tag}' cannot require "
                f"higher Class {min_target_class} prerequisite '{prereq}'.",
            )
            rejected_ids.add(rec_id)
            continue

        adj_list[concept_tag] = prereq

    # 2. Check for cycles in the prerequisite graph (A -> B -> A)
    visited: Set[str] = set()
    for start_node in list(adj_list.keys()):
        if start_node in visited:
            continue
        path_set: Set[str] = set()
        curr: Optional[str] = start_node
        while curr is not None and curr in adj_list:
            if curr in path_set:
                # Cycle detected: reject all records participating in the cycle
                for cycle_tag in path_set:
                    for r in concept_registry.get(cycle_tag, []):
                        rid = r["id"]
                        fname = record_file_map.get(rid, "<memory>")
                        report.add_error(
                            "invalid_link",
                            fname,
                            rid,
                            f"Circular prerequisite dependency detected involving '{cycle_tag}'.",
                        )
                        rejected_ids.add(rid)
                break
            path_set.add(curr)
            visited.add(curr)
            curr = adj_list.get(curr)

    return rejected_ids


# ============================================================================
# Batch Validation Entry Points
# ============================================================================

def validate_records_in_memory(
    files_payload: List[Tuple[str, Any]],
    strict_links: bool = False,
) -> ValidationReport:
    """
    Validates a collection of (file_name, parsed_json_payload) pairs and returns
    a comprehensive ValidationReport without touching disk or databases.
    """
    report = ValidationReport(source_files_count=len(files_payload))

    seen_record_ids: Dict[str, str] = {}
    seen_concept_keys: Dict[Tuple[int, str, str, str], str] = {}
    structurally_valid: List[Dict[str, Any]] = []
    record_file_map: Dict[str, str] = {}

    for file_name, payload in files_payload:
        if not isinstance(payload, list):
            report.add_error(
                "missing_metadata",
                file_name,
                "<file_root>",
                "JSON file root must be a list of concept records.",
            )
            continue

        if len(payload) == 0:
            report.add_error(
                "empty_or_suspicious",
                file_name,
                "<empty_file>",
                "JSON file contains 0 records.",
            )
            continue

        for idx, raw_rec in enumerate(payload):
            report.total_records_inspected += 1
            rec_id = (
                str(raw_rec.get("id", "")).strip()
                if isinstance(raw_rec, dict) and raw_rec.get("id")
                else f"<index_{idx}>"
            )

            errs, warns = validate_single_record_structure(raw_rec, file_name)

            # Check duplicate record ID
            if rec_id != f"<index_{idx}>":
                if rec_id in seen_record_ids:
                    prev_file = seen_record_ids[rec_id]
                    errs.append((
                        "duplicate_id",
                        f"Duplicate record ID '{rec_id}' (first seen in {prev_file}).",
                    ))
                else:
                    seen_record_ids[rec_id] = file_name

            # Check duplicate (class_level, subject, chapter_id, concept_tag)
            if isinstance(raw_rec, dict) and isinstance(raw_rec.get("metadata"), dict):
                m = raw_rec["metadata"]
                c_tag = str(m.get("concept_tag", "")).strip()
                c_lvl = m.get("class_level")
                subj = str(m.get("subject", "")).strip().lower()
                ch_id = str(m.get("chapter_id", "")).strip().lower()
                if c_tag and isinstance(c_lvl, int) and subj and ch_id:
                    concept_key = (c_lvl, subj, ch_id, c_tag)
                    if concept_key in seen_concept_keys:
                        prev_id = seen_concept_keys[concept_key]
                        errs.append((
                            "duplicate_id",
                            f"Duplicate concept_tag '{c_tag}' in Class {c_lvl} {subj} {ch_id} (already used by {prev_id}).",
                        ))
                    else:
                        seen_concept_keys[concept_key] = rec_id

            for cat, w_msg in warns:
                report.add_warning(cat, file_name, rec_id, w_msg)

            if errs:
                for cat, e_msg in errs:
                    report.add_error(cat, file_name, rec_id, e_msg)
                report.rejected_records.append(raw_rec if isinstance(raw_rec, dict) else {"raw": raw_rec})
            else:
                structurally_valid.append(raw_rec)
                record_file_map[rec_id] = file_name

    # Run Pass 2: Prerequisite Graph Validation across structurally valid records
    graph_rejected_ids = validate_prerequisite_graph(
        structurally_valid,
        record_file_map=record_file_map,
        report=report,
        strict_links=strict_links,
    )

    for rec in structurally_valid:
        if rec["id"] in graph_rejected_ids:
            report.rejected_records.append(rec)
        else:
            report.approved_records.append(rec)

    return report


def resolve_input_directory(input_dir: Optional[Path] = None) -> Path:
    """
    Selects `data/processed/` by default if it contains JSON files, falling back
    to `data/extracted_json/`.
    """
    if input_dir is not None:
        return input_dir
    if PROCESSED_DIR.exists() and any(PROCESSED_DIR.glob("*.json")):
        return PROCESSED_DIR
    return EXTRACTED_JSON_DIR


def run_validation_stage(
    input_dir: Optional[Path] = None,
    approved_output_dir: Optional[Path] = EXTRACTED_JSON_DIR,
    strict_links: bool = False,
    write_approved: bool = True,
) -> ValidationReport:
    """
    Loads JSON record files from disk, validates structure, metadata, Unicode,
    and prerequisite links, and optionally writes approved records to
    `data/extracted_json/`.
    """
    target_dir = resolve_input_directory(input_dir)
    if not target_dir.exists():
        return ValidationReport(source_files_count=0)

    json_files = sorted(target_dir.glob("*.json"))
    files_payload: List[Tuple[str, Any]] = []
    pre_load_Report = ValidationReport(source_files_count=len(json_files))

    for jf in json_files:
        try:
            raw_bytes = jf.read_bytes()
            text = raw_bytes.decode("utf-8", errors="strict")
            parsed = json.loads(text)
            files_payload.append((jf.name, parsed))
        except UnicodeDecodeError as exc:
            pre_load_Report.add_error(
                "empty_or_suspicious",
                jf.name,
                "<file_encoding>",
                f"Invalid UTF-8 byte sequence at offset {exc.start}: {exc.reason}",
            )
        except json.JSONDecodeError as exc:
            pre_load_Report.add_error(
                "missing_metadata",
                jf.name,
                "<malformed_json>",
                f"Malformed JSON syntax: {exc}",
            )

    report = validate_records_in_memory(files_payload, strict_links=strict_links)
    report.source_files_count = len(json_files)
    # Merge any file-level UTF-8/JSON errors
    report.errors = pre_load_Report.errors + report.errors
    report.empty_or_suspicious_issues = (
        pre_load_Report.empty_or_suspicious_issues + report.empty_or_suspicious_issues
    )
    report.missing_metadata_issues = (
        pre_load_Report.missing_metadata_issues + report.missing_metadata_issues
    )

    # Write approved records grouped by chapter file to data/extracted_json/ if enabled
    if write_approved and approved_output_dir is not None and report.approved_records:
        approved_output_dir.mkdir(parents=True, exist_ok=True)
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in report.approved_records:
            m = rec["metadata"]
            ch_key = f"c{m['class_level']}_{str(m['subject']).strip().lower()}_{str(m['chapter_id']).strip().lower()}"
            ch_key = re.sub(r"[^a-z0-9_]+", "_", ch_key)
            grouped.setdefault(ch_key, []).append(rec)

        for ch_key, recs in grouped.items():
            out_path = approved_output_dir / f"{ch_key}.json"
            out_path.write_text(
                json.dumps(recs, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    return report


def validate_all_records(
    input_dir: Optional[Path] = None,
    strict_links: bool = False,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Backward-compatible helper for `3_build_databases.py` and `run_pipeline.py`.
    Returns `(approved_records, warning_and_error_messages)`.
    """
    report = run_validation_stage(
        input_dir=input_dir,
        strict_links=strict_links,
        write_approved=True,
    )
    messages = [
        f"[{iss.severity}:{iss.category}] {iss.file_name} ({iss.record_id}): {iss.message}"
        for iss in (report.errors + report.warnings)
    ]
    return report.approved_records, messages


def format_validation_report(report: ValidationReport) -> str:
    """Formats the human-readable validation report required by the specification."""
    lines = [
        "=== NIDANA STAGE 2: RECORD & PREREQUISITE VALIDATION REPORT ===",
        f"Source Files Inspected    : {report.source_files_count}",
        f"Total Records Inspected   : {report.total_records_inspected}",
        f"Approved Records (Ready)  : {report.approved_count}",
        f"Rejected Records (Blocked): {report.rejected_count}",
        "---------------------------------------------------------------",
        f"Duplicate Identifiers     : {len(report.duplicate_identifiers)}",
        f"Missing/Invalid Metadata  : {len(report.missing_metadata_issues)}",
        f"Empty/Suspicious Content  : {len(report.empty_or_suspicious_issues)}",
        f"Invalid/Unlinked Prereqs  : {len(report.invalid_links)}",
        f"Human Review Flags (LLM)  : {len(report.human_review_flags)}",
        "---------------------------------------------------------------",
        f"Blocking Errors           : {len(report.errors)}",
        f"Non-Blocking Warnings     : {len(report.warnings)}",
    ]

    if report.errors:
        lines.append("\n--- BLOCKING ERRORS (Records Rejected from Knowledge Base) ---")
        for err in report.errors:
            lines.append(f"  [ERROR | {err.category}] {err.file_name} ({err.record_id}): {err.message}")

    if report.warnings:
        lines.append("\n--- WARNINGS & HUMAN REVIEW FLAGS ---")
        for warn in report.warnings:
            lines.append(f"  [WARN  | {warn.category}] {warn.file_name} ({warn.record_id}): {warn.message}")

    return "\n".join(lines)


# ============================================================================
# Automated Unit Tests
# ============================================================================

class TestValidateLinks(unittest.TestCase):
    """Automated tests verifying valid data, invalid metadata, duplicates, empty content, and Unicode preservation."""

    def _make_valid_record(
        self,
        rec_id: str = "c10_geography_ch03_s001",
        class_level: int = 10,
        subject: str = "Geography",
        chapter_id: str = "ch03",
        chapter_num: int = 3,
        language: str = "or",
        text: str = "ଜଳ ସମ୍ପଦ (Water Resources): ଓଡ଼ିଶାର ହୀରାକୁଦ ନଦୀବନ୍ଧ ଯୋଜନା ମହାନଦୀ ଉପରେ ଅବସ୍ଥିତ।",
        concept_tag: str = "ch03_s001_water_resources",
        prerequisite_concept: str = "none",
        diagnostic_q: str = "",
    ) -> Dict[str, Any]:
        return {
            "id": rec_id,
            "text": text,
            "metadata": {
                "class_level": class_level,
                "subject": subject,
                "chapter_id": chapter_id,
                "chapter_num": chapter_num,
                "chapter_name": "Water Resources",
                "language": language,
                "source_name": "BSE Odisha Class 10 Geography",
                "page_numbers": "25-38",
                "section_id": "s001",
                "section_heading": "Introduction",
                "source_file": "data/source_text/10th Water resources.txt",
                "concept_tag": concept_tag,
                "prerequisite_concept": prerequisite_concept,
                "english_tech_terms": "water resources",
                "diagnostic_question_odia": diagnostic_q,
            },
        }

    def test_1_valid_data_and_odia_unicode_preservation(self) -> None:
        """Verifies valid Odia and English records pass and their text/provenance remain untouched."""
        original_odia = "ନିଉଟନ୍‌ଙ୍କ ଦ୍ୱିତୀୟ ଗତି ନିୟମ (Newton's Second Law): F = m × a କିମ୍ବା F = dp/dt।"
        rec_c9 = self._make_valid_record(
            rec_id="c9_physics_ch01_s001",
            class_level=9,
            subject="Physics",
            chapter_id="ch01",
            chapter_num=1,
            language="or",
            text=original_odia,
            concept_tag="newtons_second_law_c9",
        )
        rec_c12 = self._make_valid_record(
            rec_id="c12_physics_ch03_s001",
            class_level=12,
            subject="Physics",
            chapter_id="ch03",
            chapter_num=3,
            language="en",
            text="Momentum conservation and impulse in classical mechanics build upon F = dp/dt.",
            concept_tag="impulse_momentum_c12",
            prerequisite_concept="newtons_second_law_c9",
        )

        report = validate_records_in_memory([("physics.json", [rec_c9, rec_c12])])
        self.assertEqual(report.approved_count, 2)
        self.assertEqual(report.rejected_count, 0)
        self.assertEqual(len(report.errors), 0)
        # Ensure exact string and ZWNJ preservation without silent rewriting
        self.assertEqual(report.approved_records[0]["text"], original_odia)
        self.assertEqual(
            report.approved_records[0]["metadata"]["source_file"],
            "data/source_text/10th Water resources.txt",
        )

    def test_2_invalid_and_missing_metadata_rejected(self) -> None:
        """Verifies invalid class_level, missing subject, and ID/class mismatch are rejected."""
        bad_class = self._make_valid_record(rec_id="c11_physics_ch01_s001", class_level=11)
        missing_subj = self._make_valid_record(rec_id="c10_geography_ch03_s002", subject="   ")
        id_mismatch = self._make_valid_record(
            rec_id="c9_geography_ch03_s003", class_level=10, concept_tag="ch03_s003_tag"
        )

        report = validate_records_in_memory(
            [("bad_meta.json", [bad_class, missing_subj, id_mismatch])]
        )
        self.assertEqual(report.approved_count, 0)
        self.assertEqual(report.rejected_count, 3)
        self.assertGreaterEqual(len(report.missing_metadata_issues), 3)

    def test_3_duplicate_identifiers_rejected(self) -> None:
        """Verifies duplicate record IDs and duplicate concept_tags within a chapter are rejected."""
        rec1 = self._make_valid_record(rec_id="c10_geography_ch03_s001", concept_tag="water_cycle")
        rec2_dup_id = self._make_valid_record(rec_id="c10_geography_ch03_s001", concept_tag="dams_odisha")
        rec3_dup_tag = self._make_valid_record(rec_id="c10_geography_ch03_s002", concept_tag="water_cycle")

        report = validate_records_in_memory([("dups.json", [rec1, rec2_dup_id, rec3_dup_tag])])
        self.assertEqual(report.approved_count, 1)
        self.assertEqual(report.rejected_count, 2)
        self.assertEqual(len(report.duplicate_identifiers), 2)

    def test_4_empty_content_and_unicode_corruption_rejected(self) -> None:
        """Verifies empty strings, replacement chars (U+FFFD), mojibake, and script mismatch are rejected."""
        empty_rec = self._make_valid_record(
            rec_id="c10_geography_ch03_s001", text="   \n\t  ", concept_tag="tag_1"
        )
        corrupt_rec = self._make_valid_record(
            rec_id="c10_geography_ch03_s002",
            text="ଓଡ଼ିଶାର ଜଳ \ufffd ସମ୍ପଦ ବିଷୟରେ ଆଲୋଚନା।",
            concept_tag="tag_2",
        )
        mojibake_rec = self._make_valid_record(
            rec_id="c10_geography_ch03_s003",
            text="à¬“à¬¡à¬¼à¬¿à¬¶à¬¾à¬° à¬œà¬³ à¬¸à¬®à­ à¬ªà¬¦",
            concept_tag="tag_3",
        )
        wrong_lang_rec = self._make_valid_record(
            rec_id="c10_geography_ch03_s004",
            language="or",
            text="This text has no Odia characters despite declaring language='or'.",
            concept_tag="tag_4",
        )

        report = validate_records_in_memory(
            [("corrupt.json", [empty_rec, corrupt_rec, mojibake_rec, wrong_lang_rec])]
        )
        self.assertEqual(report.approved_count, 0)
        self.assertEqual(report.rejected_count, 4)
        self.assertGreaterEqual(len(report.empty_or_suspicious_issues), 4)

    def test_5_prerequisite_graph_and_llm_human_review_flags(self) -> None:
        """Verifies self-links, cycles, and grade inversions fail, and LLM fields trigger human review warnings."""
        # Record with LLM diagnostic question -> Approved, but flagged for human review
        llm_rec = self._make_valid_record(
            rec_id="c10_geography_ch03_s001",
            concept_tag="rainwater_harvesting",
            diagnostic_q="ବର୍ଷା ଜଳ ସଂରକ୍ଷଣ କ'ଣ?",
        )
        # Class 9 concept requiring a Class 12 prerequisite -> Rejected (grade inversion)
        c12_rec = self._make_valid_record(
            rec_id="c12_physics_ch01_s001",
            class_level=12,
            subject="Physics",
            chapter_id="ch01",
            language="en",
            text="Advanced electrostatics and Gauss's Law in integral form.",
            concept_tag="gauss_law_c12",
        )
        c9_bad_prereq = self._make_valid_record(
            rec_id="c9_physics_ch01_s001",
            class_level=9,
            subject="Physics",
            chapter_id="ch01",
            language="en",
            text="Basic introduction to electric charges in Class 9.",
            concept_tag="charge_basics_c9",
            prerequisite_concept="gauss_law_c12",
        )

        report = validate_records_in_memory([("graph.json", [llm_rec, c12_rec, c9_bad_prereq])])
        self.assertEqual(report.approved_count, 2)  # llm_rec and c12_rec approved
        self.assertEqual(report.rejected_count, 1)  # c9_bad_prereq rejected
        self.assertEqual(len(report.invalid_links), 1)
        self.assertEqual(len(report.human_review_flags), 1)


# ============================================================================
# Command-Line Interface
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nidana Stage 2: Validate textbook records and prerequisite graph before DB build."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input directory containing structured JSON records (default: data/processed or data/extracted_json).",
    )
    parser.add_argument(
        "--strict-links",
        action="store_true",
        help="Treat unresolved external prerequisite_concept tags as blocking errors instead of warnings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print report without syncing approved records to data/extracted_json/.",
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
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestValidateLinks)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    target_dir = resolve_input_directory(args.input)
    print(f"Inspecting records in: {target_dir}")

    report = run_validation_stage(
        input_dir=target_dir,
        approved_output_dir=EXTRACTED_JSON_DIR,
        strict_links=args.strict_links,
        write_approved=not args.dry_run,
    )

    print("\n" + format_validation_report(report))

    if report.approved_count == 0:
        print("\n[FAILED] 0 approved records available for database construction.", file=sys.stderr)
        return 1

    if report.rejected_count > 0:
        print(
            f"\n[ATTENTION] {report.rejected_count} record(s) failed validation and were excluded from the approved set.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
    