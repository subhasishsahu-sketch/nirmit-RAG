#!/usr/bin/env python3
"""
PROJECT: Nidana
FILE: import_textbook.py

Plain-text textbook ingestion module for Odisha State Board (Classes 9, 10, 12)
Odia ("or") and English ("en") chapters.

Produces validated canonical JSON records in `data/processed/` (and optionally
`data/extracted_json/`) without calling PDF/OCR, vision models, or embeddings.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from pydantic import BaseModel, Field, ValidationError, field_validator


# ============================================================================
# Canonical Paths & Constants
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "source_text"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted_json"
EVALUATION_DIR = PROJECT_ROOT / "data" / "evaluation"

ALLOWED_CLASSES = {9, 10, 12}
ALLOWED_LANGUAGES = {"or", "en"}

REQUIRED_METADATA_KEYS = {
    "class_level",
    "subject",
    "chapter_name",
    "chapter_id",
    "language",
    "source_name",
}

# Key aliases mapped to canonical metadata field names (never inferred silently)
METADATA_KEY_ALIASES = {
    "class": "class_level",
    "class_level": "class_level",
    "subject": "subject",
    "chapter_name": "chapter_name",
    "chapter_title": "chapter_name",
    "chapter_id": "chapter_id",
    "chapter_num": "chapter_num",
    "language": "language",
    "lang": "language",
    "source_name": "source_name",
    "textbook": "source_name",
    "source": "source_name",
    "page_numbers": "page_numbers",
    "pages": "page_numbers",
    "prerequisite_concept": "prerequisite_concept",
}

# Headings that denote end-of-chapter evaluation questions (excluded from KB)
EVALUATION_SECTION_PATTERNS = [
    re.compile(r"^\s*#{1,4}\s*(evaluation|exercises|test\s+questions|अभ्यास|ଅଭ୍ୟାସ|ପ୍ରଶ୍ନାବଳୀ)\b", re.IGNORECASE),
    re.compile(r"^\s*\[EVALUATION_SECTION\]\s*$", re.IGNORECASE),
]


# ============================================================================
# Exceptions
# ============================================================================

class IngestionError(Exception):
    """Base exception for textbook ingestion errors."""


class InvalidUTF8Error(IngestionError):
    """Raised when a file contains invalid UTF-8 byte sequences."""


class EmptyContentError(IngestionError):
    """Raised when a file or textbook body is empty."""


class MetadataValidationError(IngestionError):
    """Raised when required metadata is missing or malformed."""


class DuplicateIdentifierError(IngestionError):
    """Raised when two chapters or sections produce colliding IDs."""


class UnsupportedFormatError(IngestionError):
    """Raised when a non-.txt file is encountered."""


class EvaluationContentExcludedError(IngestionError):
    """Raised when an evaluation question file is targeted for KB ingestion."""


# ============================================================================
# Audited Canonical Schema (Compatible with 2_validate_links & 3_build_databases)
# ============================================================================

class ConceptMetadata(BaseModel):
    """Validated metadata schema aligned with the Nidana project audit."""
    class_level: int
    subject: str
    chapter_id: str
    chapter_num: int
    chapter_name: str
    language: str
    source_name: str
    page_numbers: str = Field(default="")
    section_id: str
    section_heading: str = Field(default="Introduction")
    source_file: str
    concept_tag: str
    prerequisite_concept: str = Field(default="none")
    english_tech_terms: str = Field(default="")
    diagnostic_question_odia: str = Field(default="")

    @field_validator("class_level")
    @classmethod
    def validate_class_level(cls, v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int) or v not in ALLOWED_CLASSES:
            raise ValueError(f"class_level must be one of {sorted(ALLOWED_CLASSES)}, got {v!r}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        cleaned = str(v).strip().lower()
        if cleaned not in ALLOWED_LANGUAGES:
            raise ValueError(f"language must be one of {sorted(ALLOWED_LANGUAGES)}, got {v!r}")
        return cleaned

    @field_validator("subject", "chapter_id", "chapter_name", "source_name", "section_id", "concept_tag")
    @classmethod
    def validate_non_empty_string(cls, v: str, info: Any) -> str:
        cleaned = str(v).strip()
        if not cleaned:
            raise ValueError(f"Field '{info.field_name}' cannot be empty")
        return cleaned


class CanonicalConceptRecord(BaseModel):
    """Canonical record consumed by 2_validate_links.py and 3_build_databases.py."""
    id: str
    text: str
    metadata: ConceptMetadata


@dataclass
class FileIngestionResult:
    file_path: Path
    chapter_source_id: str
    records: List[CanonicalConceptRecord]
    char_count: int
    skipped_eval_sections: int = 0


@dataclass
class BatchIngestionReport:
    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    total_records_created: int = 0
    total_characters_processed: int = 0
    skipped_eval_sections: int = 0
    output_files_written: List[Path] = field(default_factory=list)
    errors: List[Tuple[Path, str]] = field(default_factory=list)


# ============================================================================
# Text Cleaning & Preservation Utilities
# ============================================================================

def read_utf8_strict(file_path: Path) -> str:
    """
    Reads a file strictly as UTF-8. Rejects invalid byte sequences and empty files.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise IngestionError(f"Path is not a regular file: {file_path}")

    raw_bytes = file_path.read_bytes()
    if len(raw_bytes) == 0:
        raise EmptyContentError(f"File is 0 bytes: {file_path}")

    try:
        text = raw_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise InvalidUTF8Error(
            f"Invalid UTF-8 encoding in {file_path.name} at byte {exc.start}: {exc.reason}"
        ) from exc

    return text


def sanitize_formatting_artifacts(raw_text: str) -> str:
    """
    Removes only clearly unnecessary formatting artifacts while strictly preserving:
    - Odia Unicode characters (0B00-0B7F) and joiners (ZWNJ \u200c, ZWJ \u200d)
    - Mathematical symbols, formulas, subscripts/superscripts, Greek letters
    - Numbered lists, indentation, and meaningful punctuation
    - Original wording (never rewrites textbook prose)
    """
    # 1. Strip UTF-8 BOM if present at start of file
    text = raw_text.lstrip("\ufeff")

    # 2. Standardize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Apply NFC normalization (canonical composition for Odia ligatures/matras)
    text = unicodedata.normalize("NFC", text)

    # 4. Remove non-printable ASCII control characters EXCEPT \n (\x0a) and \t (\x09).
    #    Preserves \u200c (ZWNJ) and \u200d (ZWJ) required for Odia juktakshara.
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 5. Strip trailing horizontal whitespace per line while preserving leading indentation
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    text = "\n".join(lines)

    # 6. Collapse 3+ consecutive blank lines into at most 2 blank lines (1 empty line between blocks)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def make_slug(value: str) -> str:
    """Creates an ASCII-safe deterministic slug; falls back to short hash for pure Odia strings."""
    cleaned = value.strip().lower()
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if ascii_slug:
        return ascii_slug[:40]
    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    return f"id_{digest}"


def extract_parenthetical_english_terms(text: str) -> str:
    """
    Non-destructively extracts English technical terms appearing in parentheses
    (e.g., 'ଜଡ଼ତା (Inertia)') for metadata indexing without modifying the text.
    """
    matches = re.findall(r"\(([A-Za-z][A-Za-z0-9\s,\-/]{1,60})\)", text)
    seen: List[str] = []
    for m in matches:
        term = m.strip().lower()
        if len(term) >= 2 and term not in seen:
            seen.append(term)
    return ", ".join(seen)


# ============================================================================
# Metadata Parsing (In-File Header & Sidecar .meta.json)
# ============================================================================

def parse_header_and_body(full_text: str, txt_path: Path) -> Tuple[Dict[str, Any], str]:
    """
    Extracts metadata from a sidecar `<stem>.meta.json` and/or an explicit header
    block at the top of the `.txt` file. Never infers missing metadata silently.

    Supported In-File Header Formats:
    Format A (Delimited frontmatter):
        ---
        CLASS: 9
        SUBJECT: Physics
        CHAPTER_ID: ch01
        CHAPTER_NAME: ଗତି (Motion)
        LANGUAGE: or
        SOURCE_NAME: BSE Odisha Class 9 Vigyan
        PAGE_NUMBERS: 1-14
        ---
        <Body text...>

    Format B (Top key-value block followed by a blank line):
        CLASS: 9
        SUBJECT: Physics
        CHAPTER_ID: ch01
        CHAPTER_NAME: ଗତି (Motion)
        LANGUAGE: or
        SOURCE_NAME: BSE Odisha Class 9 Vigyan

        <Body text...>
    """
    metadata: Dict[str, Any] = {}

    # 1. Check for optional sidecar file: `<filename>.meta.json`
    sidecar_path = txt_path.with_suffix(".meta.json")
    if sidecar_path.exists():
        try:
            sidecar_raw = read_utf8_strict(sidecar_path)
            sidecar_data = json.loads(sidecar_raw)
            if not isinstance(sidecar_data, dict):
                raise MetadataValidationError(
                    f"Sidecar metadata in {sidecar_path.name} must be a JSON object."
                )
            for k, v in sidecar_data.items():
                norm_key = METADATA_KEY_ALIASES.get(str(k).strip().lower())
                if norm_key:
                    metadata[norm_key] = v
        except json.JSONDecodeError as exc:
            raise MetadataValidationError(
                f"Malformed JSON in sidecar {sidecar_path.name}: {exc}"
            ) from exc

    # 2. Check for in-file header block
    lines = full_text.split("\n")
    body_start_idx = 0

    if lines and lines[0].strip() == "---":
        # Delimited frontmatter block (`---` ... `---`)
        closing_idx = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() in ("---", "==="):
                closing_idx = idx
                break
        if closing_idx is None:
            raise MetadataValidationError(
                f"Unclosed frontmatter block ('---' without closing '---') in {txt_path.name}"
            )
        header_lines = lines[1:closing_idx]
        body_start_idx = closing_idx + 1
        _parse_kv_lines(header_lines, metadata, txt_path.name)
    else:
        # Check if top lines are explicit `KEY: VALUE` pairs before the first blank line
        header_lines = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                body_start_idx = idx + 1
                break
            if ":" in stripped:
                key_part = stripped.split(":", 1)[0].strip().lower()
                if key_part in METADATA_KEY_ALIASES:
                    header_lines.append(stripped)
                    body_start_idx = idx + 1
                    continue
            # First line is not a recognized metadata header -> entire file is body
            if idx == 0:
                header_lines = []
                body_start_idx = 0
                break
            else:
                raise MetadataValidationError(
                    f"Malformed metadata line {idx + 1} in {txt_path.name}: {line!r}"
                )
        if header_lines:
            _parse_kv_lines(header_lines, metadata, txt_path.name)

    body_text = "\n".join(lines[body_start_idx:])
    cleaned_body = sanitize_formatting_artifacts(body_text)

    if not cleaned_body:
        raise EmptyContentError(
            f"Textbook chapter body is empty after metadata extraction in {txt_path.name}"
        )

    validated_meta = _validate_required_chapter_metadata(metadata, txt_path.name)
    return validated_meta, cleaned_body


def _parse_kv_lines(header_lines: List[str], target_meta: Dict[str, Any], filename: str) -> None:
    for line in header_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            raise MetadataValidationError(
                f"Malformed header line (missing ':') in {filename}: {line!r}"
            )
        raw_key, raw_val = stripped.split(":", 1)
        norm_key = METADATA_KEY_ALIASES.get(raw_key.strip().lower())
        if not norm_key:
            raise MetadataValidationError(
                f"Unknown metadata key {raw_key.strip()!r} in {filename}. "
                f"Supported keys: {sorted(METADATA_KEY_ALIASES.keys())}"
            )
        val_str = raw_val.strip()
        if not val_str:
            raise MetadataValidationError(
                f"Empty value for metadata key {raw_key.strip()!r} in {filename}"
            )
        target_meta[norm_key] = val_str


def _validate_required_chapter_metadata(meta: Dict[str, Any], filename: str) -> Dict[str, Any]:
    missing = sorted(REQUIRED_METADATA_KEYS - set(meta.keys()))
    if missing:
        raise MetadataValidationError(
            f"Missing required metadata in {filename}: {', '.join(missing)}. "
            "All metadata must be explicitly supplied via file header or .meta.json sidecar."
        )

    # Parse class_level strictly as integer 9, 10, or 12
    raw_class = meta["class_level"]
    try:
        if isinstance(raw_class, bool):
            raise ValueError("Boolean is not a valid class level")
        class_int = int(str(raw_class).strip())
    except ValueError as exc:
        raise MetadataValidationError(
            f"Invalid class_level {raw_class!r} in {filename}; must be 9, 10, or 12."
        ) from exc

    if class_int not in ALLOWED_CLASSES:
        raise MetadataValidationError(
            f"Unsupported class_level {class_int} in {filename}; allowed values are 9, 10, 12."
        )

    # Validate language strictly as 'or' or 'en'
    lang = str(meta["language"]).strip().lower()
    if lang not in ALLOWED_LANGUAGES:
        raise MetadataValidationError(
            f"Invalid language {meta['language']!r} in {filename}; allowed values are 'or' or 'en'."
        )

    chapter_id = str(meta["chapter_id"]).strip()
    if not chapter_id:
        raise MetadataValidationError(f"chapter_id cannot be empty in {filename}")

    # Determine integer chapter_num from explicit field or digits in chapter_id
    if "chapter_num" in meta and str(meta["chapter_num"]).strip():
        try:
            chapter_num = int(str(meta["chapter_num"]).strip())
        except ValueError as exc:
            raise MetadataValidationError(
                f"Invalid chapter_num {meta['chapter_num']!r} in {filename}; must be an integer."
            ) from exc
    else:
        digit_match = re.search(r"\d+", chapter_id)
        chapter_num = int(digit_match.group(0)) if digit_match else 1

    subject = str(meta["subject"]).strip()
    chapter_name = str(meta["chapter_name"]).strip()
    source_name = str(meta["source_name"]).strip()
    page_numbers = str(meta.get("page_numbers", "")).strip()
    prerequisite_concept = str(meta.get("prerequisite_concept", "none")).strip() or "none"

    if not subject or not chapter_name or not source_name:
        raise MetadataValidationError(
            f"subject, chapter_name, and source_name must be non-empty in {filename}"
        )

    return {
        "class_level": class_int,
        "subject": subject,
        "chapter_id": chapter_id,
        "chapter_num": chapter_num,
        "chapter_name": chapter_name,
        "language": lang,
        "source_name": source_name,
        "page_numbers": page_numbers,
        "prerequisite_concept": prerequisite_concept,
    }


# ============================================================================
# Section Splitting & Deterministic ID Generation
# ============================================================================

def split_into_sections(body_text: str) -> Tuple[List[Tuple[str, str]], int]:
    """
    Splits chapter text into sections using explicit Markdown headings (`#`, `##`, `###`)
    or `[SECTION: Title]` markers while preserving every character of prose, math
    expressions, and numbered lists inside each section.

    If a section heading matches `EVALUATION_SECTION_PATTERNS` (e.g., end-of-chapter
    test questions), that section is excluded from the knowledge base and counted.
    """
    lines = body_text.split("\n")
    sections: List[Tuple[str, List[str]]] = []
    current_heading = "Introduction"
    current_lines: List[str] = []
    is_current_eval = False
    skipped_eval_count = 0

    heading_regex = re.compile(r"^\s*(?:#{1,3}\s+(.+?)|\[SECTION:\s*(.+?)\])\s*$")

    def flush_section() -> None:
        nonlocal skipped_eval_count
        content = "\n".join(current_lines).strip()
        if not content:
            return
        if is_current_eval:
            skipped_eval_count += 1
            return
        sections.append((current_heading, current_lines.copy()))

    for line in lines:
        # Check if this line starts an evaluation/exercise block
        if any(pat.match(line) for pat in EVALUATION_SECTION_PATTERNS):
            flush_section()
            current_heading = line.strip("# []")
            current_lines = []
            is_current_eval = True
            continue

        match = heading_regex.match(line)
        if match:
            flush_section()
            current_heading = (match.group(1) or match.group(2)).strip()
            current_lines = [line]  # Preserve heading inside the original source text
            is_current_eval = False
        else:
            current_lines.append(line)

    flush_section()
    return [(h, "\n".join(lns).strip()) for h, lns in sections], skipped_eval_count


def enrich_with_ollama_optional(
    section_text: str,
    chapter_meta: Dict[str, Any],
    ollama_url: str,
    ollama_model: str,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Optional helper invoked ONLY when `--extract-concepts` is explicitly passed.
    Never modifies `section_text`. Extracts a concept_tag and diagnostic question.
    """
    prompt = (
        "Read the following textbook section and return ONLY a JSON object with keys "
        "'concept_tag' (snake_case string) and 'diagnostic_question_odia' (1 question in Odia).\n\n"
        f"TEXT:\n{section_text[:1500]}"
    )
    payload = json.dumps({
        "model": ollama_model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")

    req = urllib_request.Request(
        f"{ollama_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            parsed = json.loads(body.get("response", "{}"))
            return parsed.get("concept_tag"), parsed.get("diagnostic_question_odia")
    except (urllib_error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None, None


def process_textbook_file(
    txt_path: Path,
    extract_concepts: bool = False,
    ollama_url: str = "http://localhost:11434",
    ollama_model: str = "qwen2.5:3b",
) -> FileIngestionResult:
    """
    Reads, validates, and structures a single `.txt` textbook chapter into
    validated CanonicalConceptRecord instances.
    """
    # Guard against importing evaluation question files into the KB
    resolved_parts = [p.lower() for p in txt_path.resolve().parts]
    if "evaluation" in resolved_parts or "test_questions" in txt_path.stem.lower():
        raise EvaluationContentExcludedError(
            f"Refusing to import evaluation file into knowledge base: {txt_path}"
        )

    if txt_path.suffix.lower() != ".txt":
        raise UnsupportedFormatError(
            f"Unsupported file format '{txt_path.suffix}' for {txt_path.name}. Only UTF-8 .txt files are accepted."
        )

    raw_text = read_utf8_strict(txt_path)
    if not raw_text.strip():
        raise EmptyContentError(f"File contains only whitespace: {txt_path.name}")

    chapter_meta, cleaned_body = parse_header_and_body(raw_text, txt_path)
    sections, skipped_eval = split_into_sections(cleaned_body)

    if not sections:
        raise EmptyContentError(
            f"No importable textbook sections found in {txt_path.name} (skipped {skipped_eval} evaluation sections)."
        )

    class_num = chapter_meta["class_level"]
    subject_slug = make_slug(chapter_meta["subject"])
    chapter_slug = make_slug(chapter_meta["chapter_id"])
    chapter_source_id = f"c{class_num}_{subject_slug}_{chapter_slug}"

    try:
        rel_source = str(txt_path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        rel_source = txt_path.name

    records: List[CanonicalConceptRecord] = []
    seen_doc_ids: Set[str] = set()

    for idx, (heading, section_text) in enumerate(sections, start=1):
        section_id = f"s{idx:03d}"
        doc_id = f"{chapter_source_id}_{section_id}"
        if doc_id in seen_doc_ids:
            raise DuplicateIdentifierError(f"Duplicate document ID generated: {doc_id}")
        seen_doc_ids.add(doc_id)

        heading_slug = make_slug(heading)
        default_concept_tag = f"{chapter_slug}_{section_id}_{heading_slug}"[:60]
        diagnostic_q = ""

        if extract_concepts:
            llm_tag, llm_q = enrich_with_ollama_optional(
                section_text, chapter_meta, ollama_url, ollama_model
            )
            if llm_tag:
                default_concept_tag = make_slug(llm_tag)
            if llm_q:
                diagnostic_q = sanitize_formatting_artifacts(llm_q)

        tech_terms = extract_parenthetical_english_terms(section_text)

        try:
            record = CanonicalConceptRecord(
                id=doc_id,
                text=section_text,
                metadata=ConceptMetadata(
                    class_level=class_num,
                    subject=chapter_meta["subject"],
                    chapter_id=chapter_meta["chapter_id"],
                    chapter_num=chapter_meta["chapter_num"],
                    chapter_name=chapter_meta["chapter_name"],
                    language=chapter_meta["language"],
                    source_name=chapter_meta["source_name"],
                    page_numbers=chapter_meta["page_numbers"],
                    section_id=section_id,
                    section_heading=heading,
                    source_file=rel_source,
                    concept_tag=default_concept_tag,
                    prerequisite_concept=chapter_meta["prerequisite_concept"] if idx == 1 else "none",
                    english_tech_terms=tech_terms,
                    diagnostic_question_odia=diagnostic_q,
                ),
            )
        except ValidationError as exc:
            raise MetadataValidationError(
                f"Schema validation failed for {txt_path.name} ({section_id}): {exc}"
            ) from exc

        records.append(record)

    total_chars = sum(len(r.text) for r in records)
    return FileIngestionResult(
        file_path=txt_path,
        chapter_source_id=chapter_source_id,
        records=records,
        char_count=total_chars,
        skipped_eval_sections=skipped_eval,
    )


# ============================================================================
# Batch Directory Ingestion Orchestrator
# ============================================================================

def import_textbooks_from_path(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sync_extracted_dir: Optional[Path] = DEFAULT_EXTRACTED_DIR,
    dry_run: bool = False,
    extract_concepts: bool = False,
    strict: bool = False,
) -> BatchIngestionReport:
    """
    Discovers and processes textbook files under `input_path` (supporting
    nested class/subject subdirectories). Enforces global duplicate source ID
    detection across all processed files.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    report = BatchIngestionReport()
    candidate_files: List[Path] = []

    if input_path.is_file():
        candidate_files = [input_path]
    else:
        for p in sorted(input_path.rglob("*")):
            if not p.is_file():
                continue
            # Ignore sidecar metadata files (.meta.json) and hidden system files (.DS_Store, .gitkeep)
            if p.name.endswith(".meta.json") or p.name.startswith("."):
                continue
            candidate_files.append(p)

    report.files_discovered = len(candidate_files)
    seen_chapter_ids: Dict[str, Path] = {}
    seen_record_ids: Dict[str, Path] = {}
    successful_results: List[FileIngestionResult] = []

    for file_path in candidate_files:
        try:
            result = process_textbook_file(
                file_path,
                extract_concepts=extract_concepts,
            )

            # Check for duplicate chapter source IDs across different files
            if result.chapter_source_id in seen_chapter_ids:
                prev_file = seen_chapter_ids[result.chapter_source_id]
                raise DuplicateIdentifierError(
                    f"Duplicate chapter source identifier '{result.chapter_source_id}' "
                    f"in {file_path.name} (already defined in {prev_file.name})."
                )

            for rec in result.records:
                if rec.id in seen_record_ids:
                    prev_file = seen_record_ids[rec.id]
                    raise DuplicateIdentifierError(
                        f"Duplicate document ID '{rec.id}' in {file_path.name} "
                        f"(already defined in {prev_file.name})."
                    )

            seen_chapter_ids[result.chapter_source_id] = file_path
            for rec in result.records:
                seen_record_ids[rec.id] = file_path

            successful_results.append(result)
            report.files_processed += 1
            report.total_records_created += len(result.records)
            report.total_characters_processed += result.char_count
            report.skipped_eval_sections += result.skipped_eval_sections

        except IngestionError as exc:
            report.files_failed += 1
            report.errors.append((file_path, str(exc)))
            if strict:
                raise

    if not dry_run and successful_results:
        output_dir.mkdir(parents=True, exist_ok=True)
        if sync_extracted_dir is not None:
            sync_extracted_dir.mkdir(parents=True, exist_ok=True)

        for res in successful_results:
            serialized = [r.model_dump() for r in res.records]
            out_file = output_dir / f"{res.chapter_source_id}.json"
            out_file.write_text(
                json.dumps(serialized, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            report.output_files_written.append(out_file)

            if sync_extracted_dir is not None:
                ext_file = sync_extracted_dir / f"{res.chapter_source_id}.json"
                ext_file.write_text(
                    json.dumps(serialized, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    return report


# ============================================================================
# Unit Tests (Odia Unicode, Metadata, Empty Files, Duplicate IDs, Formats)
# ============================================================================

class TestImportTextbook(unittest.TestCase):
    """Built-in unit test suite verifying all ingestion requirements."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.src_dir = self.root / "source_text"
        self.out_dir = self.root / "processed"
        self.src_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_odia_unicode_and_formula_preservation(self) -> None:
        """Verifies Odia script, juktakshara joiners, math formulas, and numbered lists are preserved verbatim."""
        odia_body = (
            "## ୧.୧ ଗତି ଏବଂ ବିଶ୍ରାମ (Motion and Rest)\n"
            "ନିଉଟନ୍‌ଙ୍କ ଦ୍ୱିତୀୟ ଗତି ନିୟମ (Newton's Second Law): F = m × a କିମ୍ବା F = dp/dt।\n"
            "1. ଦୂରତା (Distance) ଏକ ଅଦିଶ ରାଶି (Scalar Quantity) ଅଟେ।\n"
            "2. ଗତିଶୀଳ ବସ୍ତୁର ଗତିଜ ଶକ୍ତି: E_k = (1/2)mv² ଏବଂ মহাকর্ষ ବଳ F = G(m₁m₂)/r²।"
        )
        content = (
            "---\n"
            "CLASS: 9\n"
            "SUBJECT: Physics\n"
            "CHAPTER_ID: ch01\n"
            "CHAPTER_NAME: ଗତି (Motion)\n"
            "LANGUAGE: or\n"
            "SOURCE_NAME: BSE Odisha Vigyan Class 9\n"
            "PAGE_NUMBERS: 1-12\n"
            "---\n\n"
            f"{odia_body}\n"
        )
        txt_file = self.src_dir / "c9_motion.txt"
        txt_file.write_text(content, encoding="utf-8")

        res = process_textbook_file(txt_file)
        self.assertEqual(len(res.records), 1)
        rec = res.records[0]

        # Exact text preservation check (no silent rewriting)
        self.assertEqual(rec.text, odia_body)
        self.assertIn("ନିଉଟନ୍‌ଙ୍କ", rec.text)  # Preserves ZWNJ (\u200c)
        self.assertIn("E_k = (1/2)mv²", rec.text)
        self.assertEqual(rec.metadata.class_level, 9)
        self.assertEqual(rec.metadata.language, "or")
        self.assertEqual(rec.metadata.page_numbers, "1-12")
        self.assertIn("newton's second law", rec.metadata.english_tech_terms)

    def test_sidecar_metadata_and_missing_metadata_rejection(self) -> None:
        """Verifies missing metadata raises MetadataValidationError and sidecar .meta.json works."""
        unannotated_file = self.src_dir / "no_meta.txt"
        unannotated_file.write_text("Some textbook paragraph without metadata.", encoding="utf-8")

        with self.assertRaises(MetadataValidationError):
            process_textbook_file(unannotated_file)

        # Now provide a valid sidecar .meta.json file
        sidecar = self.src_dir / "no_meta.meta.json"
        sidecar.write_text(
            json.dumps({
                "class": 12,
                "subject": "Physics",
                "chapter_id": "ch03",
                "chapter_name": "Current Electricity",
                "language": "en",
                "source_name": "CHSE Odisha Physics Class 12",
                "page_numbers": "88-104",
            }),
            encoding="utf-8",
        )
        res = process_textbook_file(unannotated_file)
        self.assertEqual(res.records[0].metadata.class_level, 12)
        self.assertEqual(res.records[0].metadata.language, "en")
        self.assertEqual(res.records[0].id, "c12_physics_ch03_s001")

    def test_empty_file_and_whitespace_only_rejection(self) -> None:
        """Verifies 0-byte files and header-only files with empty bodies are rejected."""
        empty_file = self.src_dir / "empty.txt"
        empty_file.write_bytes(b"")
        with self.assertRaises(EmptyContentError):
            process_textbook_file(empty_file)

        header_only = self.src_dir / "header_only.txt"
        header_only.write_text(
            "CLASS: 10\nSUBJECT: Science\nCHAPTER_ID: ch02\n"
            "CHAPTER_NAME: Acids\nLANGUAGE: en\nSOURCE_NAME: BSE\n\n   \n",
            encoding="utf-8",
        )
        with self.assertRaises(EmptyContentError):
            process_textbook_file(header_only)

    def test_duplicate_source_identifier_detection(self) -> None:
        """Verifies duplicate chapter identifiers across two files raise DuplicateIdentifierError."""
        sample_content = (
            "CLASS: 10\nSUBJECT: Physics\nCHAPTER_ID: ch01\n"
            "CHAPTER_NAME: Electricity\nLANGUAGE: en\nSOURCE_NAME: BSE Odisha\n\n"
            "Ohm's Law states that V = IR at constant temperature."
        )
        sub1 = self.src_dir / "folder_a"
        sub2 = self.src_dir / "folder_b"
        sub1.mkdir()
        sub2.mkdir()
        (sub1 / "elec_1.txt").write_text(sample_content, encoding="utf-8")
        (sub2 / "elec_duplicate.txt").write_text(sample_content, encoding="utf-8")

        with self.assertRaises(DuplicateIdentifierError):
            import_textbooks_from_path(self.src_dir, self.out_dir, sync_extracted_dir=None, strict=True)

    def test_invalid_utf8_and_evaluation_exclusion(self) -> None:
        """Verifies invalid UTF-8 bytes are rejected and evaluation sections are stripped."""
        bad_utf8 = self.src_dir / "corrupt.txt"
        bad_utf8.write_bytes(b"\xff\xfeInvalid UTF-8 sequence")
        with self.assertRaises(InvalidUTF8Error):
            process_textbook_file(bad_utf8)

        chapter_with_eval = self.src_dir / "with_eval.txt"
        chapter_with_eval.write_text(
            "CLASS: 9\nSUBJECT: Science\nCHAPTER_ID: ch04\n"
            "CHAPTER_NAME: Atoms\nLANGUAGE: or\nSOURCE_NAME: BSE\n\n"
            "## ୪.୧ ପରମାଣୁ ଗଠନ (Structure of Atom)\n"
            "ପରମାଣୁରେ ପ୍ରୋଟନ୍, ନ୍ୟୁଟ୍ରନ୍ ଓ ଇଲେକ୍ଟ୍ରନ୍ ଥାଏ।\n\n"
            "## ଅଭ୍ୟାସ (Exercises)\n"
            "1. ପରମାଣୁର କେନ୍ଦ୍ରকକୁ କ'ଣ କୁହାଯାଏ?\n",
            encoding="utf-8",
        )
        res = process_textbook_file(chapter_with_eval)
        self.assertEqual(len(res.records), 1)
        self.assertEqual(res.skipped_eval_sections, 1)
        self.assertNotIn("କେନ୍ଦ୍ରকକୁ କ'ଣ କୁହାଯାଏ", res.records[0].text)


# ============================================================================
# Command-Line Interface
# ============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nidana Plain-Text Textbook Ingestion Module (Odia & English)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Input directory or .txt file path (default: data/source_text).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for validated JSON records (default: data/processed).",
    )
    parser.add_argument(
        "--no-sync-extracted",
        action="store_true",
        help="Do not copy validated records into data/extracted_json/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report files, sections, and character counts without writing files.",
    )
    parser.add_argument(
        "--extract-concepts",
        action="store_true",
        help="Explicitly enable optional local Ollama concept_tag & question extraction.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Abort immediately on the first invalid file.",
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
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestImportTextbook)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    sync_dir = None if args.no_sync_extracted else DEFAULT_EXTRACTED_DIR
    mode_label = "DRY RUN (No files written)" if args.dry_run else "PRODUCTION INGESTION"
    print(f"=== NIDANA TEXTBOOK IMPORTER [{mode_label}] ===")
    print(f"Input Path : {args.input}")
    if not args.dry_run:
        print(f"Output Path: {args.output}")

    try:
        report = import_textbooks_from_path(
            input_path=args.input,
            output_dir=args.output,
            sync_extracted_dir=sync_dir,
            dry_run=args.dry_run,
            extract_concepts=args.extract_concepts,
            strict=args.strict,
        )
    except Exception as exc:
        print(f"\n[FATAL ERROR] {exc}", file=sys.stderr)
        return 1

    print("\n--- Ingestion Summary ---")
    print(f"Files Discovered          : {report.files_discovered}")
    print(f"Files Successfully Parsed : {report.files_processed}")
    print(f"Files Rejected / Failed   : {report.files_failed}")
    print(f"Validated Sections Created: {report.total_records_created}")
    print(f"Total Characters Processed: {report.total_characters_processed}")
    print(f"Eval Sections Excluded    : {report.skipped_eval_sections}")

    if not args.dry_run:
        print(f"JSON Files Written        : {len(report.output_files_written)}")

    if report.errors:
        print("\n--- Validation & Ingestion Errors ---")
        for failed_path, err_msg in report.errors:
            print(f"  [REJECTED] {failed_path}: {err_msg}")
        return 1

    if report.files_processed == 0:
        print("\n[WARNING] No valid textbook .txt files were found to process.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())