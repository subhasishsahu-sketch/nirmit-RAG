#!/usr/bin/env python3
"""
PROJECT: Nidana
FILE: embeddings.py

Reusable singleton embedding service for the one-time Nidana RAG ingestion
pipeline and offline runtime.

Model: intfloat/multilingual-e5-base (768 dimensions, L2-normalized for cosine similarity)
- Prefixes indexed documents with: "passage: "
- Prefixes search queries with   : "query: "
"""

import argparse
import hashlib
import os
import re
import sys
import threading
import unicodedata
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


# ============================================================================
# Canonical Project Paths & Configuration Constants
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_TEXT_DIR = DATA_DIR / "source_text"
PROCESSED_DIR = DATA_DIR / "processed"
EXTRACTED_JSON_DIR = DATA_DIR / "extracted_json"
EVALUATION_DIR = DATA_DIR / "evaluation"

DB_DIR = PROJECT_ROOT / "db"
CHROMA_DB_PATH = DB_DIR / "chroma_odisha_kb"
SQLITE_DB_PATH = DB_DIR / "offline_curriculum.db"
CHROMA_COLLECTION_NAME = "odisha_board_stem"

CANONICAL_EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EXPECTED_EMBEDDING_DIM = 768

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
_PREFIX_STRIP_RE = re.compile(r"^(?:(?:passage|query)\s*:\s*)+", re.IGNORECASE)

DEFAULT_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
LARGE_COLLECTION_THRESHOLD = 16


# ============================================================================
# Exceptions
# ============================================================================

class EmbeddingError(Exception):
    """Base exception for embedding operations."""


class EmptyInputError(EmbeddingError):
    """Raised when an empty or whitespace-only text/record is passed for embedding."""


class ModelConfigurationError(EmbeddingError):
    """Raised when an unapproved model is configured without explicit justification."""


class ModelLoadError(EmbeddingError):
    """Raised when the embedding model fails to load from cache or hub."""


# ============================================================================
# Directory & Text Preprocessing Helpers
# ============================================================================

def ensure_directories() -> None:
    """Safely creates all canonical project directories without deleting existing files."""
    for folder in (
        SOURCE_TEXT_DIR,
        PROCESSED_DIR,
        EXTRACTED_JSON_DIR,
        EVALUATION_DIR,
        DB_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def preprocess_text(text: str) -> str:
    """
    Normalizes text consistently for multilingual-e5-base while preserving:
    - Odia Unicode characters (0B00-0B7F) and joiners (ZWNJ \\u200c, ZWJ \\u200d)
    - English prose, technical terms, mathematical formulas, and punctuation
    """
    if not isinstance(text, str):
        raise EmptyInputError(f"Expected string input for embedding, got {type(text).__name__}")

    cleaned = text.lstrip("\ufeff")
    cleaned = unicodedata.normalize("NFC", cleaned)
    # Remove non-printable ASCII control chars except newline and tab; keep ZWNJ/ZWJ
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    # Collapse internal runs of whitespace/newlines into single spaces for E5 input consistency
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def strip_e5_prefix(text: str) -> str:
    """Removes any leading 'passage:' or 'query:' prefixes regardless of trailing spaces."""
    if not isinstance(text, str):
        return ""
    return _PREFIX_STRIP_RE.sub("", text.strip()).strip()


def format_passage(text: str, english_terms: str = "") -> str:
    """
    Formats document content with the required 'passage: ' prefix for
    intfloat/multilingual-e5-base. Rejects empty or prefix-only strings.
    """
    if not isinstance(text, str):
        raise EmptyInputError("Document text must be a string.")

    core_text = preprocess_text(strip_e5_prefix(text))
    if not core_text:
        raise EmptyInputError("Cannot format an empty or whitespace-only passage.")

    if english_terms:
        core_lower = core_text.lower()
        missing_terms = [
            preprocess_text(t)
            for t in english_terms.split(",")
            if preprocess_text(t) and preprocess_text(t).lower() not in core_lower
        ]
        if missing_terms:
            return f"{PASSAGE_PREFIX}{core_text} (Terms: {', '.join(missing_terms)})"

    return f"{PASSAGE_PREFIX}{core_text}"


def format_query(query: str) -> str:
    """
    Formats a search query with the required 'query: ' prefix for
    intfloat/multilingual-e5-base. Rejects empty or prefix-only queries.
    """
    if not isinstance(query, str):
        raise EmptyInputError("Search query must be a string.")

    core_query = preprocess_text(strip_e5_prefix(query))
    if not core_query:
        raise EmptyInputError("Cannot format an empty or whitespace-only search query.")

    return f"{QUERY_PREFIX}{core_query}"


# ============================================================================
# Record Validation & Incremental Change Detection (Avoid Re-Embedding)
# ============================================================================

def is_valid_record_for_embedding(record: Dict[str, Any]) -> bool:
    """
    Returns True only if the record contains non-empty textbook prose.
    Rejects empty or metadata-only records.
    """
    if not isinstance(record, dict):
        return False
    raw_text = record.get("text")
    if not isinstance(raw_text, str):
        return False
    core_text = preprocess_text(strip_e5_prefix(raw_text))
    return len(core_text) > 0


def compute_record_hash(record: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 fingerprint of a canonical record's embeddable
    content so unchanged records are not unnecessarily re-embedded during incremental updates.
    """
    if not is_valid_record_for_embedding(record):
        raise EmptyInputError(f"Cannot hash empty or metadata-only record: {record.get('id')!r}")

    meta = record.get("metadata") or {}
    formatted_passage = format_passage(
        record["text"],
        english_terms=str(meta.get("english_tech_terms", "")),
    )
    payload = "|".join([
        str(record.get("id", "")).strip(),
        formatted_passage,
        str(meta.get("class_level", "")),
        str(meta.get("subject", "")),
        str(meta.get("chapter_id", meta.get("chapter_num", ""))),
        str(meta.get("language", "")),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def filter_unchanged_records(
    records: Sequence[Dict[str, Any]],
    existing_hashes: Dict[str, str],
    skip_empty: bool = True,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Filters a list of canonical records before embedding:
    - Rejects/skips empty or metadata-only records.
    - Skips records whose SHA-256 content_hash matches `existing_hashes[record_id]`.
    - Attaches `content_hash` into `record["metadata"]["content_hash"]` for persistence.

    Returns:
        (records_to_embed, skipped_unchanged_count, skipped_empty_count)
    """
    to_embed: List[Dict[str, Any]] = []
    skipped_unchanged = 0
    skipped_empty = 0

    for rec in records:
        if not is_valid_record_for_embedding(rec):
            if skip_empty:
                skipped_empty += 1
                continue
            raise EmptyInputError(f"Record {rec.get('id')!r} has empty or metadata-only content.")

        rec_id = str(rec.get("id", "")).strip()
        if not rec_id:
            if skip_empty:
                skipped_empty += 1
                continue
            raise EmptyInputError("Record is missing a valid 'id'.")

        current_hash = compute_record_hash(rec)
        if existing_hashes.get(rec_id) == current_hash:
            skipped_unchanged += 1
            continue

        rec_copy = dict(rec)
        meta_copy = dict(rec.get("metadata") or {})
        meta_copy["content_hash"] = current_hash
        rec_copy["metadata"] = meta_copy
        to_embed.append(rec_copy)

    return to_embed, skipped_unchanged, skipped_empty


# ============================================================================
# Singleton Embedding Service & ChromaDB Adapter
# ============================================================================

def resolve_execution_device(requested_device: Optional[str] = None) -> str:
    """Selects a CPU-compatible execution device when a GPU is unavailable."""
    env_device = (requested_device or os.getenv("EMBEDDING_DEVICE", "cpu")).strip().lower()
    if env_device.startswith("cuda"):
        try:
            import torch
            if torch.cuda.is_available():
                return env_device
            print("[embeddings] CUDA requested but no GPU detected; falling back to CPU.")
            return "cpu"
        except ImportError:
            return "cpu"
    return "cpu"


class NidanaEmbeddingService:
    """
    Process-wide singleton managing `intfloat/multilingual-e5-base`.
    Ensures the model is loaded at most once per process and never silently
    substitutes a different model.
    """

    _instance: Optional["NidanaEmbeddingService"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        configured_model = (
            model_name
            or os.getenv("EMBEDDING_MODEL_NAME", CANONICAL_EMBEDDING_MODEL)
        ).strip()

        if configured_model != CANONICAL_EMBEDDING_MODEL:
            raise ModelConfigurationError(
                f"Configured model '{configured_model}' does not match the audited Nidana model "
                f"'{CANONICAL_EMBEDDING_MODEL}'. Silent model switching is prohibited."
            )

        self.model_name = configured_model
        self.device = resolve_execution_device(device)
        self._model = None
        self.load_count = 0

    def _ensure_model_loaded(self) -> Any:
        """Lazily loads SentenceTransformer once per process in a thread-safe lock."""
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ModelLoadError(
                    "sentence-transformers is not installed. Run: pip install sentence-transformers"
                ) from exc

            try:
                try:
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                        local_files_only=True,
                    )
                except Exception:
                    print(f"[embeddings] Loading '{self.model_name}' on device='{self.device}'...")
                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                    )
                self.load_count += 1
            except Exception as exc:
                raise ModelLoadError(
                    f"Failed to load embedding model '{self.model_name}' on '{self.device}': {exc}"
                ) from exc

        return self._model

    @property
    def dimension(self) -> int:
        model = self._ensure_model_loaded()
        dim = model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else EXPECTED_EMBEDDING_DIM

    def embed_passages(
        self,
        texts: Sequence[str],
        english_terms_list: Optional[Sequence[str]] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: Optional[bool] = None,
        skip_empty: bool = False,
    ) -> List[List[float]]:
        """
        Embeds a batch of document passages using the 'passage: ' prefix.
        Returns L2-normalized vectors in ChromaDB's required List[List[float]] format.
        """
        if texts is None or len(texts) == 0:
            raise EmptyInputError("Cannot embed an empty list of passages.")

        if english_terms_list is not None and len(english_terms_list) != len(texts):
            raise ValueError("english_terms_list length must match texts length.")

        formatted_inputs: List[str] = []
        for idx, raw_text in enumerate(texts):
            terms = english_terms_list[idx] if english_terms_list is not None else ""
            try:
                formatted_inputs.append(format_passage(raw_text, english_terms=terms))
            except EmptyInputError:
                if skip_empty:
                    continue
                raise EmptyInputError(f"Passage at index {idx} is empty or whitespace-only.")

        if not formatted_inputs:
            raise EmptyInputError("No non-empty passages remained after filtering.")

        return self._encode_formatted_batch(
            formatted_inputs,
            batch_size=batch_size,
            show_progress=show_progress,
            desc="Embedding passages",
        )

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single search query using the 'query: ' prefix.
        Returns a 768-dimensional L2-normalized List[float].
        """
        formatted_q = format_query(query)
        vectors = self._encode_formatted_batch(
            [formatted_q],
            batch_size=1,
            show_progress=False,
        )
        return vectors[0]

    def embed_queries(
        self,
        queries: Sequence[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> List[List[float]]:
        """Embeds multiple search queries using the 'query: ' prefix."""
        if queries is None or len(queries) == 0:
            raise EmptyInputError("Cannot embed an empty list of queries.")

        formatted_queries = [format_query(q) for q in queries]
        return self._encode_formatted_batch(
            formatted_queries,
            batch_size=batch_size,
            show_progress=False,
        )

    def _encode_formatted_batch(
        self,
        formatted_texts: List[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: Optional[bool] = None,
        desc: str = "Embedding",
    ) -> List[List[float]]:
        model = self._ensure_model_loaded()
        effective_batch_size = max(1, int(batch_size))
        total = len(formatted_texts)

        if show_progress is None:
            show_progress = total >= LARGE_COLLECTION_THRESHOLD

        all_embeddings: List[List[float]] = []
        indices = range(0, total, effective_batch_size)
        iterator = (
            tqdm(indices, desc=desc, unit="batch")
            if show_progress
            else indices
        )

        for start_idx in iterator:
            batch_slice = formatted_texts[start_idx : start_idx + effective_batch_size]
            vecs = model.encode(
                batch_slice,
                batch_size=effective_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for row in vecs:
                all_embeddings.append([float(x) for x in row.tolist()])

        return all_embeddings


class ChromaE5EmbeddingFunction:
    """
    ChromaDB-compatible EmbeddingFunction wrapper around the singleton
    NidanaEmbeddingService.
    """

    def __init__(self, service: NidanaEmbeddingService) -> None:
        self._service = service

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        if input is None or len(input) == 0:
            raise EmptyInputError("ChromaDB passed an empty input list to EmbeddingFunction.")

        formatted_list: List[str] = []
        for idx, item in enumerate(input):
            if not isinstance(item, str) or not preprocess_text(strip_e5_prefix(item)):
                raise EmptyInputError(f"Empty document at index {idx} cannot be embedded.")
            stripped = item.strip()
            if stripped.lower().startswith("query:"):
                formatted_list.append(format_query(stripped))
            else:
                formatted_list.append(format_passage(stripped))

        return self._service._encode_formatted_batch(
            formatted_list,
            batch_size=DEFAULT_BATCH_SIZE,
        )


# ============================================================================
# Public Module-Level Singleton Accessors
# ============================================================================

_SERVICE_SINGLETON: Optional[NidanaEmbeddingService] = None
_CHROMA_EF_SINGLETON: Optional[ChromaE5EmbeddingFunction] = None
_SINGLETON_LOCK = threading.Lock()


def get_embedding_service() -> NidanaEmbeddingService:
    """Returns the process-wide singleton NidanaEmbeddingService."""
    global _SERVICE_SINGLETON
    if _SERVICE_SINGLETON is None:
        with _SINGLETON_LOCK:
            if _SERVICE_SINGLETON is None:
                _SERVICE_SINGLETON = NidanaEmbeddingService()
    return _SERVICE_SINGLETON


def get_embedding_function() -> ChromaE5EmbeddingFunction:
    """
    Returns the singleton ChromaDB-compatible embedding function.
    Used by 3_build_databases.py and 4_test_rag.py.
    """
    global _CHROMA_EF_SINGLETON
    if _CHROMA_EF_SINGLETON is None:
        with _SINGLETON_LOCK:
            if _CHROMA_EF_SINGLETON is None:
                _CHROMA_EF_SINGLETON = ChromaE5EmbeddingFunction(get_embedding_service())
    return _CHROMA_EF_SINGLETON


def embed_passages(
    texts: Sequence[str],
    english_terms_list: Optional[Sequence[str]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    show_progress: Optional[bool] = None,
    skip_empty: bool = False,
) -> List[List[float]]:
    """Convenience wrapper calling the singleton service's embed_passages()."""
    return get_embedding_service().embed_passages(
        texts=texts,
        english_terms_list=english_terms_list,
        batch_size=batch_size,
        show_progress=show_progress,
        skip_empty=skip_empty,
    )


def embed_query(query: str) -> List[float]:
    """Convenience wrapper calling the singleton service's embed_query()."""
    return get_embedding_service().embed_query(query)


# ============================================================================
# Unit Tests (Dimensions, Empty Handling, Batch Count, Odia/English, Prefixes)
# ============================================================================

class TestNidanaEmbeddings(unittest.TestCase):
    """Built-in unit tests verifying all requirements for embeddings.py."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.service = get_embedding_service()

    def test_1_prefixes_applied_without_duplication(self) -> None:
        """Verifies 'passage: ' and 'query: ' prefixes are applied and never doubled."""
        p1 = format_passage("ଜଡ଼ତା ଏବଂ ବଳ (Inertia and Force)")
        self.assertTrue(p1.startswith("passage: "))
        self.assertFalse(p1.startswith("passage: passage: "))

        # Passing an already-prefixed passage with terms already present in the text
        p2 = format_passage(p1, english_terms="inertia, force")
        self.assertEqual(p2, "passage: ଜଡ଼ତା ଏବଂ ବଳ (Inertia and Force)")

        # Passing a new term not in the text appends only the missing term
        p3 = format_passage(p1, english_terms="inertia, momentum")
        self.assertEqual(p3, "passage: ଜଡ଼ତା ଏବଂ ବଳ (Inertia and Force) (Terms: momentum)")

        q1 = format_query("What is Ohm's Law?")
        self.assertEqual(q1, "query: What is Ohm's Law?")
        q2 = format_query(q1)
        self.assertEqual(q2, "query: What is Ohm's Law?")

    def test_2_empty_documents_rejected_or_skipped_safely(self) -> None:
        """Verifies empty strings, whitespace, and metadata-only records are rejected or skipped."""
        with self.assertRaises(EmptyInputError):
            format_passage("   \n\t  ")

        with self.assertRaises(EmptyInputError):
            format_passage("passage:    ")

        with self.assertRaises(EmptyInputError):
            format_query("")

        with self.assertRaises(EmptyInputError):
            self.service.embed_passages(["Valid text", "   "], skip_empty=False)

        vecs = self.service.embed_passages(["Valid textbook passage", "   "], skip_empty=True)
        self.assertEqual(len(vecs), 1)

        meta_only_record = {"id": "c9_001", "text": "   ", "metadata": {"class_level": 9}}
        self.assertFalse(is_valid_record_for_embedding(meta_only_record))

    def test_3_embedding_dimensions_and_singleton_reuse(self) -> None:
        """Verifies 768-dim vector output and single model load per process."""
        q_vec = embed_query("ବିଦ୍ୟୁତ୍ ସ୍ରୋତ କ'ଣ (What is electric current)?")
        self.assertEqual(len(q_vec), EXPECTED_EMBEDDING_DIM)
        self.assertTrue(all(isinstance(v, float) for v in q_vec))

        service_again = get_embedding_service()
        self.assertIs(self.service, service_again)
        self.assertEqual(self.service.load_count, 1)

    def test_4_batch_output_count_and_odia_english_support(self) -> None:
        """Verifies batch output count matches input count across Odia and English passages."""
        passages = [
            "ଗତି ଏବଂ ବିଶ୍ରାମ (Motion and Rest): ବସ୍ତୁର ସ୍ଥିତି ସମୟ ସହିତ ପରିବର୍ତ୍ତନ ହୁଏ।",
            "Water resources in India include surface water, rivers, and groundwater aquifers.",
            "ଓମ୍‌ଙ୍କ ନିୟମ (Ohm's Law): V = I × R, ଯେଉଁଠାରେ R ହେଉଛି ପ୍ରତିରୋଧ (Resistance)।",
        ]
        embeddings = embed_passages(passages, batch_size=2, show_progress=False)
        self.assertEqual(len(embeddings), len(passages))
        for vec in embeddings:
            self.assertEqual(len(vec), EXPECTED_EMBEDDING_DIM)
            norm_sq = sum(x * x for x in vec)
            self.assertAlmostEqual(norm_sq, 1.0, places=3)

    def test_5_incremental_hash_avoids_reembedding_unchanged_records(self) -> None:
        """Verifies unchanged records are skipped during incremental updates."""
        rec1 = {
            "id": "c10_geo_ch03_s001",
            "text": "Water scarcity and conservation in Odisha.",
            "metadata": {"class_level": 10, "subject": "Geography", "chapter_id": "ch03", "language": "en"},
        }
        rec2 = {
            "id": "c10_geo_ch03_s002",
            "text": "Multipurpose river valley projects like Hirakud Dam.",
            "metadata": {"class_level": 10, "subject": "Geography", "chapter_id": "ch03", "language": "en"},
        }
        h1 = compute_record_hash(rec1)
        existing = {"c10_geo_ch03_s001": h1}

        to_embed, skipped_unchanged, skipped_empty = filter_unchanged_records(
            [rec1, rec2, {"id": "empty_rec", "text": "  ", "metadata": {}}],
            existing_hashes=existing,
            skip_empty=True,
        )
        self.assertEqual(len(to_embed), 1)
        self.assertEqual(to_embed[0]["id"], "c10_geo_ch03_s002")
        self.assertEqual(skipped_unchanged, 1)
        self.assertEqual(skipped_empty, 1)

    def test_6_prohibit_silent_model_change(self) -> None:
        """Verifies attempting to initialize with a different model raises ModelConfigurationError."""
        with self.assertRaises(ModelConfigurationError):
            NidanaEmbeddingService(model_name="all-MiniLM-L6-v2")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Nidana Embedding Service (intfloat/multilingual-e5-base).")
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the built-in unit test suite verifying dimensions, prefixes, batching, and Odia/English support.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a quick test embedding on the provided query string.",
    )
    args = parser.parse_args(argv)

    if args.run_tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestNidanaEmbeddings)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return 0 if result.wasSuccessful() else 1

    service = get_embedding_service()
    sample_query = args.query or "ଜଡ଼ତା ଏବଂ ବଳ କ'ଣ (What is inertia and force)?"
    vec = service.embed_query(sample_query)
    print("=== NIDANA EMBEDDING SERVICE READY ===")
    print(f"Model Name       : {service.model_name}")
    print(f"Execution Device : {service.device}")
    print(f"Vector Dimension : {len(vec)}")
    print(f"Formatted Query  : {format_query(sample_query)}")
    print(f"Sample Vector[:5]: {[round(x, 5) for x in vec[:5]]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())