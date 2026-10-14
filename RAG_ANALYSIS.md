# Technical Architecture & RAG Analysis: Project Nidana
**Offline Odisha State Board RAG Assistant (Classes 9, 10, 12)**

---

## Executive Summary

**Project Nidana** is an offline, air-gapped, zero-cloud Retrieval-Augmented Generation (RAG) system engineered for students of the Odisha State Board (BSE & CHSE Odisha). It runs entirely on local hardware (CPU/GPU-compatible) using an embedded dual-database architecture (**ChromaDB** + **SQLite FTS5**) and a local LLM daemon (**Ollama**).

This document delivers an in-depth technical analysis answering the five core architectural questions:
1. **Type of RAG Architecture**
2. **System Purpose & Target Domain**
3. **Architectural & Operational Limitations**
4. **Parameter Count & Comprehensive Variable/Hyperparameter Audit**
5. **Agentic Loop & Control Flow Classification**

---

## 1. What Type of RAG Is Being Used?

### Primary Classification: **Metadata-Gated Hybrid RAG with Reciprocal Rank Fusion (RRF) & Relevance Guardrails**

Rather than relying on basic naive vector retrieval, the codebase implements an advanced **Hybrid Retrieval-Augmented Generation** architecture combining dense semantic vector search with sparse lexical BM25 full-text search, fused via normalized Reciprocal Rank Fusion and filtered through strict pedagogical metadata gates.

```
                           +----------------------------------------+
                           |  Student Query (Odia / English)       |
                           |  + Filters: Class, Subject, Chapter   |
                           +-------------------+--------------------+
                                               |
                     +-------------------------+-------------------------+
                     |                                                   |
                     v                                                   v
   +----------------------------------+                +----------------------------------+
   | Dense Semantic Search            |                | Sparse Lexical Keyword Search    |
   | Engine: ChromaDB (HNSW)          |                | Engine: SQLite FTS5 (BM25)       |
   | Model : multilingual-e5-base     |                | Tokenizer: Regex [A-Za-z0-9\u0B00|
   | Prefix: "query: <text>"          |                | Filter   : STOPWORDS_EN_OR       |
   | Metric: Cosine Distance          |                | Match    : "tok1" OR "tok2"*     |
   +-----------------+----------------+                +-----------------+----------------+
                     |                                                   |
                     +-------------------------+-------------------------+
                                               |
                                               v
                          +------------------------------------------+
                          | Canonical Identity & Content Hash Dedup  |
                          | (SHA-256 fingerprinting on NFC text)    |
                          +--------------------+---------------------+
                                               |
                                               v
                          +------------------------------------------+
                          | Anti-Hallucination Relevance Gate        |
                          | Drop if Cosine Sim < 0.76 AND Lexical=0  |
                          +--------------------+---------------------+
                                               |
                                               v
                          +------------------------------------------+
                          | Score Fusion & Dual-Evidence Boosting    |
                          | RRF(k=60) + 0.55*Sem + 0.30*KW + 0.15*RRF|
                          | (1.15x Multiplier if in BOTH DBs)        |
                          +--------------------+---------------------+
                                               |
                                               v
                          +------------------------------------------+
                          | Context Budgeting (<= 3200 characters)   |
                          | Structured Citations: [1], [2], ...      |
                          +--------------------+---------------------+
                                               |
                                               v
                          +------------------------------------------+
                          | Sufficiency Gate                         |
                          | Context empty?                           |
                          |  ├─► YES: Refuse ("Insufficient context")|
                          |  └─► NO : Invoke Local Ollama (qwen2.5)  |
                          +------------------------------------------+
```

### Detailed Structural Components:

1. **Dual-Index Retrieval (Hybrid Sparse-Dense)**:
   - **Dense Vector Search**: Managed by `chromadb.PersistentClient` (`db/chroma_odisha_kb/`). Documents are embedded with `passage: ` prefix and queries with `query: ` using `intfloat/multilingual-e5-base`. Vectors are L2-normalized, allowing cosine similarity computation via $1.0 - \text{cosine\_distance}$.
   - **Sparse Lexical Search**: Managed by SQLite's native `fts5` virtual table (`offline_chunks_fts` in `db/offline_curriculum.db`). BM25 rank scores are normalized via:
     $$\text{Normalized Keyword Score} = \frac{|\text{raw\_bm25}|}{1.0 + |\text{raw\_bm25}|}$$
2. **Metadata Pre-Filtering & Query Pushdown**:
   - Searches are pre-filtered on `class_level` (Classes 9, 10, 12), `subject`, `chapter_id`/`chapter_name`, and `language` (`or`, `en`, `bilingual`).
   - In ChromaDB, this is executed as an `$and` filter pushed directly into HNSW index traversal (`build_chroma_where_clause`).
   - In SQLite FTS5, this is executed via parameterized SQL `WHERE` clauses alongside `MATCH ?`.
3. **Deduplication by Canonical Identity & Content Hash**:
   - Deduplicates candidate hits across both stores using chunk IDs and SHA-256 content hashes (`seen_text_signatures`) so duplicate passages never distort ranking or occupy redundant context window slots.
4. **Anti-Hallucination Relevance Gate**:
   - A candidate chunk retrieved purely by semantic similarity is filtered out if its cosine similarity falls below `0.76` **and** it contains zero query keyword tokens.
5. **Reciprocal Rank Fusion (RRF) & Weighted Score Blending**:
   - Computes standard RRF with smoothing constant $k = 60$:
     $$RRF(d) = \sum_{m \in \{\text{semantic}, \text{keyword}\}} \frac{1}{60 + \text{rank}_m(d)}$$
   - Computes blended composite score:
     $$\text{Combined Score} = 0.55 \cdot S_{\text{semantic}} + 0.30 \cdot S_{\text{keyword}} + 0.15 \cdot RRF_{\text{norm}}$$
   - **Dual-Evidence Boost**: If a chunk is validated by **both** ChromaDB and SQLite FTS5, its score receives a **$1.15\times$ (15%) multiplier**.
6. **Dual-Section Output Architecture**:
   - Verbatim textbook passages (Section 1) are visibly and structurally separated from LLM generation (Section 2) so answers are immediately auditable by educators and students.

---

## 2. What Is Its Purpose?

### Project Identification
- **Project Name**: **Nidana**
- **Domain**: Secondary and Higher Secondary School Education (Odisha State Board: BSE Odisha Class 9 & 10; CHSE Odisha Class 12).
- **Subjects**: STEM and Social Sciences (Physics/Physical Science, Geography, Water Resources, History, etc.).
- **Languages**: Odia (`or`), English (`en`), and Bilingual text preserving Odia Unicode script (`\u0B00-\u0B7F`) and transliterated English terminology.

### Key Objectives & Real-World Use Cases:

1. **100% Offline Classroom & Study Assistant**:
   - Engineered specifically for schools, teachers, and students in rural or low-connectivity areas of Odisha where internet access is intermittent or non-existent.
   - Operates with zero cloud dependencies: no OpenAI, no Anthropic, no Google Cloud, and no external API keys.
2. **Strictly Grounded, Zero-Hallucination Question Answering (`answer_question`)**:
   - Answers student queries exclusively using verified Odisha board textbook material.
   - Enforces numbered source citations (`[1]`, `[2]`).
   - **Explicit Failure Disclosure**: If relevant passages are missing, the system refuses to guess or invent facts, explicitly outputting: *"Insufficient textbook context... I cannot answer this question without inventing facts outside the textbook."*
3. **Textbook-Isolated Practice Question Generation (`generate_practice_questions`)**:
   - Generates Odisha State Board-style 4-option Multiple Choice Questions (MCQs) and Descriptive Questions with model answers.
   - **Data-Poisoning Prevention Guarantee**: Generated questions are strictly forbidden from being written back into ChromaDB or SQLite FTS5; they are saved only to isolated evaluation banks (`data/evaluation/`).
4. **Cross-Class Prerequisite Remediation (`allow_remediation`)**:
   - Allows Class 10 queries to retrieve foundational prerequisite concepts indexed under Class 9 when a student has conceptual gaps.

---

## 3. What Are Its Limitations?

| Limitation Category | Specific Bottleneck in Codebase | Technical Impact |
| :--- | :--- | :--- |
| **1. Ingestion / Multimodality** | Plain-text only (`import_textbook.py` excludes PDF/OCR and vision). | Cannot ingest diagrams, circuit diagrams, chemical structures, maps, or mathematical graphs crucial for STEM subjects. |
| **2. Context Window Budget** | `DEFAULT_MAX_CONTEXT_CHARS = 3200` (~600-800 words) in `rag_runtime.py`. | Complex synthesis spanning multiple sections is truncated to fit the character budget. |
| **3. Chunking Heuristics** | Word-count windowing (`180` words, `30` overlap) in `3_build_databases.py`. | Although formulas and definitions are regex-protected, large tabular data or extended proofs may experience semantic fragmentation. |
| **4. Odia Morphological Analysis** | SQLite FTS5 uses a simple character token regex `[A-Za-z0-9\u0B00-\u0B7F]{2,}` without Odia lemmatization or sandhi splitting. | Inflected Odia verb forms, case markers, or compound words can fail exact BM25 keyword matching. |
| **5. Small Local LLM Capacity** | Default generator is `qwen2.5:1.5b` (1.54B parameters) via Ollama. | Complex logical reasoning, strict multi-step adherence, and subtle regional linguistic nuances are constrained compared to larger models. |
| **6. Fixed Cosine Gate** | `DEFAULT_MIN_SEMANTIC_SIMILARITY = 0.76`. | In low-resource Indic languages, cross-lingual representations occasionally score below 0.76, creating false negatives if keyword tokens do not overlap. |
| **7. Stateless Session Model** | `rag_runtime.py` operates as a single-turn query handler. | No conversation memory or multi-turn conversational context is preserved between questions. |
| **8. Non-Pedagogical Verification** | As documented in `4_test_rag.py`: automated tests verify retrieval metrics and lexical overlap, not human pedagogical completeness. | Does not replace human review by qualified Odisha State Board teachers. |

---

## 4. Parameter Count & Variables Deciding Accuracy & Health

### 4.1 Model Parameter Count Breakdown

```
+-----------------------------------------------------------------------------------------+
|                                     ACTIVE MODELS                                       |
+------------------------------+---------------------------+----------------+-------------+
| Role                         | Model Identifier          | Param Count    | Quant / Dim |
+------------------------------+---------------------------+----------------+-------------+
| Dense Vector Embeddings      | intfloat/multilingual-e5-base | ~278 Million   | 768 float32 |
| Local Generator (Default LLM)| qwen2.5:1.5b              | ~1.54 Billion  | Q4_K_M/FP16 |
+------------------------------+---------------------------+----------------+-------------+
| TOTAL ACTIVE RUNTIME PARAMS  |                           | ~1.82 Billion  |             |
+------------------------------+---------------------------+----------------+-------------+
| Optional Concept Extractor   | qwen2.5:3b (offline prep) | ~3.09 Billion  | Q4_K_M      |
+------------------------------+---------------------------+----------------+-------------+
```

- **Embedding Model (`intfloat/multilingual-e5-base`)**:
  - **Architecture**: XLM-RoBERTa Transformer Encoder (12 layers, 768 hidden dimensions, 12 attention heads, 3072 feed-forward dimension).
  - **Exact Parameter Count**: **278,043,648 parameters** (~278M).
  - **Vocabulary Size**: 250,002 tokens (sentencepiece multilingual vocabulary covering Odia).
  - **Maximum Context Sequence Length**: 512 tokens.
- **Generation Model (`qwen2.5:1.5b`)**:
  - **Architecture**: Qwen 2.5 Decoder-only Transformer with RoPE, SwiGLU, and Grouped Query Attention (GQA).
  - **Exact Parameter Count**: **1,543,714,816 parameters** (~1.54B; ~1.31B non-embedding parameters).
  - **Execution Footprint**: ~0.98 GB VRAM in 4-bit quantized mode (`Q4_K_M`), ~3.1 GB in FP16.

---

### 4.2 Comprehensive Variables Deciding Accuracy & Overall Health

The accuracy, recall, and health of this RAG pipeline are governed by the following variables across all pipeline stages:

#### A. Ingestion & Preprocessing Variables (`import_textbook.py`, `embeddings.py`)
- `ALLOWED_CLASSES = {9, 10, 12}`: Enforces strict target grade boundaries.
- `ALLOWED_LANGUAGES = {"or", "en", "bilingual"}`: Rejects unsupported languages.
- `MIN_CONTENT_LENGTH = 10`: Rejects empty or corrupt chunk imports.
- `unicodedata.normalize("NFC", text)`: Canonical Unicode normalization preserving Odia scripts (`\u0B00-\u0B7F`) and joiners (ZWNJ `\u200c`, ZWJ `\u200d`).
- `EVALUATION_SECTION_PATTERNS`: Regex excluding exercise questions from the textbook body to prevent circular self-referencing.
- `MOJIBAKE_REGEX` & `LEGACY_FONT_GARBAGE_REGEX`: Detects and flags corrupted legacy ASCII Odia fonts (e.g. Akruti / ShreeLipi byte corruption).

#### B. Chunking & Segmentation Variables (`3_build_databases.py`)
- `DEFAULT_CHUNK_SIZE_WORDS = 180`: Target chunk length in words.
- `DEFAULT_CHUNK_OVERLAP_WORDS = 30`: Word overlap preserving semantic continuity across chunk edges.
- `FORMULA_OR_DEF_REGEX`: Regex pattern ensuring mathematical equations, scientific definitions, and theorem statements are fused with their introductory context.
- `HEADING_LINE_REGEX` & `SENTENCE_SPLIT_REGEX`: Splits on sentence boundaries (`[.!?।॥]`) rather than arbitrary character cuts.

#### C. Vector Embedding & Storage Variables (`embeddings.py`)
- `EXPECTED_EMBEDDING_DIM = 768`: Output dimension of E5 embedding vectors.
- `PASSAGE_PREFIX = "passage: "` / `QUERY_PREFIX = "query: "`: Mandatory prefixes required for asymmetrical semantic retrieval in E5 models.
- `DEFAULT_BATCH_SIZE = 32`: Parallel embedding throughput batch size.
- `compute_record_hash`: SHA-256 fingerprint for incremental delta-updates (skips re-embedding unchanged documents).

#### D. Lexical FTS5 Search Variables (`rag_runtime.py`, `3_build_databases.py`)
- `FTS_TOKEN_REGEX = r"[A-Za-z0-9\u0B00-\u0B7F]{2,}"`: Token extraction regex preserving English and Odia characters.
- `STOPWORDS_EN_OR`: Curated list of English conversational interrogatives (`what`, `why`, `explain`) and Odia question words (`କଣ`, `କାହିଁକି`, `କିପରି`, `କେଉଁ`).
- `UNINDEXED` columns: In `offline_chunks_fts`, only 5 columns (`chapter_name`, `section_heading`, `concept_tag`, `english_tech_terms`, `content_odia`) are indexed by FTS5 to keep the index lean.

#### E. Ranking, Fusion & Filtering Variables (`rag_runtime.py`)
- `DEFAULT_TOP_K = 5`: Number of candidate chunks selected for prompt context.
- `DEFAULT_RRF_K = 60`: Smoothing denominator in Reciprocal Rank Fusion.
- `DEFAULT_MIN_SEMANTIC_SIMILARITY = 0.76`: Anti-hallucination cosine similarity threshold.
- **Score Blending Weights**:
  - Semantic Weight: **`0.55`** (55%)
  - Keyword Weight: **`0.30`** (30%)
  - RRF Weight: **`0.15`** (15%)
- **Dual-Evidence Boost**: **`1.15x`** multiplier when a chunk appears in both ChromaDB and SQLite FTS5.
- `allow_remediation = False`: When enabled, relaxes Class filter to include foundational prerequisite concepts from earlier grades.

#### F. Context Assembly & Budgeting Variables (`rag_runtime.py`)
- `DEFAULT_MAX_CONTEXT_CHARS = 3200`: Maximum character limit for the retrieved context block.
- Snippet Truncation Threshold (`120` chars): Minimum viable character slice allowed when clipping the last chunk to fit the budget.

#### G. LLM Generation Variables (`rag_runtime.py`)
- `temperature = 0.1`: Low temperature minimizing creative hallucinations and enforcing deterministic adherence to the text.
- `num_predict = 600`: Maximum tokens allowed in Ollama's generation output.
- `DEFAULT_OLLAMA_TIMEOUT_SEC = 45`: Request timeout before reporting Ollama unavailability.
- System prompt instruction: Strictly forces inline citation markers (`[1]`, `[2]`).

#### H. Health & Grounding Audit Metrics (`4_test_rag.py`)
- `1:1 Database Parity`: Validates that ChromaDB and SQLite FTS5 contain identical chunk IDs and identical content hashes.
- `Recall@k` ($k \in \{1, 3, 5\}$): Quantitative benchmark of retrieval accuracy against gold-standard evaluation questions.
- `min_term_overlap = 0.20`: Audits that at least 20% of content tokens in the LLM's response overlap with retrieved passages.
- Citation Index Bounds Check: Asserts citation markers are within the exact range $1 \dots N$.

---

## 5. What Type of Agentic Loop Does This RAG Use?

### Architectural Assessment: **Deterministic Gated Control Flow with Anti-Hallucination Guardrails, Short-Circuit Fallbacks, and Ingestion/Evaluation Hooks**

```
                         [User Query & Filter Input]
                                      │
                                      ▼
                        [Filter Validation Gate]
                                      │
                                      ▼
                    [Parallel Hybrid Retrieval Stage]
                         ├── ChromaDB (Semantic)
                         └── SQLite FTS5 (Lexical)
                                      │
                                      ▼
                      [Deduplication & RRF Ranking]
                                      │
                                      ▼
                      [Relevance Gate (Sim >= 0.76?)]
                                      │
                                      ▼
                      [Context Assembly & Budgeting]
                                      │
                                      ▼
                    ┌───────────────────────────────────┐
                    │     Context Sufficiency Gate      │
                    └─────────────────┬─────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼ (Empty Context)                         ▼ (Chunks Found)
   ┌───────────────────────────┐             ┌───────────────────────────┐
   │ Short-Circuit Refusal     │             │ Construct Grounded Prompt │
   │ (Skip LLM Execution)      │             └─────────────┬─────────────┘
   └───────────────────────────┘                           │
                                                           ▼
                                             ┌───────────────────────────┐
                                             │      Ollama Call Gate     │
                                             └─────────────┬─────────────┘
                                                           │
                                          ┌────────────────┴────────────────┐
                                          ▼ (Timeout / Unreachable)         ▼ (200 OK)
                            ┌───────────────────────────┐     ┌───────────────────────────┐
                            │ Honest Fallback Notice    │     │ Output Dual-Section Result│
                            │ + Verbatim Passages Only  │     │ 1. Verbatim Passages      │
                            └───────────────────────────┘     │ 2. Grounded Explanation   │
                                                              └───────────────────────────┘
```

### Detailed Loop Analysis:

1. **Not an Autonomous ReAct / Tool-Use Loop at Runtime**:
   - The runtime query path (`answer_question` in `rag_runtime.py`) does **not** execute an open-ended, autonomous ReAct (Reason + Act), Plan-and-Solve, or multi-hop query-rewriting agentic loop.
   - It is intentionally designed as a **deterministic, predictable, and fail-safe pipeline** to ensure rapid local response times and zero uncontrolled LLM branching in resource-limited classroom settings.

2. **The 4 Guardrail & Feedback Mechanisms in the Codebase**:

   - **A. Short-Circuit Sufficiency Guardrail**:
     Before calling the LLM, the system checks `if not chunks or not context_str.strip()`. If no approved passages pass the relevance gate, the pipeline short-circuits: it does not call Ollama, avoids token waste, and issues an explicit refusal message.
   - **B. Fault-Tolerant Availability Guardrail**:
     If Ollama fails to respond within 45 seconds or the local daemon is down, the system does not crash or fabricate an answer; it outputs the verbatim retrieved textbook chunks with an honest status notification (`llm_status="unavailable"`).
   - **C. Curriculum Remediation Traversal Loop (`allow_remediation`)**:
     When remediation is enabled, the filter logic relaxes the class boundary:
     $$\text{class\_level} \le \text{target\_class}$$
     allowing the retriever to traverse backward through the curriculum dependency graph to pull foundational concepts.
   - **D. Offline Semi-Agentic Concept Extraction Loop (`import_textbook.py`)**:
     During data preparation (`--extract-concepts`), `enrich_with_ollama_optional` executes a structured JSON extraction loop over textbook sections, prompting the LLM to output valid `concept_tag` and `diagnostic_question_odia` attributes.
   - **E. Post-Generation Grounding Audit Loop (`4_test_rag.py`)**:
     The evaluation harness (`audit_answer_grounding`) acts as a verification loop checking citation marker validity (`[1]`, `[2]`), citation index bounds, and lexical overlap ratio ($> 0.20$) to catch hallucinations.

---

## Summary Matrix

| Question | Architectural Answer | Key References in Code |
| :--- | :--- | :--- |
| **RAG Type** | **Hybrid RAG** (ChromaDB Vector + SQLite FTS5 BM25) with Reciprocal Rank Fusion (RRF), Dual-Evidence Boosting, and Metadata Pre-Filtering. | `rag_runtime.py:540-665`, `rag_runtime.py:668-761`, `embeddings.py:40-52` |
| **Purpose** | **Offline Odisha State Board Assistant (Nidana)** for Classes 9, 10, 12: Zero-hallucination Q&A and isolated practice question generation in Odia and English. | `rag_runtime.py:863-1017`, `rag_runtime.py:1019-1180`, `import_textbook.py:1-12` |
| **Limitations** | Plain-text only (no diagrams/circuits), 3200-character context budget, 180-word chunk boundaries, basic Odia regex tokenization without lemmatization, and small 1.5B generator. | `rag_runtime.py:73-77`, `3_build_databases.py:59-60`, `rag_runtime.py:94-95` |
| **Parameter Count** | **~1.82 Billion total active parameters** (~278M E5 embedding model + ~1.54B Qwen2.5-1.5B LLM). | `embeddings.py:47-49`, `rag_runtime.py:70`, `requirements.txt:12-28` |
| **Agentic Loop** | **Deterministic Gated Pipeline with Guardrails & Short-Circuits** (not an unconstrained ReAct loop), featuring a sufficiency gate, remediation traversal, and grounding audit. | `rag_runtime.py:904-934`, `rag_runtime.py:963-991`, `4_test_rag.py:77-140` |

---
*Generated from technical static analysis of the Project Nidana codebase.*
