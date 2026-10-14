# Atlas Evidence

A local Arabic/English retrieval evaluation workbench. Search documents, inspect ranked source excerpts, import a small corpus and export reproducible evaluation reports. Built as an AI engineering portfolio prototype with AI coding assistance.

## Run in one command

Requires Python 3.10 or newer. No API keys, paid services or pip install.

```sh
python app.py
# Open http://127.0.0.1:8765
python -m unittest discover -v
```

## What works

- Unicode normalization, Arabic diacritic removal and alef normalization.
- Overlapping word chunks and BM25 ranking with explicit source/chunk IDs.
- Extractive answers: returned text is directly traceable to documents.
- Abstention when no query terms match.
- Recall@k and mean reciprocal rank against a bundled five-query smoke benchmark.
- JSON corpus import, ranked evidence cards and JSON report export.
- Bounded request sizes, input validation, cross-origin write rejection and loopback-only server.
- GitHub Actions test matrix for Python 3.10, 3.12 and 3.13.

## Architecture

```text
Browser -> same-origin HTTP API -> EvidenceIndex
                              -> Unicode tokenizer
                              -> overlapping chunks -> BM25 -> cited excerpts
                              -> benchmark evaluator -> JSON report
```

`engine.py` implements retrieval/evaluation independently from the server and UI. `app.py` serves JSON routes and the browser interface. No document text is interpreted as code or system instructions; UI rendering uses textContent.

## Data and APIs

`sample_documents.json` contains authored synthetic technical notes, not scraped or private data. Import an array such as:

```json
[{"id":"manual-1","title":"My note","text":"Arabic or English text here"}]
```

POST `/api/search`: `{"query":"BM25 retrieval","k":3}`. POST `/api/evaluate`: `{"k":3}`. POST `/api/documents`: `{"documents":[...]}`. GET `/api/health` returns readiness.

## Evaluation and limits

Run `python -m unittest discover -v`. Tests cover Arabic normalization, ranking, empty queries/corpora, unknown-term abstention, chunk boundaries, duplicate IDs, citation fidelity and hand-checkable ranking metrics.

The bundled benchmark is deliberately tiny and synthetic. It is a smoke test, not a general retrieval quality claim. This is **lexical retrieval**, not semantic embeddings, translation or generative RAG. Matching a word is not proof of answer relevance. Evidence can contain errors or malicious text. There is no automatic factual verification or prompt-injection classifier.

Corpus replacement is in memory and single-user oriented. Threaded imports are atomic index replacements, not versioned transactions. The development HTTP server has no authentication and must not be exposed publicly. Future work: versioned persistent indexes, human-labeled bilingual benchmarks, semantic retrieval and grounded-generation evaluation.

## Interview walkthrough

Explain BM25 length normalization, why Arabic normalization matters, the difference between recall and reciprocal rank, and why this demo returns extracts instead of implying LLM verification. Change chunk size/overlap in `EvidenceIndex` and measure the effect with independent queries.

MIT licensed. Author: Ali Farhat.
