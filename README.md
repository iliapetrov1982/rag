# Filmoskop Support RAG

A retrieval-augmented support assistant for a fictional streaming service ("Filmoskop"): subscriptions,
devices, playback error codes, accounts and profiles. The knowledge base is noisy (three export formats,
markup leftovers, duplicates), the test set is labelled, and the goal is a measured, reproducible answer to
one question: can this system be trusted, and what should be improved first.

Status: complete. Verdict: retrieval meets its target, answer correctness does not; the system is not ready
for an unattended pilot. Details are in [Results](#results).

## What is inside

```
project/
  README.md              # this file: results, how to run
  filmoskop_rag.ipynb    # the whole pipeline in six stages
  data/                  # filmoskop_kb.json and generated artifacts (not in git)
```

The notebook walks through six stages, each ending with a short written conclusion:

1. Corpus audit and cleaning: junk ratio, markup leftovers, length outliers, duplicates; before/after on
   concrete documents.
2. Chunking and embeddings: two chunking strategies compared on retrieval metrics and by reading the chunks.
3. Vector store and retrieval pipeline: Qdrant collection, similarity threshold picked from the score
   distribution, metadata filtering, baseline precision@5 / recall@5 / MRR and the recall@K curve.
4. Retrieval diagnosis and improvement: where the relevant documents rank, manual review of the misses,
   a cross-encoder reranker chosen by the diagnosis, before/after on the same queries, the cost of the change.
5. Answer generation: LCEL chain, grounded prompt, explicit refusal when the knowledge base has no answer,
   temperature 0.
6. Evaluation: LLM-as-judge (faithfulness, answer relevance, binary correctness against a gold fact),
   a quality profile by query type, a diagnosis matrix, manual error analysis, a gold-context run and
   a go/no-go verdict.

Target for a pilot: recall@5 >= 0.85 on answerable queries and correctness >= 0.90 overall, with a
consistent refusal on unanswerable ones.

## Results

All numbers are from the 66 labelled test queries: 56 with an answer in the knowledge base and 10 without.

**Targets.**

| target | goal | result | met |
|---|---|---|---|
| recall@5, queries with an answer | >= 0.85 | 0.938 | yes |
| correctness, all queries | >= 0.90 | 0.848 | no |
| queries without an answer: no invented facts, one refusal phrase | 10 of 10 | no invented facts in 10 of 10, the exact phrase in 8 of 10 | partly |

**Pipeline.** 181 raw sections are cleaned down to 130 documents and split into 300 chunks (recursive split
on paragraphs and sentences, 128 tokens, overlap 32, the document title prepended to the embedded text).
Dense search with `intfloat/multilingual-e5-base` in Qdrant returns 40 candidates above a 0.78 similarity
floor, the cross-encoder `BAAI/bge-reranker-v2-m3` keeps the top 5, and a local `Qwen3.5-4B` answers from
them or refuses with a fixed phrase.

**Retrieval, before and after the reranker** (56 queries; the top 5 chunks are mapped to their source
documents):

| | precision@5 | recall@5 | MRR |
|---|---|---|---|
| dense baseline | 0.339 | 0.705 | 0.701 |
| dense, 40 candidates + reranker | 0.407 | 0.938 | 0.867 |

The diagnosis was a ranking problem: 19 of the 24 missed documents sat at chunk ranks 6-20. The reranker
improves 18 queries and worsens 1. Its cost is about 650 ms per query instead of about 10 ms on an Apple
M4 Pro, and one more model of 568M parameters.

**Quality profile by query type.** Faithfulness and answer relevance are on a 1-5 scale, correctness is the
share of answers that convey the gold fact.

| | queries | recall@5 | faithfulness | answer relevance | correctness |
|---|---|---|---|---|---|
| all | 66 | 0.938 | 4.70 | 4.21 | 0.848 |
| exact | 15 | 1.000 | 4.67 | 4.87 | 0.867 |
| semantic | 25 | 0.940 | 4.60 | 4.72 | 0.840 |
| multi | 16 | 0.875 | 4.88 | 4.38 | 0.812 |
| gap | 10 | - | 4.70 | 1.70 | 0.900 |

Answer relevance of `gap` is low by design of the scale: a refusal does not answer the question.

**Where the errors come from.**

- 9 answers are wrong on queries with an answer. In 6 of them every relevant document is in the top 5, in 3
  a relevant document is missing.
- Manual review of the 11 worst cases: generation 6, retrieval 3, judge error 1, refusal in other words 1.
  In all six generation errors the chunk with the answer is in the prompt, in five at position 1.
- Gold-context run: with the labelled documents in the prompt instead of the retrieved chunks correctness is
  0.821 against 0.839 on the same 56 queries. A perfect retrieval does not raise correctness, so the
  bottleneck is the 4B generator.

**Verdict.** Not ready for an unattended pilot: 56 correct answers of 66 where the target needs 60, and some
wrong answers are confident statements. A pilot is possible with an operator in the loop.

**Recommendations, in order.**

1. Replace the generator with a stronger model and repeat the evaluation unchanged.
2. Fix the answer contract in the prompt with few-shot examples: a complete answer and the exact refusal
   phrase; check the change on queries that were not used to write the prompt.
3. Make the measurement reliable: an independent and stronger judge, a hand-labelled sample to measure the
   judge's error, a larger test set, a paired bootstrap interval for every before/after comparison.

Further retrieval work is deliberately not in the top three: the gold-context run limits its effect to two
or three queries. The next retrieval step would be splitting multi-part questions into sub-questions.

**Limits.** The judge is the generator model itself, and four judge errors were found by reading. One query
weighs 0.015, and a repeated run changed one answer at temperature 0, so correctness is known to about
+-0.03. The similarity floor, the number of candidates and the prompt were chosen on the same 66 queries;
there is no held-out set.

## Data

`data/filmoskop_kb.json` holds 181 raw knowledge-base sections (`doc_id`, `title`, `category`, `format`,
`raw`) and 66 labelled test queries. Each query has a `kind` (`exact` terms and error codes, `semantic`
paraphrases, `multi` needing two sections, `gap` with no answer in the base), the ids of all relevant
sections (`relevant_docs`) and a `gold` fact the answer must convey. Relevance is labelled at section level,
so the labels survive any re-chunking; retrieval metrics map found chunks back to their source sections.

The `data/` folder is not in git. The notebook writes its own artifacts there: the cleaned corpus, the
chunks and their vectors.

## How to run

Everything runs locally on a Mac (Apple M4 Pro, 24 GB, MPS); no GPU box is required.

1. Python 3.13 environment managed by uv from the repository root (`uv sync`). Key packages:
   `langchain-core`, `langchain-huggingface`, `langchain-qdrant`, `langchain-openai`,
   `langchain-text-splitters`, `qdrant-client`, `sentence-transformers`, `pandas`, `matplotlib`,
   `beautifulsoup4`, `pydantic`.
2. Put `filmoskop_kb.json` into `data/`.
3. Start a local OpenAI-compatible LLM server. The notebook uses `Qwen3.5-4B` (GGUF, Q4_K_M, about 2.7 GB)
   through llama.cpp on Metal:

   ```bash
   brew install llama.cpp
   llama-server -hf unsloth/Qwen3.5-4B-GGUF:Q4_K_M --host 127.0.0.1 --port 8080 -ngl 99 -c 4096
   curl http://localhost:8080/health
   ```

   The notebook talks to it through `ChatOpenAI` from `langchain-openai`
   (`base_url="http://localhost:8080/v1"`, `temperature=0`, thinking mode disabled via
   `chat_template_kwargs`). Another OpenAI-compatible server works by changing `LLM_URL`, provided it
   supports JSON-schema constrained output: the judge relies on it.
4. Embedder `intfloat/multilingual-e5-base` and reranker `BAAI/bge-reranker-v2-m3` run on MPS; weights are
   downloaded from Hugging Face on first use (`HF_TOKEN` is read from `.env`).
5. Run the notebook top to bottom. The two evaluation cells of stage 6 call the LLM a few hundred times and
   take about 15 minutes together.

Reproducibility: the bootstrap is seeded and the LLM runs at temperature 0. Two identical calls in a row
return the same text, but between full runs one answer of 56 changed, so the stage 6 metrics are
reproducible to within one query.
