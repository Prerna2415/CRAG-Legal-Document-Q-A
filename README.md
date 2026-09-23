# Legal Advisor Q&A — Corrective RAG (CRAG)

A retrieval-augmented Q&A system for legal documents, built on **Corrective RAG (CRAG)** — instead of trusting whatever the vector store retrieves, a lightweight evaluator grades every retrieved chunk first, and only feeds the LLM context that actually clears the bar. When too few chunks clear it, the system corrects by falling back to a live web search (Tavily) instead of answering on weak context.

> ⚠️ **Not legal advice.** This system retrieves and summarizes information from the documents it's given. It does not replace a licensed attorney and should not be relied on for actual legal decisions.

---

## How it works

```
User Question
      │
      ▼
Vector Store Retrieval  (top-k chunks from the FAISS legal corpus)
      │
      ▼
Retrieval Evaluator  ──── grades each chunk: relevant / irrelevant  (Llama 3.2, structured output)
      │
      ▼
Relevant-chunk fraction determines the CRAG action:
      │
      ├── ≥ 50% relevant  ───────────────────────────► GENERATE  (local chunks only)
      │
      ├── some relevant, but < 50% ──────────────────► PARTIAL   (local chunks + Tavily web search)
      │
      └── 0% relevant ───────────────────────────────► WEBSEARCH (Tavily web search only)
                                                              │
                                                              ▼
                                                      Generate grounded,
                                                      citation-tagged answer
```

The core idea, following the original CRAG approach: retrieval isn't trusted blindly. A dedicated evaluator step scores every retrieved chunk before generation, and the fraction of chunks that pass the bar decides whether to generate directly, supplement with web search, or fall back to web search entirely — instead of letting the LLM hallucinate over irrelevant context. (Note: this implementation corrects via a web-search fallback rather than by rewriting the query and re-querying the vector store.)

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | [LangChain](https://www.langchain.com/) |
| LLM | Llama 3.2, served locally via [Ollama](https://ollama.com/) (`ChatOllama`) |
| Retrieval evaluation | Structured (Pydantic) LLM grading chain — binary relevant/irrelevant label + reason per chunk |
| External/backup search | [Tavily API](https://tavily.com/) (advanced search depth) |
| Vector store | [FAISS](https://github.com/facebookresearch/faiss), persisted locally to `vectorstore/` |
| Embeddings | `BAAI/bge-small-en-v1.5` via `langchain_huggingface.HuggingFaceEmbeddings` (CPU, normalized embeddings) |
| UI | [Streamlit](https://streamlit.io/) chat interface |

---

## Features

- **Document ingestion** — load and chunk legal documents (statutes, case summaries, contracts, etc.) into the vector store.
- **In-app upload** — add PDF/DOCX contracts straight from the Streamlit sidebar; uploaded files are saved to `data/contracts/`, chunked, and embedded into the existing FAISS index incrementally (no full re-build needed).
- **Grounded Q&A** — answers are generated only from retrieved context, not the model's parametric knowledge, to reduce hallucination on legal specifics.
- **Corrective retrieval loop** — a retrieval evaluator scores each retrieved chunk relevant/irrelevant; the fraction of relevant chunks decides whether to generate directly, blend in web search, or fall back to web search entirely, rather than answering on a low-confidence retrieval.
- **External search fallback (Tavily)** — automatically triggered (fully or partially) when the internal corpus doesn't sufficiently cover a query, using Tavily's advanced search depth.
- **Explainability panel** — the Streamlit UI shows the CRAG action taken, per-chunk grades with the LLM's stated reason, and elapsed time for each query.
- **Source attribution** — every context block passed to the LLM is labelled `[Source N: filename]`, the generation prompt requires citing `[Source N]` for each key claim, and the Streamlit UI surfaces the deduplicated source filenames as pills under each answer, plus a per-chunk grading panel (relevant/irrelevant + LLM's stated reason) for full transparency.

---

## Project Structure

```
.
├── data/
│   └── contracts/         # sample legal contracts (PDF) used to build the vectorstore
├── vectorstore/           # persisted FAISS index (index.faiss, index.pkl)
├── src/
│   ├── ingest.py          # load + clean + chunk PDFs/DOCX (RecursiveCharacterTextSplitter)
│   ├── retriever.py       # build/load the FAISS index, embed + retrieve top-k chunks
│   ├── grader.py          # DocumentGrader — grades chunks, decides the CRAG action
│   ├── generator.py       # ContractGenerator — grounded generation + Tavily web search
│   ├── crag_pipeline.py   # wires ingest → retrieve → grade → generate into one pipeline
│   └── app.py             # Streamlit chat UI — entry point
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone <repo-url>
cd <repo-name>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need [Ollama](https://ollama.com/) installed and running locally with the `llama3.2` model pulled:

```bash
ollama pull llama3.2
```

`CONTRACTS_DIR` (`src/ingest.py`) and `VECTORSTORE_DIR` (`src/retriever.py`) resolve relative to the repo root (`data/contracts/` and `vectorstore/`), so no path editing is needed after cloning.

### Environment variables

Create a `.env` file:

```
TAVILY_API_KEY=your_tavily_key
```

No API key is needed for the LLM or embeddings — Llama 3.2 runs locally via Ollama, and embeddings (`BAAI/bge-small-en-v1.5`) run locally via Hugging Face `sentence-transformers` on CPU.

### Run

```bash
# 1. Build the FAISS index (once, or whenever data/contracts/ changes)
python src/retriever.py --build

# 2. Launch the UI
streamlit run src/app.py
```

---

## Example

```
> What are the termination conditions in this contract?

[retriever] Retrieved 5 chunk(s) for query: 'What are the termination conditions...'
[grader] Grading 5 chunk(s)...
  [chunk 1] ✓ RELEVANT   — Describes conditions under which either party may terminate.
  [chunk 2] ✗ IRRELEVANT — This is a signature block with no termination content.
  ...
[grader] Result: 3/5 relevant (60%) → action = GENERATE

Answer:
"Either party may terminate this Agreement upon 30 days' written notice to the
other party [Source 1]. Termination is immediate in the event of a material
breach that remains uncured for 15 days after written notice [Source 2]."

Sources cited: 📄 service_agreement1.pdf
```

---

## Status

Core CRAG pipeline (retrieval → evaluation → conditional correction → generation) is working end-to-end; current work is on polish.
