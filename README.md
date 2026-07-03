# Legal Advisor Q&A — Corrective RAG (CRAG)

A retrieval-augmented Q&A system for legal documents, built on **Corrective RAG (CRAG)** — instead of trusting whatever the vector store retrieves, a lightweight evaluator grades retrieval quality first, and only feeds the LLM context that actually clears the bar. When it doesn't, the system self-corrects by rewriting the query and re-querying the vector store rather than answering on weak context.

> ⚠️ **Not legal advice.** This system retrieves and summarizes information from the documents it's given. It does not replace a licensed attorney and should not be relied on for actual legal decisions.

---

## How it works

```
User Question
      │
      ▼
Vector Store Retrieval  (top-k chunks from legal corpus)
      │
      ▼
Retrieval Evaluator  ──── grades each chunk: relevant / ambiguous / irrelevant
      │
      ├── Relevant enough ────────────────► Generate answer (LLaMA via LangChain)
      │
      └── Not relevant enough
                │
                ▼
        Query Rewriting  (LLM reformulates the question)
                │
                ▼
        Re-query Vector Store  ──► Generate answer
```

The core idea, following the original CRAG approach: retrieval isn't trusted blindly. A dedicated evaluator step scores retrieved chunks before generation, and low-confidence retrievals trigger a correction loop (query rewrite → re-retrieval) instead of letting the LLM hallucinate over irrelevant context.

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | [LangChain](https://www.langchain.com/) |
| LLM | LLaMA (via LangChain LLM integration) |
| Retrieval evaluation / query rewriting | LLM-based grading chain |
| External/backup search | [Tavily API](https://tavily.com/) |
| Vector store | *(fill in: FAISS / Chroma / etc.)* |
| Embeddings | *(fill in: which embedding model)* |

---

## Features

- **Document ingestion** — load and chunk legal documents (statutes, case summaries, contracts, etc.) into the vector store.
- **Grounded Q&A** — answers are generated only from retrieved context, not the model's parametric knowledge, to reduce hallucination on legal specifics.
- **Corrective retrieval loop** — a retrieval evaluator scores each retrieved chunk; low-quality retrievals trigger automatic query rewriting and re-querying against the vector store rather than a low-confidence answer.
- **External search fallback (Tavily)** — available as a supplementary source when the internal corpus doesn't sufficiently cover a query.
- **Source attribution** — *(fill in: does it cite which document/section an answer came from?)*

---

## Project Structure

```
.
├── ingestion/          # document loading, chunking, embedding
├── retrieval/          # vector store queries + retrieval evaluator
├── correction/         # query rewriting + re-query loop
├── generation/         # LLM answer generation
├── app.py / main.py    # entry point
├── requirements.txt
└── README.md
```
*(adjust to match your actual repo layout)*

---

## Setup

```bash
git clone <repo-url>
cd <repo-name>
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file:

```
TAVILY_API_KEY=your_tavily_key
# add LLaMA/model-hosting credentials as needed, e.g.:
# HUGGINGFACEHUB_API_TOKEN=your_token
```

### Run

```bash
python app.py
```

*(replace with your actual run command — e.g. `streamlit run app.py` if there's a UI)*

---

## Example

```
> What are the notice period requirements for terminating a commercial lease?

Retrieval evaluator: chunks scored below threshold on first pass
→ Rewriting query: "commercial lease termination notice period requirements"
→ Re-querying vector store...

Answer: [grounded answer citing retrieved clauses]
```

---

## Status

Core CRAG pipeline (retrieval → evaluation → conditional correction → generation) is working end-to-end; current work is on polish — *(fill in what's left: e.g. answer citation formatting, evaluator threshold tuning, latency, UI)*.
