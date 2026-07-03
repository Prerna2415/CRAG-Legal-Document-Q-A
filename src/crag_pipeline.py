# src/crag_pipeline.py
# ─────────────────────────────────────────────────────────────
# PURPOSE: Single entry point that wires all components together.
#   ingest → retrieve → grade → generate
#
# This is what app.py (Streamlit UI) will call.
# ─────────────────────────────────────────────────────────────

import time
from pathlib import Path
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from retriever  import load_vectorstore, retrieve_chunks, build_vectorstore
from grader     import DocumentGrader, CRAGAction
from generator  import ContractGenerator, generate_with_crag_action, GeneratorOutput


# ── Pipeline result dataclass ─────────────────────────────────
@dataclass
class PipelineResult:
    """Everything the UI needs to display a complete CRAG response."""
    query           : str
    answer          : str
    sources         : list[str]
    action          : str           # "generate" | "websearch" | "partial"
    used_web        : bool
    relevant_chunks : list[Document]
    irrelevant_chunks: list[Document]
    elapsed_sec     : float
    # Chunk-level grade details for the UI explainability panel
    chunk_grades    : list[dict] = field(default_factory=list)


# ── Main pipeline class ───────────────────────────────────────
class CRAGPipeline:
    """
    Orchestrates the full CRAG loop:
      1. Load vectorstore (once at startup)
      2. For each query:
         a. Retrieve top-K chunks
         b. Grade each chunk for relevance
         c. Decide action (GENERATE / WEBSEARCH / PARTIAL)
         d. Generate grounded answer
    """

    def __init__(self, vectorstore: FAISS = None):
        print("[pipeline] Initialising CRAG pipeline...")

        # Load vectorstore (from disk if not passed in)
        self.vectorstore = vectorstore or load_vectorstore()

        # Instantiate grader and generator once — reuse across queries
        self.grader    = DocumentGrader()
        self.generator = ContractGenerator()

        print("[pipeline] Ready.")

    def run(self, query: str, k: int = 5) -> PipelineResult:
        """
        Run the full CRAG pipeline for a single query.

        Args:
            query : the user's natural language question
            k     : number of chunks to retrieve (default 5)

        Returns:
            PipelineResult with answer, sources, grades, and metadata
        """
        start = time.time()
        print(f"\n{'='*60}")
        print(f"[pipeline] Query: {query}")
        print(f"{'='*60}")

        # ── Step 1: Retrieve ───────────────────────────────────
        chunks = retrieve_chunks(query, self.vectorstore, k=k)

        # ── Step 2: Grade ──────────────────────────────────────
        relevant, irrelevant, action = self.grader.grade_all(query, chunks)

        # ── Step 3: Generate ───────────────────────────────────
        output: GeneratorOutput = generate_with_crag_action(
            query           = query,
            relevant_chunks = relevant,
            action          = action.value,
            generator       = self.generator,
        )

        elapsed = round(time.time() - start, 2)

        # ── Step 4: Build chunk grade details for UI ───────────
        chunk_grades = []
        for doc in chunks:
            chunk_grades.append({
                "source"  : doc.metadata.get("source", "unknown"),
                "chunk"   : doc.metadata.get("chunk", "?"),
                "grade"   : doc.metadata.get("grade", "ungraded"),
                "reason"  : doc.metadata.get("grade_reason", ""),
                "preview" : doc.page_content[:200].strip(),
            })

        print(f"\n[pipeline] Done in {elapsed}s")

        return PipelineResult(
            query            = query,
            answer           = output.answer,
            sources          = output.sources,
            action           = action.value,
            used_web         = output.used_web,
            relevant_chunks  = relevant,
            irrelevant_chunks= irrelevant,
            elapsed_sec      = elapsed,
            chunk_grades     = chunk_grades,
        )


# ── Convenience function for app.py ──────────────────────────
# app.py creates the pipeline once at startup, then calls this
# for every user query.

_pipeline_instance: CRAGPipeline = None

def get_pipeline() -> CRAGPipeline:
    """Returns a singleton CRAGPipeline (initialised once)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = CRAGPipeline()
    return _pipeline_instance

def run_query(query: str) -> PipelineResult:
    """One-line entry point for app.py."""
    return get_pipeline().run(query)


# ── CLI test ──────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = CRAGPipeline()

    test_queries = [
        "What are the termination conditions in this contract?",
        "Who are the parties involved in this agreement?",
        "What are the payment obligations?",
    ]

    for query in test_queries:
        result = pipeline.run(query)

        print(f"\n── Answer ────────────────────────────────────────")
        print(result.answer)
        print(f"\n── Action taken : {result.action.upper()}")
        print(f"── Web used     : {result.used_web}")
        print(f"── Time         : {result.elapsed_sec}s")
        print(f"── Sources      : {', '.join(result.sources)}")
        print(f"\n── Chunk grades ──────────────────────────────────")
        for g in result.chunk_grades:
            icon = "✓" if g["grade"] == "relevant" else "✗"
            print(f"  {icon} [{g['source']} chunk {g['chunk']}] {g['reason'][:80]}")
        print()