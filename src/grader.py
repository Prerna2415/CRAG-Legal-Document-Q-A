# src/grader.py
# ─────────────────────────────────────────────────────────────
# PURPOSE: The heart of CRAG.
#   1. Grade each retrieved chunk as "relevant" or "irrelevant"
#      to the user's query using GPT-4.
#   2. Decide which action to take:
#        GENERATE  → enough relevant chunks found, proceed
#        WEBSEARCH → too many irrelevant chunks, fall back to Tavily
#        PARTIAL   → mix of both, use good chunks + web search
# ─────────────────────────────────────────────────────────────

import os
from enum import Enum
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()


# ── Config ────────────────────────────────────────────────────
GRADER_MODEL = "llama3.2"  # cheap + fast for binary grading
                                   # no need to use full gpt-4 here
RELEVANCE_CUTOFF = 0.5             # fraction of chunks that must be
                                   # relevant to skip web search
                                   # e.g. 0.5 = at least 50% relevant


# ── Grade label enum ──────────────────────────────────────────
class GradeLabel(str, Enum):
    RELEVANT   = "relevant"
    IRRELEVANT = "irrelevant"


# ── Action enum ───────────────────────────────────────────────
class CRAGAction(str, Enum):
    GENERATE   = "generate"    # use retrieved chunks as-is
    WEBSEARCH  = "websearch"   # discard chunks, search the web
    PARTIAL    = "partial"     # use good chunks AND web search


# ── Pydantic schema for structured GPT-4 output ───────────────
# LangChain's .with_structured_output() forces the LLM to return
# a valid instance of this schema — no free-text parsing needed.
class GradeChunk(BaseModel):
    """Relevance grade for a single retrieved chunk."""
    label: GradeLabel = Field(
        description="'relevant' if the chunk contains information "
                    "useful to answer the query, 'irrelevant' otherwise."
    )
    reason: str = Field(
        description="One sentence explaining the grade decision."
    )


# ── Grader prompt ─────────────────────────────────────────────
# The prompt is deliberately strict:
#   - We tell the model it is grading LEGAL CONTRACT chunks
#   - We ask for a binary label + a short reason
#   - We explicitly forbid hedging / "maybe" answers
GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise legal document relevance grader.
Your job is to decide whether a retrieved contract chunk is useful
for answering a specific legal question.

Rules:
- Label as 'relevant' if the chunk contains ANY information that
  directly or partially answers the question (e.g. definitions,
  clauses, obligations, dates, parties, conditions).
- Label as 'irrelevant' if the chunk is a table of contents entry,
  a signature block, a blank page, boilerplate header/footer, or
  completely off-topic.
- Be strict. A chunk about indemnity is NOT relevant to a question
  about termination conditions.
- Give a one-sentence reason for your decision."""
    ),
    (
        "human",
        """Question: {query}

Retrieved chunk:
\"\"\"
{chunk_text}
\"\"\"

Grade this chunk."""
    ),
])


# ── Grader class ──────────────────────────────────────────────

class DocumentGrader:
    """
    Grades a list of retrieved chunks against a user query.
    Uses GPT-4o-mini with structured output (no text parsing).
    """

    def __init__(self):
        llm = ChatOllama(model=GRADER_MODEL, temperature=0)
        # .with_structured_output() binds the LLM to return a GradeChunk object
        self.grader_chain = GRADER_PROMPT | llm.with_structured_output(GradeChunk)

    def grade_chunk(self, query: str, chunk: Document) -> GradeChunk:
        """
        Grade a single chunk against the query.

        Returns a GradeChunk with:
          .label  → GradeLabel.RELEVANT or GradeLabel.IRRELEVANT
          .reason → one-sentence explanation
        """
        result = self.grader_chain.invoke({
            "query"     : query,
            "chunk_text": chunk.page_content[:1500],  # cap to avoid token waste
        })
        return result

    def grade_all(
        self,
        query : str,
        chunks: list[Document],
    ) -> tuple[list[Document], list[Document], CRAGAction]:
        """
        Grade all retrieved chunks and decide on a CRAG action.

        Args:
            query  : the user's question
            chunks : list of retrieved Document objects

        Returns:
            (relevant_chunks, irrelevant_chunks, action)

            action is one of:
              CRAGAction.GENERATE   → use relevant_chunks directly
              CRAGAction.WEBSEARCH  → all chunks bad, search web
              CRAGAction.PARTIAL    → mixed, use both sources
        """
        relevant_chunks   = []
        irrelevant_chunks = []

        print(f"\n[grader] Grading {len(chunks)} chunk(s) for: '{query[:60]}'")

        for i, chunk in enumerate(chunks):
            grade = self.grade_chunk(query, chunk)

            # Attach grade metadata to the chunk for traceability
            chunk.metadata["grade"]        = grade.label.value
            chunk.metadata["grade_reason"] = grade.reason

            if grade.label == GradeLabel.RELEVANT:
                relevant_chunks.append(chunk)
                print(f"  [chunk {i+1}] ✓ RELEVANT   — {grade.reason[:80]}")
            else:
                irrelevant_chunks.append(chunk)
                print(f"  [chunk {i+1}] ✗ IRRELEVANT — {grade.reason[:80]}")

        # ── Decide CRAG action ─────────────────────────────────
        total    = len(chunks)
        n_rel    = len(relevant_chunks)
        rel_frac = n_rel / total if total > 0 else 0

        if n_rel == 0:
            # No relevant chunks at all → must go to web
            action = CRAGAction.WEBSEARCH
        elif rel_frac >= RELEVANCE_CUTOFF:
            # Majority relevant → generate directly
            action = CRAGAction.GENERATE
        else:
            # Some relevant but not enough → supplement with web
            action = CRAGAction.PARTIAL

        print(f"\n[grader] Result: {n_rel}/{total} relevant "
              f"({rel_frac:.0%}) → action = {action.value.upper()}")

        return relevant_chunks, irrelevant_chunks, action


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    from retriever import load_vectorstore, retrieve_chunks

    # Load vectorstore (must have run --build first)
    vs     = load_vectorstore()
    grader = DocumentGrader()

    query  = "What are the termination conditions in this contract?"
    chunks = retrieve_chunks(query, vs)

    relevant, irrelevant, action = grader.grade_all(query, chunks)

    print("\n── Relevant chunks preview ───────────────────────")
    for chunk in relevant:
        print(f"\n  Source : {chunk.metadata['source']}")
        print(f"  Reason : {chunk.metadata['grade_reason']}")
        print(f"  Text   : {chunk.page_content[:200]}...")