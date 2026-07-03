# src/generator.py
# ─────────────────────────────────────────────────────────────
# PURPOSE: Takes verified context (relevant chunks + optional
#          web results) and generates a precise, citation-grounded
#          answer using GPT-4o.
#
#          Also handles the Tavily web search fallback when the
#          grader decides retrieval quality is too poor.
# ─────────────────────────────────────────────────────────────

import os
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# ── Config ────────────────────────────────────────────────────
GENERATOR_MODEL = "llama3.2"         
MAX_CONTEXT_DOCS = 6                # max chunks fed into the prompt
                                    # more = richer context but higher cost
TAVILY_MAX_RESULTS = 3              # web search results to fetch


# ── Output dataclass ──────────────────────────────────────────
@dataclass
class GeneratorOutput:
    """Structured result returned by the generator."""
    answer     : str          # the final answer text
    sources    : list[str]    # list of source filenames / URLs used
    used_web   : bool         # True if Tavily web search was used
    context_used: str         # the full context string fed to GPT-4o


# ── Web search (Tavily fallback) ──────────────────────────────

def web_search(query: str) -> list[Document]:
    """
    Calls Tavily search API to retrieve web results.
    Used when the grader decides local chunks are insufficient.

    Returns a list of Documents with:
      - page_content : snippet text from the web result
      - metadata     : { source: URL, title: page title }
    """
    try:
        from tavily import TavilyClient
        client  = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        results = client.search(
            query=query,
            max_results=TAVILY_MAX_RESULTS,
            search_depth="advanced",    # deeper crawl for legal queries
        )
        docs = []
        for r in results.get("results", []):
            docs.append(Document(
                page_content=r.get("content", ""),
                metadata={
                    "source": r.get("url", "web"),
                    "title" : r.get("title", ""),
                }
            ))
        print(f"[generator] Web search returned {len(docs)} result(s).")
        return docs

    except Exception as e:
        print(f"[generator] Web search failed: {e}")
        print("[generator] Proceeding without web results.")
        return []


# ── Context builder ───────────────────────────────────────────

def build_context(
    relevant_chunks : list[Document],
    web_docs        : list[Document] = None,
) -> tuple[str, list[str]]:
    """
    Merges relevant chunks and web results into a single
    labelled context string for the prompt.

    Each source is clearly labelled so GPT-4o can cite it.

    Returns:
        (context_string, list_of_source_names)
    """
    web_docs = web_docs or []
    all_docs = relevant_chunks + web_docs
    all_docs = all_docs[:MAX_CONTEXT_DOCS]   # hard cap

    context_parts = []
    sources       = []

    for i, doc in enumerate(all_docs):
        src   = doc.metadata.get("source", f"source_{i+1}")
        title = doc.metadata.get("title", "")

        # Label each block clearly — helps GPT-4o cite correctly
        label = f"[Source {i+1}: {src}]"
        if title:
            label += f" {title}"

        context_parts.append(f"{label}\n{doc.page_content.strip()}")
        sources.append(src)

    context_str = "\n\n---\n\n".join(context_parts)
    return context_str, sources


# ── Generator prompt ──────────────────────────────────────────
# The prompt is tuned for legal contract analysis:
#   - instructs the model to cite sources explicitly
#   - forbids fabrication ("if not in context, say so")
#   - asks for structured, clause-level precision

GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a precise legal contract analysis assistant.
Your job is to answer questions about contracts using ONLY the
provided context. You must never fabricate clauses, dates,
parties, or obligations that are not explicitly stated in the context.

Rules:
1. Base your answer strictly on the provided context.
2. Cite the source for every key claim using [Source N] notation.
3. If the answer is not found in the context, say:
   "This information was not found in the provided contract documents."
4. Be precise and concise. Use bullet points for multiple clauses.
5. If quoting a clause directly, use quotation marks and cite the source.
6. Do not infer, assume, or extrapolate beyond what is written."""
    ),
    (
        "human",
        """Question: {query}

Context:
{context}

Provide a precise, citation-grounded answer."""
    ),
])


# ── Generator class ───────────────────────────────────────────

class ContractGenerator:
    """
    Generates grounded answers from verified context.
    Handles both pure-retrieval and web-augmented generation.
    """

    def __init__(self):
        llm = ChatOllama(model=GENERATOR_MODEL, temperature=0)
        self.chain = GENERATOR_PROMPT | llm | StrOutputParser()

    def generate(
        self,
        query           : str,
        relevant_chunks : list[Document],
        web_docs        : list[Document] = None,
    ) -> GeneratorOutput:
        """
        Generate a final answer from verified context.

        Args:
            query           : the user's original question
            relevant_chunks : chunks graded as relevant by grader.py
            web_docs        : optional web search results (from Tavily)

        Returns:
            GeneratorOutput with answer, sources, and metadata
        """
        web_docs  = web_docs or []
        used_web  = len(web_docs) > 0

        # ── Build context string ───────────────────────────────
        context, sources = build_context(relevant_chunks, web_docs)

        if not context.strip():
            return GeneratorOutput(
                answer="No relevant context was found to answer this question.",
                sources=[],
                used_web=used_web,
                context_used="",
            )

        print(f"[generator] Generating answer from "
              f"{len(relevant_chunks)} chunk(s)"
              + (f" + {len(web_docs)} web result(s)" if used_web else "")
              + "...")

        # ── Call GPT-4o ────────────────────────────────────────
        answer = self.chain.invoke({
            "query"  : query,
            "context": context,
        })

        print(f"[generator] Answer generated ({len(answer)} chars).")

        return GeneratorOutput(
            answer      = answer,
            sources     = list(dict.fromkeys(sources)),  # deduplicated
            used_web    = used_web,
            context_used= context,
        )


# ── Convenience wrapper ───────────────────────────────────────
# Used by crag_pipeline.py — handles all three CRAG actions.

def generate_with_crag_action(
    query           : str,
    relevant_chunks : list[Document],
    action          : str,               # CRAGAction value string
    generator       : ContractGenerator = None,
) -> GeneratorOutput:
    """
    Wrapper that handles all three CRAG actions:
      GENERATE  → use relevant_chunks only
      WEBSEARCH → discard chunks, use web only
      PARTIAL   → use relevant_chunks + web results

    Args:
        query           : user's question
        relevant_chunks : chunks graded relevant
        action          : one of "generate", "websearch", "partial"
        generator       : reuse an existing ContractGenerator instance
    """
    gen = generator or ContractGenerator()

    if action == "generate":
        # ── Good retrieval: use chunks directly ─────────────────
        print("[generator] Action: GENERATE — using local chunks only.")
        return gen.generate(query, relevant_chunks)

    elif action == "websearch":
        # ── Poor retrieval: fall back entirely to web ────────────
        print("[generator] Action: WEBSEARCH — querying Tavily.")
        web_docs = web_search(query)
        return gen.generate(query, [], web_docs)

    elif action == "partial":
        # ── Mixed: supplement good chunks with web ───────────────
        print("[generator] Action: PARTIAL — combining chunks + web.")
        web_docs = web_search(query)
        return gen.generate(query, relevant_chunks, web_docs)

    else:
        raise ValueError(f"Unknown CRAG action: '{action}'")


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    from retriever import load_vectorstore, retrieve_chunks
    from grader    import DocumentGrader

    vs      = load_vectorstore()
    grader  = DocumentGrader()
    gen     = ContractGenerator()

    query   = "What are the termination conditions in this contract?"
    chunks  = retrieve_chunks(query, vs)

    relevant, _, action = grader.grade_all(query, chunks)

    result = generate_with_crag_action(
        query           = query,
        relevant_chunks = relevant,
        action          = action.value,
        generator       = gen,
    )

    print("\n── Final Answer ──────────────────────────────────")
    print(result.answer)
    print("\n── Sources used ──────────────────────────────────")
    for s in result.sources:
        print(f"  • {s}")
    print(f"\n── Web search used: {result.used_web}")