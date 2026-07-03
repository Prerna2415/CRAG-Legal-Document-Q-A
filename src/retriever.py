# src/retriever.py
# ─────────────────────────────────────────────────────────────
# PURPOSE: Two jobs:
#   1. BUILD  — embed all chunks and save a FAISS index to disk
#   2. QUERY  — load the saved index and retrieve top-K chunks
#               most semantically similar to a user's question
# ─────────────────────────────────────────────────────────────

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Load OPENAI_API_KEY from .env
load_dotenv()


# ── Config ────────────────────────────────────────────────────
VECTORSTORE_DIR = Path("C:\\Users\\Prerna\\OneDrive\\Documents\\GENAICRAG\\vectorstore")   # where the FAISS index is saved
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # cheapest OpenAI embedding model
                                            # swap to "text-embedding-3-large"
                                            # for higher accuracy if needed
TOP_K = 5   # how many chunks to retrieve per query
            # 5 is a good balance — too few misses context,
            # too many floods the grader with noise


# ── Build ─────────────────────────────────────────────────────

def build_vectorstore(chunks: list[Document]) -> FAISS:
    """
    Embeds all chunks using OpenAI and saves the FAISS index locally.

    Call this ONCE after ingesting your contracts.
    Re-run only when you add new contracts.

    Args:
        chunks: list of Document objects from ingest.load_and_chunk_contracts()

    Returns:
        The in-memory FAISS vectorstore object.
    """
    print(f"[retriever] Embedding {len(chunks)} chunks with '{EMBEDDING_MODEL}'...")
    print("[retriever] This may take a moment depending on document size.")

    # OpenAIEmbeddings batches requests automatically
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # FAISS.from_documents:
    #   - calls embeddings.embed_documents() on all chunk texts
    #   - builds an in-memory FAISS index
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist to disk so we don't re-embed on every run
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))

    print(f"[retriever] Vectorstore saved to '{VECTORSTORE_DIR}/'")
    return vectorstore


# ── Load ──────────────────────────────────────────────────────

def load_vectorstore() -> FAISS:
    """
    Loads the previously saved FAISS index from disk.

    Raises FileNotFoundError if build_vectorstore() hasn't been run yet.
    """
    index_file = VECTORSTORE_DIR / "index.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{VECTORSTORE_DIR}/'.\n"
            "Run build_vectorstore() first (or run: python src/retriever.py --build)."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # allow_dangerous_deserialization=True is required by LangChain
    # for loading local FAISS indexes (it's safe here since WE saved it)
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print(f"[retriever] Loaded vectorstore from '{VECTORSTORE_DIR}/'")
    return vectorstore


# ── Query ─────────────────────────────────────────────────────

def retrieve_chunks(query: str, vectorstore: FAISS = None, k: int = TOP_K) -> list[Document]:
    """
    Retrieves the top-K most relevant chunks for a given query.

    Args:
        query      : the user's natural language question
        vectorstore: a loaded FAISS vectorstore (loads from disk if None)
        k          : number of chunks to return (default: TOP_K)

    Returns:
        List of Document objects, ranked by cosine similarity (best first).
        Each Document has:
          - page_content : the chunk text
          - metadata     : { source, page, chunk }
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    # similarity_search embeds the query on the fly and finds nearest neighbors
    results = vectorstore.similarity_search(query, k=k)

    print(f"[retriever] Retrieved {len(results)} chunk(s) for query: '{query[:60]}...'")
    return results


def retrieve_with_scores(query: str, vectorstore: FAISS = None, k: int = TOP_K):
    """
    Same as retrieve_chunks() but also returns similarity scores.
    Useful for debugging — scores closer to 1.0 = more relevant.

    Returns:
        List of (Document, score) tuples.
    """
    if vectorstore is None:
        vectorstore = load_vectorstore()

    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\n[retriever] Top-{k} results for: '{query}'")
    for i, (doc, score) in enumerate(results):
        print(f"  [{i+1}] score={score:.4f} | "
              f"source={doc.metadata.get('source')} | "
              f"chunk={doc.metadata.get('chunk')}")
        print(f"       preview: {doc.page_content[:120].strip()}...")
    return results


# ── CLI entry point ───────────────────────────────────────────
# Run this file directly to build or test the vectorstore.
#
# Build:  python src/retriever.py --build
# Query:  python src/retriever.py --query "What is the termination clause?"

if __name__ == "__main__":
    import sys
    from ingest import load_and_chunk_contracts

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python src/retriever.py --build")
        print("  python src/retriever.py --query \"your question here\"")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "--build":
        chunks = load_and_chunk_contracts()
        build_vectorstore(chunks)

    elif mode == "--query":
        if len(sys.argv) < 3:
            print("Provide a query string. Example:")
            print('  python src/retriever.py --query "What are the payment terms?"')
            sys.exit(1)
        query = sys.argv[2]
        retrieve_with_scores(query)

    else:
        print(f"Unknown mode: {mode}. Use --build or --query.")