# src/app.py
# ─────────────────────────────────────────────────────────────
# PURPOSE: Streamlit web UI for the CRAG legal contract analyser.
# Run with: streamlit run src/app.py
# ─────────────────────────────────────────────────────────────

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # ensure src/ imports work

import streamlit as st
from crag_pipeline import CRAGPipeline, PipelineResult


# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Legal CRAG Analyser",
    page_icon="⚖️",
    layout="wide",
)


# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .action-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .badge-generate  { background:#d1fae5; color:#065f46; }
    .badge-partial   { background:#fef3c7; color:#92400e; }
    .badge-websearch { background:#fee2e2; color:#991b1b; }
    .chunk-relevant   { border-left: 4px solid #10b981; padding-left:10px; margin:6px 0; }
    .chunk-irrelevant { border-left: 4px solid #ef4444; padding-left:10px; margin:6px 0; }
    .source-pill {
        display: inline-block;
        background: #ede9fe;
        color: #4c1d95;
        border-radius: 8px;
        padding: 2px 10px;
        font-size: 12px;
        margin: 2px;
    }
    .metric-box {
        background: #f8fafc;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 4px 0;
        border: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# ── Pipeline singleton (cached across reruns) ─────────────────
@st.cache_resource(show_spinner="Loading contract vectorstore...")
def load_pipeline() -> CRAGPipeline:
    return CRAGPipeline()


# ── Action badge helper ───────────────────────────────────────
def action_badge(action: str) -> str:
    labels = {
        "generate" : ("✓ GENERATE",  "badge-generate"),
        "partial"  : ("⚡ PARTIAL",   "badge-partial"),
        "websearch": ("🌐 WEBSEARCH", "badge-websearch"),
    }
    text, cls = labels.get(action, (action.upper(), "badge-generate"))
    return f'<span class="action-badge {cls}">{text}</span>'


# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("⚖️ Legal CRAG")
    st.caption("Precision-First Contract Analysis")
    st.divider()

    st.markdown("### How it works")
    st.markdown("""
1. **Retrieve** — find top-5 contract chunks
2. **Grade** — LLM scores each chunk
3. **Correct** — web search if retrieval is poor
4. **Generate** — grounded answer with citations
    """)

    st.divider()
    st.markdown("### CRAG Actions")
    st.markdown("""
- 🟢 **GENERATE** — local chunks sufficient
- 🟡 **PARTIAL** — chunks + web search
- 🔴 **WEBSEARCH** — web only (poor retrieval)
    """)

    st.divider()
    k_val = st.slider("Chunks to retrieve (K)", 3, 10, 5)
    st.caption("Higher K = more context, slower grading")

    st.divider()
    st.markdown("### Sample questions")
    sample_questions = [
        "What are the termination conditions?",
        "Who are the parties in this agreement?",
        "What are the payment obligations?",
        "What are the confidentiality clauses?",
        "What happens in case of breach?",
        "What is the governing law?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state["prefill_query"] = q


# ── Main area ─────────────────────────────────────────────────
st.title("⚖️ Legal Contract Analyser")
st.caption("Powered by Corrective RAG (CRAG) — answers grounded strictly in your contracts")
st.divider()

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(msg["content"], unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])

# Handle sidebar sample question prefill
prefill = st.session_state.pop("prefill_query", None)

# Chat input
query = st.chat_input("Ask a question about your contracts...") or prefill

if query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving → Grading → Generating..."):
            pipeline = load_pipeline()
            result: PipelineResult = pipeline.run(query, k=k_val)

        # ── Action badge ───────────────────────────────────────
        st.markdown(action_badge(result.action), unsafe_allow_html=True)

        # ── Answer ─────────────────────────────────────────────
        st.markdown(result.answer)

        # ── Metadata row ───────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        col1.metric("⏱ Time", f"{result.elapsed_sec}s")
        col2.metric("✓ Relevant chunks",
                    f"{len(result.relevant_chunks)}/{len(result.chunk_grades)}")
        col3.metric("🌐 Web used", "Yes" if result.used_web else "No")

        # ── Sources ────────────────────────────────────────────
        if result.sources:
            st.markdown("**Sources cited:**")
            pills = " ".join(
                f'<span class="source-pill">📄 {s}</span>'
                for s in result.sources
            )
            st.markdown(pills, unsafe_allow_html=True)

        # ── Chunk grading explainability panel ─────────────────
        with st.expander("🔍 View chunk grading details (CRAG transparency)"):
            st.caption(
                "These are the raw chunks retrieved from your contracts "
                "and how the grader scored each one."
            )
            for i, g in enumerate(result.chunk_grades):
                is_relevant = g["grade"] == "relevant"
                icon  = "✅" if is_relevant else "❌"
                cls   = "chunk-relevant" if is_relevant else "chunk-irrelevant"
                label = "RELEVANT" if is_relevant else "IRRELEVANT"

                st.markdown(
                    f'<div class="{cls}">'
                    f'<strong>{icon} Chunk {i+1} — {label}</strong><br>'
                    f'<small>📄 {g["source"]} | chunk index: {g["chunk"]}</small><br>'
                    f'<small>💬 {g["reason"]}</small>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"  View chunk {i+1} text"):
                    st.code(g["preview"], language=None)

        # Save to history
        response_md = (
            action_badge(result.action) + "\n\n" +
            result.answer
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_md,
        })