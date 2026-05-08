"""
rag_app.py — Streamlit chat interface for NBC Part 4 RAG

Run:
  streamlit run rag_app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="NBC Part 4 Assistant",
    page_icon="🏗️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading embedding model and ChromaDB…")
def load_engine():
    import rag_query
    return rag_query


engine = load_engine()

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "api_history" not in st.session_state:
    st.session_state.api_history: list[dict] = []


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏗️ NBC Part 4 RAG")
    st.caption("National Building Code of Canada — Structural Design")
    st.divider()

    st.subheader("Retrieval settings")
    engine.TOP_K_SENTENCES = st.slider("Sentences retrieved (k)", 3, 10, engine.TOP_K_SENTENCES)
    engine.TOP_K_TABLES    = st.slider("Tables retrieved (k)", 1, 5,  engine.TOP_K_TABLES)

    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages    = []
        st.session_state.api_history = []
        st.rerun()

    st.divider()
    st.subheader("Try a sample question")
    samples = [
        "What loads must be considered in building design?",
        "What are the importance categories for buildings?",
        "How is the specified snow load calculated?",
        "What are the wind load requirements for low-rise buildings?",
        "What are the earthquake design requirements?",
        "What are the live load requirements for floors?",
    ]
    for s in samples:
        if st.button(s, use_container_width=True, key=f"sample_{hash(s)}"):
            st.session_state["_prefill"] = s
            st.rerun()

    st.divider()
    st.caption("565 sentence chunks · 35 tables")
    st.caption("Embed: all-MiniLM-L6-v2")
    st.caption("LLM: claude-sonnet-4-6")


# ── source expander helper ────────────────────────────────────────────────────
def render_sources(s_hits: list[dict], t_hits: list[dict]) -> None:
    if s_hits:
        st.markdown("**Sentence hits**")
        for h in s_hits:
            dist        = h.get("distance", 0)
            sid         = h.get("sentence_id", "")
            sub         = h.get("subsection", "")
            txt         = h.get("original_text", "")[:200]
            needs_review = h.get("needs_review", "False") == "True"
            badge       = " ⚠️" if needs_review else ""
            st.markdown(
                f"- **[{sid}]**{badge} `dist={dist:.3f}` — *{sub}*  \n"
                f"  _{txt}…_"
            )
    if t_hits:
        st.markdown("**Table hits**")
        for h in t_hits:
            dist = h.get("distance", 0)
            tid  = h.get("table_id", "")
            ttl  = h.get("table_title", "")[:70]
            st.markdown(f"- **[Table {tid}]** `dist={dist:.3f}` — {ttl}")


# ── main chat area ────────────────────────────────────────────────────────────
st.title("NBC Part 4 – Structural Design Assistant")
st.caption("Ask questions about structural loads, design requirements, and code provisions.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Retrieved sources", expanded=False):
                render_sources(
                    msg["sources"].get("sentences", []),
                    msg["sources"].get("tables", []),
                )


# ── input handling ────────────────────────────────────────────────────────────
prefill    = st.session_state.pop("_prefill", "")
user_input = st.chat_input("Ask about NBC Part 4…")

# sample-button click takes priority over typed input
if prefill and not user_input:
    user_input = prefill

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            ans, s_hits, t_hits = engine.answer(
                user_input,
                history=st.session_state.api_history,
            )

        st.markdown(ans)

        with st.expander("📚 Retrieved sources", expanded=False):
            render_sources(s_hits, t_hits)

    sources = {"sentences": s_hits, "tables": t_hits}
    st.session_state.messages.append({
        "role":    "assistant",
        "content": ans,
        "sources": sources,
    })
    # keep API history compact — only question + answer, no context blobs
    st.session_state.api_history.extend([
        {"role": "user",      "content": user_input},
        {"role": "assistant", "content": ans},
    ])
