"""
Streamlit Web UI for Customer Support RAG Bot.

Provides a clean, interactive chat interface with session history,
expandable source provenance badges, and sidebar retrieval settings.
"""

import streamlit as st
from config import settings
from rag import CustomerSupportRAGChain, FALLBACK_RESPONSE
from index import is_typesense_alive

st.set_page_config(
    page_title="Customer Support AI - RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize RAG chain in session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = CustomerSupportRAGChain()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "👋 Hello! I am your Customer Support AI Assistant. "
            "Ask me anything about deploying, configuring secrets, caching, or optimizing your applications!"
        ),
        "sources": []
    }]

# ==============================================================================
# Sidebar Configurations
# ==============================================================================
with st.sidebar:
    st.title("⚙️ RAG Engine Settings")
    st.markdown("---")

    is_alive = is_typesense_alive()
    if is_alive:
        st.success("🟢 Typesense Server: Connected")
    else:
        st.warning("🟡 Typesense Server: In-Memory Fallback")

    st.markdown("### Retrieval Controls")
    search_mode = st.selectbox(
        "Search Mode",
        options=["hybrid", "vector", "keyword"],
        index=0,
        help="Hybrid combines BM25 keyword matching with dense neural vectors."
    )

    top_k = st.slider(
        "Top-K Chunks",
        min_value=1,
        max_value=8,
        value=4,
        step=1,
        help="Number of documentation passages retrieved."
    )

    hybrid_alpha = st.slider(
        "Hybrid Alpha (Vector vs Keyword)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="0.0 = 100% Keyword, 1.0 = 100% Vector, 0.7 = Recommended"
    )

    st.markdown("---")
    st.markdown("### Model Details")
    st.info(f"**Embedding**: `{settings.embedding_model_name}`\n\n**LLM**: `{settings.llm_model}`")

    if st.button("🧹 Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

# ==============================================================================
# Main Chat Area
# ==============================================================================
st.title("🤖 Customer Support Knowledge Assistant")
st.caption("Powered by Typesense Hybrid Search, LangChain, and Strict Grounded Generation.")

# Render message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Verified Sources ({len(msg['sources'])})"):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**[{idx}] [{src['title']}]({src['url']})**")
                    if src.get("heading_path"):
                        st.caption(f"Section: `{src['heading_path']}`")

# Handle user input
if prompt := st.chat_input("Ask a support question (e.g. 'How do I add dependencies?')..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Typesense and synthesizing grounded response..."):
            result = st.session_state.rag_chain.query(
                question=prompt,
                search_mode=search_mode,
                top_k=top_k,
                alpha=hybrid_alpha
            )

            answer = result["answer"]
            sources = result.get("sources", [])
            is_fallback = result.get("is_fallback", False)

            if is_fallback:
                st.warning("⚠️ Query escalated to support team.")

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Verified Sources ({len(sources)})"):
                    for idx, src in enumerate(sources, 1):
                        st.markdown(f"**[{idx}] [{src['title']}]({src['url']})**")
                        if src.get("heading_path"):
                            st.caption(f"Section: `{src['heading_path']}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
