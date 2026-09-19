"""
Chainlit Conversational UI for Customer Support RAG Bot.

INTERVIEW RATIONALE & DESIGN CHOICES:
-------------------------------------
1. Why Chainlit for Conversational RAG?
   - Chainlit is purpose-built for LLM and RAG applications, featuring native support
     for intermediate reasoning steps (`cl.Step`), retrieved context element sidebars,
     and chat profile switching.

2. Transparent Retrieval Inspection (Explainability):
   - Using `cl.Step("Typesense Retrieval")`, users and evaluators can inspect the exact
     chunks retrieved from Typesense, their similarity scores, and section hierarchy
     before the final response is synthesized.

3. Dynamic Search Profiles:
   - Allows live toggling between Hybrid Search (BM25 + Dense Vectors), Vector Search,
     and Keyword Search to demonstrate retrieval trade-offs during live demos.
"""

import os
from typing import Dict, List, Optional
import chainlit as cl
from chainlit.input_widget import Select, Slider, Switch

from config import settings
from rag import CustomerSupportRAGChain, FALLBACK_RESPONSE
from index import TypesenseSearcher, is_typesense_alive


# Global RAG chain instance
rag_pipeline = CustomerSupportRAGChain()


@cl.set_starters
async def set_starters():
    """Suggested starter questions for quick demonstration."""
    return [
        cl.Starter(
            label="Secrets Management",
            message="How do I manage secrets in Streamlit Community Cloud?",
            icon="🔒"
        ),
        cl.Starter(
            label="Caching Behavior",
            message="What is the difference between st.cache_data and st.cache_resource?",
            icon="⚡"
        ),
        cl.Starter(
            label="Custom Themes",
            message="How do I configure custom theme colors and borders?",
            icon="🎨"
        ),
        cl.Starter(
            label="Out-of-Domain Test",
            message="How to bake a chocolate cake with frosting?",
            icon="🍰"
        ),
    ]


@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session and send welcome message."""
    # Check Typesense connection
    typesense_status = "🟢 Connected (Local Typesense Server)" if is_typesense_alive() else "🟡 In-Memory Fallback Engine"

    # Configure Chat Settings Sidebar
    chat_settings = await cl.ChatSettings([
        Select(
            id="search_mode",
            label="Retrieval Search Mode",
            values=["hybrid", "vector", "keyword"],
            initial_index=0,
            description="Hybrid (BM25 + Vector) is recommended for customer support"
        ),
        Slider(
            id="top_k",
            label="Top-K Chunks to Retrieve",
            initial=4,
            min=1,
            max=8,
            step=1,
            description="Number of context passages passed to LLM"
        ),
        Slider(
            id="hybrid_alpha",
            label="Hybrid Alpha (Vector vs Keyword)",
            initial=0.7,
            min=0.0,
            max=1.0,
            step=0.05,
            description="0.0 = Pure Keyword (BM25), 1.0 = Pure Vector (Semantic)"
        )
    ]).send()

    cl.user_session.set("settings", {
        "search_mode": "hybrid",
        "top_k": 4,
        "hybrid_alpha": 0.7
    })

    welcome_msg = (
        "### 👋 Welcome to the Customer Support RAG Assistant\n\n"
        "I am an AI support assistant indexed over official product documentation using **Typesense Hybrid Search** and **LangChain**.\n\n"
        f"**Engine Status**: `{typesense_status}` | **Embedding**: `{settings.embedding_model_name}`\n\n"
        "**Features**:\n"
        "- 🎯 **Zero-Hallucination Grounding**: Answers are strictly backed by retrieved documentation.\n"
        "- 🔗 **Verified Provenance**: Clickable citations with direct article links.\n"
        "- 🛡️ **Support Escalation**: Automatically triggers *'I don't know, please contact support'* for unverified questions.\n\n"
        "Select a starter prompt below or type your technical support question!"
    )

    await cl.Message(content=welcome_msg).send()


@cl.on_settings_update
async def setup_agent(settings_dict):
    """Handle settings updates from the UI sidebar."""
    cl.user_session.set("settings", settings_dict)
    await cl.Message(
        content=f"⚙️ Search configuration updated: Mode=`{settings_dict.get('search_mode')}`, Top-K=`{settings_dict.get('top_k')}`, Alpha=`{settings_dict.get('hybrid_alpha')}`"
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Process incoming user message and generate grounded response."""
    user_query = message.content.strip()
    session_settings = cl.user_session.get("settings", {
        "search_mode": "hybrid",
        "top_k": 4,
        "hybrid_alpha": 0.7
    })

    search_mode = session_settings.get("search_mode", "hybrid")
    top_k = int(session_settings.get("top_k", 4))
    hybrid_alpha = float(session_settings.get("hybrid_alpha", 0.7))

    # 1. Retrieval Step (Visualized in Chainlit Step drawer)
    async with cl.Step(name=f"Typesense Retrieval ({search_mode.capitalize()} Search)") as step:
        step.input = f"Query: {user_query} (Top-K: {top_k}, Alpha: {hybrid_alpha})"
        
        # Execute query synchronously via pipeline
        rag_result = rag_pipeline.query(
            question=user_query,
            search_mode=search_mode,
            top_k=top_k,
            alpha=hybrid_alpha
        )

        retrieved_chunks = rag_result.get("retrieved_chunks", [])
        sources = rag_result.get("sources", [])

        # Display retrieved chunks in step details
        step_output_lines = [f"Retrieved **{len(retrieved_chunks)}** relevant passages:\n"]
        for rank, chunk in enumerate(retrieved_chunks, 1):
            score_str = f"Score: {chunk.get('score', 0):.4f}" if chunk.get("score") is not None else ""
            step_output_lines.append(
                f"- **[{rank}] {chunk.get('title')}** ({chunk.get('heading_path')}) {score_str}\n"
                f"  *URL*: {chunk.get('url')}\n"
            )
        step.output = "\n".join(step_output_lines)

    # 2. Prepare Context Elements (Side Panel in Chainlit)
    elements = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        content = chunk.get("raw_content", chunk.get("content", ""))
        title = chunk.get("title", f"Passage {idx}")
        elements.append(
            cl.Text(
                name=f"Source {idx}: {title}",
                content=f"**Source URL**: {chunk.get('url')}\n**Section**: {chunk.get('heading_path')}\n\n---\n\n{content}",
                display="side"
            )
        )

    # 3. Format Final Message
    answer_text = rag_result["answer"]
    is_fallback = rag_result.get("is_fallback", False)

    if is_fallback:
        answer_text = (
            "⚠️ **Support Escalation Required**\n\n"
            f"> {FALLBACK_RESPONSE}\n\n"
            "Our documentation does not contain sufficient verified details to answer this query safely. "
            "Please reach out to our official support team at **support@example.com** or check the community forum."
        )

    await cl.Message(
        content=answer_text,
        elements=elements
    ).send()


if __name__ == "__main__":
    print("Run this application using:")
    print("  chainlit run app.py -w")
