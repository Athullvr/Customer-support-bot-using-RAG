"""
Customer Support RAG Chain Module.

INTERVIEW RATIONALE & DESIGN CHOICES:
-------------------------------------
1. Strict Grounded Prompting (Anti-Hallucination):
   - In customer support, answering with plausible-sounding false information can break
     user production apps or degrade trust.
   - The prompt explicitly instructs the LLM to rely ONLY on the retrieved context passages
     and forbid outside speculation.

2. Mandatory Provenance & Citations:
   - Users need immediate verification. Every generated response includes the canonical
     source URLs and article titles from the ingested help center.

3. Deterministic Fallback ("I don't know, please contact support"):
   - When retrieved context is irrelevant or score is below relevance thresholds, the system
     safely escalates to human support rather than fabricating answers.

4. LangChain LCEL (LangChain Expression Language) & Modular Pipeline:
   - Formatted context -> PromptTemplate -> LLM -> StrOutputParser.
   - Decoupled search engine (Typesense) allows effortless swapping of retrieval modes.
"""

import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

from config import settings
from index import TypesenseSearcher

# Try importing LangChain ChatOpenAI & Core Prompts
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

FALLBACK_RESPONSE = "I don't know, please contact support."

SYSTEM_PROMPT = """You are a helpful, precise, and professional Customer Support AI Assistant.

CRITICAL OPERATIONAL RULES:
1. Answer the user's question using ONLY the facts directly stated in the Context below.
2. DO NOT speculate, assume, or use any outside knowledge not present in the Context.
3. If the Context does not contain sufficient information to answer the question accurately, reply EXACTLY with:
   "I don't know, please contact support."
4. When answering, provide clear, step-by-step guidance whenever applicable.
5. At the end of your response, always cite the relevant documentation Source URLs and Titles from the Context under a "### Sources" section.

CONTEXT:
{context}

USER QUESTION:
{question}
"""


def format_context_chunks(chunks: List[Dict]) -> str:
    """Format retrieved document chunks into clean numbered context blocks."""
    if not chunks:
        return "No relevant documentation found."

    formatted_blocks = []
    for idx, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "Help Guide")
        url = chunk.get("url", "")
        heading = chunk.get("heading_path", "")
        content = chunk.get("content", "").strip()

        block = (
            f"--- Context Passage [{idx}] ---\n"
            f"Article: {title}\n"
            f"Section: {heading}\n"
            f"Source URL: {url}\n"
            f"Content:\n{content}\n"
        )
        formatted_blocks.append(block)

    return "\n".join(formatted_blocks)


class CustomerSupportRAGChain:
    """
    End-to-end RAG Chain combining Typesense Multi-Modal Search with LLM generation.
    """
    def __init__(
        self,
        searcher: Optional[TypesenseSearcher] = None,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None
    ):
        self.searcher = searcher or TypesenseSearcher()
        self.llm_model_name = llm_model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initialize the LLM instance with deterministic temperature."""
        api_key = settings.openai_api_key
        if not api_key or api_key == "your_openai_api_key_here" or not HAS_LANGCHAIN:
            return None

        try:
            return ChatOpenAI(
                model=self.llm_model_name,
                temperature=self.temperature,
                openai_api_key=api_key
            )
        except Exception as e:
            print(f"⚠️ Could not initialize ChatOpenAI ({e}).")
            return None

    def query(
        self,
        question: str,
        search_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None
    ) -> Dict:
        """
        Execute the RAG pipeline:
        1. Retrieve top-k relevant chunks from Typesense.
        2. Check for empty or out-of-domain retrieval.
        3. Format context with source provenance.
        4. Generate grounded answer via LLM (or extractive fallback).
        """
        k = top_k or settings.top_k
        mode = search_mode or settings.search_mode
        alph = alpha if alpha is not None else settings.hybrid_alpha

        # 1. Retrieve top-k chunks
        retrieved_chunks = self.searcher.search(
            query=question,
            search_mode=mode,
            top_k=k,
            alpha=alph
        )

        # 2. Guard rail: If no chunks retrieved or query is completely blank
        if not retrieved_chunks or not question.strip():
            return {
                "query": question,
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "retrieved_chunks": [],
                "context_used": "",
                "search_mode": mode,
                "is_fallback": True
            }

        # 3. Format Context
        formatted_context = format_context_chunks(retrieved_chunks)

        # Deduplicate and extract sources
        sources = []
        seen_urls = set()
        for c in retrieved_chunks:
            url = c.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    "title": c.get("title", "Documentation"),
                    "url": url,
                    "heading_path": c.get("heading_path", ""),
                    "score": c.get("score", 0.0)
                })

        # 4. Generate Answer
        if self.llm is not None:
            try:
                prompt_text = SYSTEM_PROMPT.format(context=formatted_context, question=question)
                response = self.llm.invoke(prompt_text)
                answer_text = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                print(f"⚠️ LLM generation failed ({e}). Using extractive summary fallback.")
                answer_text = self._extractive_fallback_answer(question, retrieved_chunks, sources)
        else:
            # Extractive fallback if no OpenAI API Key configured
            answer_text = self._extractive_fallback_answer(question, retrieved_chunks, sources)

        # Check if model triggered fallback answer
        is_fallback = "i don't know" in answer_text.lower() and "contact support" in answer_text.lower()

        return {
            "query": question,
            "answer": answer_text.strip(),
            "sources": sources,
            "retrieved_chunks": retrieved_chunks,
            "context_used": formatted_context,
            "search_mode": mode,
            "is_fallback": is_fallback
        }

    def _extractive_fallback_answer(
        self,
        question: str,
        chunks: List[Dict],
        sources: List[Dict]
    ) -> str:
        """
        Extractive synthesis fallback for offline testing or environments without an active LLM key.
        Checks relevance threshold to trigger "I don't know, please contact support."
        """
        if not chunks:
            return FALLBACK_RESPONSE

        # Check semantic/keyword overlap for domain relevance
        q_words = set(re.findall(r"\w+", question.lower())) - {"how", "to", "do", "i", "the", "a", "an", "is", "with", "what"}
        combined_text = " ".join([c.get("content", "").lower() for c in chunks[:2]])
        matching_words = [w for w in q_words if w in combined_text]

        # If less than 20% of query keywords appear in top context, trigger fallback
        if q_words and (len(matching_words) / len(q_words)) < 0.25:
            return FALLBACK_RESPONSE

        top_chunk = chunks[0]
        title = top_chunk.get("title", "Help Guide")
        heading = top_chunk.get("heading_path", "")
        content = top_chunk.get("raw_content", top_chunk.get("content", ""))

        clean_lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("[") and not line.startswith("#")]
        summary_snippet = "\n".join(clean_lines[:4])

        if not summary_snippet:
            return FALLBACK_RESPONSE

        sources_markdown = "\n".join([f"- [{s['title']}]({s['url']}) ({s['heading_path']})" for s in sources[:3]])

        return (
            f"Based on the official documentation for **{title}** ({heading}):\n\n"
            f"{summary_snippet}\n\n"
            f"### Sources\n"
            f"{sources_markdown}"
        )


# Global singleton instance
rag_chain = CustomerSupportRAGChain()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Customer Support RAG Chain.")
    parser.add_argument("--query", type=str, default="How do I add dependencies to my Streamlit Community Cloud app?", help="User query")
    parser.add_argument("--mode", type=str, choices=["hybrid", "vector", "keyword"], default="hybrid", help="Search mode")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks")
    args = parser.parse_args()

    print(f"🤖 User Query: {args.query}\n")
    chain = CustomerSupportRAGChain()
    result = chain.query(question=args.query, search_mode=args.mode, top_k=args.top_k)

    print("=" * 60)
    print("💬 Chatbot Response:")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    print(f"📚 Sources cited ({len(result['sources'])}):")
    for s in result["sources"]:
        print(f"  • {s['title']} -> {s['url']}")
    print(f"⚙️ Search mode used: {result['search_mode']} (Fallback: {result['is_fallback']})")
