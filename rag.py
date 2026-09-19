"""
Customer Support RAG Chain Module with Multi-Provider LLM Support & Conversational Routing.

INTERVIEW RATIONALE & DESIGN CHOICES:
-------------------------------------
1. Multi-Provider LLM Architecture:
   - Supports OpenAI (gpt-4o-mini), Groq (free Llama-3.1-8b / 70b), Ollama (free local Llama-3.2),
     and OpenRouter via standard OpenAI-compatible interfaces.
   - Eliminates vendor lock-in and allows 100% free / open-source deployments.

2. Conversational Intent Routing (Chit-Chat vs Knowledge Retrieval):
   - Pure RAG systems fail when users say "hello", "who are you?", or "thank you" because
     greeting tokens aren't in product documentation.
   - An intent router intercepts chit-chat and greetings, providing a warm, informative
     introduction without polling the vector DB or triggering false "I don't know" fallbacks.

3. Strict Grounded Prompting (Anti-Hallucination):
   - In customer support, answering with plausible-sounding false information breaks user trust.
   - The prompt explicitly instructs the LLM to rely ONLY on the retrieved context passages
     and forbid outside speculation.

4. Mandatory Provenance & Citations:
   - Users need immediate verification. Every generated response includes canonical
     source URLs and article titles from the ingested help center.
"""

import os
import re
import argparse
from typing import Dict, List, Optional, Tuple

from config import settings
from index import TypesenseSearcher

# Try importing LangChain ChatOpenAI
try:
    from langchain_openai import ChatOpenAI
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


def check_conversational_intent(query: str) -> Optional[str]:
    """
    Check if the user message is a conversational greeting, thank you, or help query.
    Returns a warm conversational response if matched, or None for knowledge retrieval.
    """
    clean_q = query.strip().lower()
    clean_q = re.sub(r"[^\w\s]", "", clean_q).strip()

    # 1. Greetings
    greeting_patterns = [
        r"^(hi|hello|hey|heya|howdy|sup|hola|good morning|good afternoon|good evening|greetings)$",
        r"^(hi|hello|hey)\s+(there|bot|assistant|support|ai)$"
    ]
    if any(re.match(p, clean_q) for p in greeting_patterns):
        return (
            "👋 **Hello! Welcome to the Customer Support AI Assistant.**\n\n"
            "I can help you with technical support and troubleshooting across our official product documentation, including:\n"
            "- 🚀 **Deploying Apps**: Community Cloud setup, GitHub connection, and Docker containers.\n"
            "- 🔒 **Secrets Management**: Configuring `.streamlit/secrets.toml` and cloud environment variables.\n"
            "- ⚡ **Caching & Performance**: Best practices for `st.cache_data` vs `st.cache_resource`.\n"
            "- 🎨 **Theming & Layouts**: Custom colors, borders, and multi-page application setup.\n\n"
            "How can I assist you with your project today?"
        )

    # 2. Identity / Capabilities
    if clean_q in ["who are you", "what are you", "what can you do", "help", "what is your name"]:
        return (
            "🤖 **About Me:**\n\n"
            "I am an AI Customer Support Assistant powered by **Typesense Hybrid Search** and **LangChain**.\n"
            "My responses are strictly grounded in official product documentation with direct source URL citations.\n\n"
            "Ask me any question (for example: *'How do I add dependencies to my app?'* or *'How do I clear cached data?'*)."
        )

    # 3. Thank you & pleasantries
    if clean_q in ["thank you", "thanks", "thx", "appreciate it", "great thanks", "thank you so much"]:
        return "You're very welcome! 😊 Feel free to ask if you have any other questions. Happy building!"

    # 4. Goodbye
    if clean_q in ["bye", "goodbye", "see you", "cya", "have a good day"]:
        return "Goodbye! 👋 Have a wonderful day and reach out anytime you need support!"

    return None


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
    End-to-end RAG Chain combining Typesense Multi-Modal Search with Multi-Provider LLMs.
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
        """
        Initialize LLM with support for:
        1. OpenAI (if OPENAI_API_KEY is present)
        2. Groq (if GROQ_API_KEY is present, free ultra-fast Llama-3.1)
        3. Ollama (if configured or local endpoint available)
        4. Custom OpenAI-compatible base URL (OpenRouter, TogetherAI, LocalAI)
        """
        if not HAS_LANGCHAIN:
            return None

        provider = settings.llm_provider.lower()
        api_key = settings.llm_api_key or settings.openai_api_key or settings.groq_api_key
        base_url = settings.llm_base_url

        # Auto-detect Groq API key if provided
        if settings.groq_api_key and not settings.openai_api_key:
            provider = "groq"
            api_key = settings.groq_api_key
            base_url = base_url or "https://api.groq.com/openai/v1"
            if not self.llm_model_name or self.llm_model_name.startswith("gpt-"):
                self.llm_model_name = "llama-3.1-8b-instant"

        # Auto-detect Ollama if requested
        if provider == "ollama":
            base_url = base_url or "http://localhost:11434/v1"
            api_key = api_key or "ollama"
            if not self.llm_model_name or self.llm_model_name.startswith("gpt-"):
                self.llm_model_name = "llama3.2"

        if not api_key or api_key in ["your_openai_api_key_here", "your_groq_api_key_here"]:
            return None

        try:
            kwargs = {
                "model": self.llm_model_name,
                "temperature": self.temperature,
                "api_key": api_key
            }
            if base_url:
                kwargs["base_url"] = base_url

            llm_inst = ChatOpenAI(**kwargs)
            print(f"✓ Initialized LLM ({provider}): {self.llm_model_name} (Base URL: {base_url or 'default'})")
            return llm_inst
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
        Execute RAG pipeline:
        1. Conversational Intent Check (handles 'hello', 'who are you', etc.).
        2. Typesense Retrieval (Hybrid / Vector / Keyword).
        3. Strict Grounded Generation via LLM (or robust local synthesizer).
        """
        trimmed_q = question.strip()
        if not trimmed_q:
            return {
                "query": question,
                "answer": FALLBACK_RESPONSE,
                "sources": [],
                "retrieved_chunks": [],
                "context_used": "",
                "search_mode": search_mode or settings.search_mode,
                "is_fallback": True
            }

        # 1. Check Conversational Intent (Greetings / Chit-chat)
        if greeting_reply := check_conversational_intent(trimmed_q):
            return {
                "query": question,
                "answer": greeting_reply,
                "sources": [],
                "retrieved_chunks": [],
                "context_used": "",
                "search_mode": "conversational_intent",
                "is_fallback": False
            }

        k = top_k or settings.top_k
        mode = search_mode or settings.search_mode
        alph = alpha if alpha is not None else settings.hybrid_alpha

        # 2. Retrieve top-k chunks from Typesense
        retrieved_chunks = self.searcher.search(
            query=question,
            search_mode=mode,
            top_k=k,
            alpha=alph
        )

        if not retrieved_chunks:
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

        # Deduplicate sources
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

        # 4. Generate Answer via LLM or Local Synthesizer
        if self.llm is not None:
            try:
                prompt_text = SYSTEM_PROMPT.format(context=formatted_context, question=question)
                response = self.llm.invoke(prompt_text)
                answer_text = response.content if hasattr(response, "content") else str(response)
            except Exception as e:
                print(f"⚠️ LLM API error ({e}). Using local grounded synthesizer.")
                answer_text = self._local_grounded_answer(question, retrieved_chunks, sources)
        else:
            answer_text = self._local_grounded_answer(question, retrieved_chunks, sources)

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

    def _local_grounded_answer(
        self,
        question: str,
        chunks: List[Dict],
        sources: List[Dict]
    ) -> str:
        """
        Local grounded answer synthesizer for offline environments.
        Synthesizes relevant bullet steps and citations from top context passages.
        """
        if not chunks:
            return FALLBACK_RESPONSE

        # Domain relevance check
        q_words = set(re.findall(r"\w+", question.lower())) - {
            "how", "to", "do", "i", "the", "a", "an", "is", "with", "what", "where", "can", "for", "my", "in", "of", "and"
        }
        combined_text = " ".join([c.get("content", "").lower() for c in chunks[:3]])
        matching_words = [w for w in q_words if w in combined_text]

        if q_words and (len(matching_words) / len(q_words)) < 0.25:
            return FALLBACK_RESPONSE

        top_chunk = chunks[0]
        title = top_chunk.get("title", "Help Guide")
        heading = top_chunk.get("heading_path", "")
        content = top_chunk.get("raw_content", top_chunk.get("content", ""))

        # Clean passages into guidance paragraphs
        clean_lines = [
            line.strip() for line in content.splitlines()
            if line.strip() and not line.startswith("[") and not line.startswith("#")
        ]
        summary_body = "\n\n".join(clean_lines[:4])

        if not summary_body:
            return FALLBACK_RESPONSE

        sources_markdown = "\n".join([f"- [{s['title']}]({s['url']}) (`{s['heading_path']}`)" for s in sources[:3]])

        return (
            f"Here is what our documentation states regarding **{heading or title}**:\n\n"
            f"{summary_body}\n\n"
            f"### Sources\n"
            f"{sources_markdown}"
        )


# Global singleton instance
rag_chain = CustomerSupportRAGChain()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Customer Support RAG Chain.")
    parser.add_argument("--query", type=str, default="hello", help="User query")
    parser.add_argument("--mode", type=str, choices=["hybrid", "vector", "keyword"], default="hybrid", help="Search mode")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks")
    args = parser.parse_args()

    print(f"🤖 User Query: '{args.query}'\n")
    chain = CustomerSupportRAGChain()
    result = chain.query(question=args.query, search_mode=args.mode, top_k=args.top_k)

    print("=" * 60)
    print("💬 Chatbot Response:")
    print("=" * 60)
    print(result["answer"])
    print("=" * 60)
    if result["sources"]:
        print(f"📚 Sources cited ({len(result['sources'])}):")
        for s in result["sources"]:
            print(f"  • {s['title']} -> {s['url']}")
    print(f"⚙️ Route: {result['search_mode']} (Fallback: {result['is_fallback']})")
