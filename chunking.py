"""
Heading-Aware Chunking & Embedding Module for Customer Support RAG Bot.

INTERVIEW RATIONALE & DESIGN CHOICES:
-------------------------------------
1. Why Heading-Aware Chunking?
   - Naive character/token splitting arbitrarily cuts text across section headers, tables,
     code snippets, and steps.
   - Heading-aware chunking parses Markdown headers (#, ##, ###) first, creating semantic
     boundaries around complete topics.

2. Why Chunk Overlap (e.g., 100 characters)?
   - If a crucial step or explanation sits right at the cut-off boundary between two chunks,
     a query might match only one half and lose necessary context.
   - An overlap (20% of chunk size) ensures boundary continuity so full context is retained.

3. Context Injection (Heading Path / Breadcrumbs):
   - Chunks from deep within a document (e.g., "Troubleshooting 403 Forbidden") often lack
     the global product context (e.g., "Streamlit Community Cloud Secrets").
   - By prepending the heading hierarchy `[Document: Title > Section: Subtitle]` to the chunk text,
     both BM25 keyword search and dense vector embeddings achieve higher retrieval precision.

4. Source Metadata Preservation:
   - Preserves canonical URL, article title, and chunk index in metadata for verified citations.
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import settings
from embeddings import EmbeddingManager


def split_markdown_by_headings(markdown_text: str, default_title: str = "") -> List[Dict[str, str]]:
    """
    Split markdown document into structural sections based on headers (#, ##, ###).
    Maintains the hierarchical breadcrumb path for each section.
    """
    lines = markdown_text.splitlines()
    sections: List[Dict[str, str]] = []
    
    current_heading_hierarchy: List[str] = [default_title] if default_title else []
    current_lines: List[str] = []
    
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$")

    for line in lines:
        match = header_pattern.match(line.strip())
        if match:
            # Flush existing section
            content = "\n".join(current_lines).strip()
            if content:
                heading_path = " > ".join(current_heading_hierarchy)
                sections.append({
                    "heading_path": heading_path,
                    "section_title": current_heading_hierarchy[-1] if current_heading_hierarchy else default_title,
                    "content": content
                })
                current_lines = []

            level = len(match.group(1))
            heading_text = match.group(2).strip()

            # Adjust hierarchy based on level (level 1 = root, level 2 = sub, etc.)
            if level == 1:
                current_heading_hierarchy = [heading_text]
            else:
                # Keep hierarchy up to level - 1, then append new heading
                current_heading_hierarchy = current_heading_hierarchy[:level - 1]
                current_heading_hierarchy.append(heading_text)
        else:
            current_lines.append(line)

    # Flush last remaining section
    content = "\n".join(current_lines).strip()
    if content:
        heading_path = " > ".join(current_heading_hierarchy)
        sections.append({
            "heading_path": heading_path or default_title,
            "section_title": current_heading_hierarchy[-1] if current_heading_hierarchy else default_title,
            "content": content
        })

    return sections


def recursive_text_split(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[str]:
    """
    Recursively split text into chunks of target chunk_size with chunk_overlap,
    prioritizing natural split boundaries (\n\n, \n, . , space).
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        if end >= text_len:
            chunks.append(text[start:].strip())
            break

        # Look for natural split point within the overlap window
        split_pos = -1
        window = text[start:end]

        # 1. Try double newline (paragraph break)
        p_idx = window.rfind("\n\n")
        if p_idx > chunk_size // 2:
            split_pos = start + p_idx + 2
        # 2. Try single newline
        elif (n_idx := window.rfind("\n")) > chunk_size // 2:
            split_pos = start + n_idx + 1
        # 3. Try sentence boundary (". ")
        elif (s_idx := window.rfind(". ")) > chunk_size // 2:
            split_pos = start + s_idx + 2
        # 4. Try space
        elif (sp_idx := window.rfind(" ")) > chunk_size // 2:
            split_pos = start + sp_idx + 1
        else:
            split_pos = end

        chunk_content = text[start:split_pos].strip()
        if chunk_content:
            chunks.append(chunk_content)

        # Advance start point, subtracting overlap
        start = max(start + 1, split_pos - chunk_overlap)

    return chunks


def process_and_chunk_articles(
    articles: List[Dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    inject_heading_context: bool = True
) -> List[Dict]:
    """
    Process a list of ingested articles:
    1. Parse Markdown heading hierarchy.
    2. Split section content using recursive sliding window.
    3. Inject heading context into chunk body.
    4. Attach rich citation metadata.
    """
    all_chunks: List[Dict] = []
    global_chunk_idx = 1

    for article in articles:
        doc_id = article.get("id", f"doc_{global_chunk_idx}")
        title = article.get("title", "Support Guide")
        url = article.get("url", "")
        content = article.get("content", "")

        sections = split_markdown_by_headings(content, default_title=title)

        for sec_idx, section in enumerate(sections, 1):
            heading_path = section["heading_path"]
            section_title = section["section_title"]
            sec_content = section["content"]

            sub_chunks = recursive_text_split(
                sec_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            for sub_idx, raw_chunk_text in enumerate(sub_chunks, 1):
                # Context injection: Prepend document title and section breadcrumb
                if inject_heading_context:
                    formatted_content = f"[{title} > {heading_path}]\n\n{raw_chunk_text}"
                else:
                    formatted_content = raw_chunk_text

                chunk_obj = {
                    "id": f"{doc_id}_c{sec_idx}_{sub_idx}",
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "heading_path": heading_path,
                    "section_title": section_title,
                    "content": formatted_content,
                    "raw_content": raw_chunk_text,
                    "char_count": len(formatted_content),
                    "token_estimate": max(1, len(formatted_content) // 4)
                }
                all_chunks.append(chunk_obj)
                global_chunk_idx += 1

    return all_chunks


def generate_chunk_embeddings(
    chunks: List[Dict],
    embedding_mgr: Optional[EmbeddingManager] = None,
    batch_size: int = 32
) -> List[Dict]:
    """
    Generate vector embeddings for all processed chunks in batches.
    """
    mgr = embedding_mgr or EmbeddingManager()
    print(f"🧠 Generating vector embeddings for {len(chunks)} chunks using {mgr.provider} ({mgr.model_name})...")

    texts_to_embed = [c["content"] for c in chunks]
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts_to_embed), batch_size):
        batch = texts_to_embed[i:i + batch_size]
        vecs = mgr.embed_documents(batch)
        all_embeddings.extend(vecs)
        if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(texts_to_embed):
            print(f"  Processed {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)} chunk embeddings...")

    # Attach embedding vectors to chunk dictionaries
    for chunk, vec in zip(chunks, all_embeddings):
        chunk["embedding"] = vec

    return chunks


def build_and_save_chunks(
    input_file: Optional[Path] = None,
    output_file: Optional[Path] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    compute_embeddings: bool = True
) -> List[Dict]:
    """Full chunking & embedding pipeline orchestration."""
    input_path = input_file or (settings.raw_data_dir / "articles.json")
    output_path = output_file or (settings.processed_data_dir / "chunks.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Raw articles file not found at {input_path}. Please run ingest.py first.")

    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"📄 Loaded {len(articles)} raw articles from {input_path}")
    print(f"⚙️ Chunking configuration: size={chunk_size} chars, overlap={chunk_overlap} chars")

    chunks = process_and_chunk_articles(
        articles=articles,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        inject_heading_context=True
    )
    print(f"✂️ Generated {len(chunks)} heading-aware chunks.")

    if compute_embeddings:
        chunks = generate_chunk_embeddings(chunks)

    # Save processed chunks
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✅ Chunking & Embedding complete! Saved {len(chunks)} chunks to {output_path}")
    return chunks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk and embed scraped documentation articles.")
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size, help="Target chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap, help="Chunk overlap in characters")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding generation")
    args = parser.parse_args()

    build_and_save_chunks(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        compute_embeddings=not args.no_embeddings
    )
