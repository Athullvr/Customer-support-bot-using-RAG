"""
Data Ingestion Module for Customer Support RAG Bot.

INTERVIEW RATIONALE & DESIGN CHOICES:
-------------------------------------
1. Robots.txt & Rate Limiting:
   - Always verify crawler permissions via urllib.robotparser to adhere to web standards.
   - Introduce polite delays (0.3s) between requests to prevent rate-limiting/429 errors.

2. Markdown Format Preservation:
   - Converting HTML to structured Markdown (retaining #, ##, ### headers) is crucial
     for Step 3 (Heading-Aware Chunking). Preserving hierarchy allows semantic chunking
     without breaking logical sections or code blocks.

3. Rich Metadata Extraction (Title, URL, Category):
   - In customer support RAG, the LLM must provide direct, clickable citations.
   - Capturing the canonical URL and title at ingestion time ensures that downstream
     retrieved chunks always carry accurate provenance.
"""

import os
import re
import json
import time
import argparse
import urllib.robotparser
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from config import settings


USER_AGENT = "CustomerSupportRAGBot/1.0 (+https://github.com/Athullvr/Customer-support-bot-using-RAG)"


def is_allowed_by_robots(url: str, user_agent: str = USER_AGENT) -> bool:
    """Check robots.txt permissions for a given URL."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception as e:
        print(f"⚠️ Warning: Could not check robots.txt for {url} ({e}). Proceeding politely.")
        return True


def clean_html_to_markdown(html_content: str, base_url: str) -> Dict[str, str]:
    """
    Extract title and clean markdown body from article HTML.
    Strips navigation bars, sidebars, cookie banners, and footers.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract article title
    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True).split("|")[0].split(" - ")[0].strip()
    else:
        title = "Help Center Article"

    # Remove non-content elements to prevent garbage in vector embeddings
    unwanted_tags = [
        "nav", "header", "footer", "aside", "script", "style",
        "noscript", "svg", "button", "form", "iframe"
    ]
    for tag in soup.find_all(unwanted_tags):
        tag.decompose()

    # Target main content container if available
    main_content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"(content|article|document|markdown|doc-content)", re.I))
        or soup.find("body")
    )

    if not main_content:
        return {"title": title, "markdown": ""}

    # Convert clean HTML to GitHub-flavored Markdown
    markdown_text = md(
        str(main_content),
        heading_style="ATX",
        bullets="-",
        strip=["a", "img"] if False else []
    )

    # Normalize excessive newlines and whitespace
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text).strip()

    return {
        "title": title,
        "markdown": markdown_text
    }


def fetch_article_urls_from_sitemap(sitemap_url: str, limit: int = 75) -> List[str]:
    """Discover help/documentation URLs from sitemap XML."""
    print(f"🔍 Fetching sitemap: {sitemap_url}")
    headers = {"User-Agent": USER_AGENT}
    resp = requests.get(sitemap_url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "xml")
    locs = [loc.get_text(strip=True) for loc in soup.find_all("loc")]

    # If sitemap index, fetch sub-sitemaps
    if any(loc.endswith(".xml") for loc in locs):
        expanded_locs = []
        for sub_map in locs[:3]:
            try:
                sub_resp = requests.get(sub_map, headers=headers, timeout=15)
                sub_soup = BeautifulSoup(sub_resp.content, "xml")
                expanded_locs.extend([l.get_text(strip=True) for l in sub_soup.find_all("loc")])
            except Exception:
                pass
        locs = expanded_locs

    # Filter for relevant guide/concept/tutorial/deploy/knowledge-base pages
    filtered_urls = []
    for u in locs:
        # Avoid static assets and search endpoints
        if any(u.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".svg", ".pdf", ".zip"]):
            continue
        if any(term in u for term in ["/develop/concepts/", "/develop/quick-reference/", "/deploy/", "/develop/tutorials/", "/knowledge-base/", "/guide/", "/docs/"]):
            filtered_urls.append(u)

    # If filtering returned too few, fallback to raw HTML doc links
    if len(filtered_urls) < 10:
        filtered_urls = [u for u in locs if not u.endswith(".xml")]

    return filtered_urls[:limit]


def scrape_help_center(
    sitemap_url: str = "https://docs.streamlit.io/sitemap-0.xml",
    limit: int = 60,
    output_dir: Optional[Path] = None,
    delay_sec: float = 0.3
) -> List[Dict]:
    """
    Main ingestion pipeline:
    1. Discovers article URLs.
    2. Validates against robots.txt.
    3. Fetches each page politely.
    4. Converts HTML to clean Markdown with metadata.
    5. Saves to JSON and Markdown in data/raw/.
    """
    output_dir = output_dir or settings.raw_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = fetch_article_urls_from_sitemap(sitemap_url, limit=limit)
    print(f"📋 Found {len(urls)} target articles for ingestion (Limit: {limit})")

    articles: List[Dict] = []
    headers = {"User-Agent": USER_AGENT}

    for idx, url in enumerate(urls, 1):
        if not is_allowed_by_robots(url):
            print(f"⛔ Skipping {url} (Disallowed by robots.txt)")
            continue

        try:
            print(f"[{idx}/{len(urls)}] Scraping: {url}")
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code != 200:
                print(f"⚠️ Non-200 status {resp.status_code} for {url}")
                continue

            extracted = clean_html_to_markdown(resp.text, url)
            title = extracted["title"]
            markdown_content = extracted["markdown"]

            # Skip trivial/empty pages
            if len(markdown_content) < 80:
                continue

            slug = re.sub(r"[^a-zA-Z0-9_\-]+", "_", urlparse(url).path.strip("/")).strip("_")
            if not slug:
                slug = f"article_{idx}"

            article_data = {
                "id": f"doc_{idx}",
                "slug": slug,
                "title": title,
                "url": url,
                "content": markdown_content,
                "length_chars": len(markdown_content),
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            articles.append(article_data)

            # Also save individual markdown file for quick manual inspection
            md_file = output_dir / f"{slug}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n**Source**: {url}\n\n---\n\n{markdown_content}\n")

            # Polite delay between requests
            time.sleep(delay_sec)

        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

    # Save aggregate dataset in JSON
    aggregate_json_path = output_dir / "articles.json"
    with open(aggregate_json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Ingestion complete! Successfully scraped and saved {len(articles)} articles.")
    print(f"📁 JSON dataset: {aggregate_json_path}")
    print(f"📁 Individual Markdown files: {output_dir}")

    return articles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape and ingest help center documentation.")
    parser.add_argument(
        "--sitemap",
        type=str,
        default="https://docs.streamlit.io/sitemap-0.xml",
        help="Sitemap XML URL of the help center / documentation site"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=60,
        help="Maximum number of articles to scrape (e.g. 50-100)"
    )
    args = parser.parse_args()

    scrape_help_center(sitemap_url=args.sitemap, limit=args.limit)
