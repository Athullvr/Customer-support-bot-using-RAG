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

4. Robust Fallback Architecture:
   - Implements both high-fidelity BeautifulSoup/markdownify parsing and pure
     Python standard library (html.parser/urllib) parsing for zero-dependency portability.
"""

import os
import re
import json
import time
import argparse
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from config import settings

USER_AGENT = "CustomerSupportRAGBot/1.0 (+https://github.com/Athullvr/Customer-support-bot-using-RAG)"

# Try importing third-party libraries; fall back gracefully to standard library
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from bs4 import BeautifulSoup
    from markdownify import markdownify as md
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class SimpleHTMLToMarkdownParser(HTMLParser):
    """Zero-dependency HTML to clean Markdown converter using standard library."""
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip_tags = {"script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "button", "form"}
        self.skip_depth = 0
        self.in_heading = False
        self.heading_level = 1
        self.title = ""
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self.skip_depth += 1
            return
        if self.skip_depth > 0:
            return

        if tag == "title":
            self.in_title = True
        elif tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.in_heading = True
            self.heading_level = int(tag[1])
            self.result.append("\n\n" + "#" * self.heading_level + " ")
        elif tag in ["p", "div", "section", "article"]:
            self.result.append("\n\n")
        elif tag == "li":
            self.result.append("\n- ")
        elif tag == "code":
            self.result.append(" `")
        elif tag == "pre":
            self.result.append("\n```\n")

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth > 0:
            return

        if tag == "title":
            self.in_title = False
        elif tag in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            self.in_heading = False
            self.result.append("\n\n")
        elif tag == "code":
            self.result.append("` ")
        elif tag == "pre":
            self.result.append("\n```\n")

    def handle_data(self, data):
        if self.skip_depth > 0:
            return
        text = data.strip()
        if not text:
            return
        if self.in_title and not self.title:
            self.title = text.split("|")[0].split(" - ")[0].strip()
        if self.in_heading and not self.title:
            self.title = text
        self.result.append(data)

    def get_markdown(self) -> str:
        raw_text = "".join(self.result)
        # Normalize whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", raw_text).strip()
        return cleaned


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


def fetch_url_content(url: str, timeout: int = 12) -> Optional[str]:
    """Fetch URL content with standard headers."""
    if HAS_REQUESTS:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        return None
    else:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8", errors="ignore")
        return None


def clean_html_to_markdown(html_content: str, base_url: str) -> Dict[str, str]:
    """Extract title and clean markdown body from article HTML."""
    if HAS_BS4:
        soup = BeautifulSoup(html_content, "html.parser")
        title = ""
        h1 = soup.find("h1")
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)
        elif soup.title and soup.title.get_text(strip=True):
            title = soup.title.get_text(strip=True).split("|")[0].split(" - ")[0].strip()
        else:
            title = "Help Center Article"

        for tag in soup.find_all(["nav", "header", "footer", "aside", "script", "style", "noscript", "svg", "button", "form", "iframe"]):
            tag.decompose()

        main_content = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"(content|article|document|markdown|doc-content)", re.I))
            or soup.find("body")
        )

        if not main_content:
            return {"title": title, "markdown": ""}

        markdown_text = md(str(main_content), heading_style="ATX", bullets="-")
        markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text).strip()
        return {"title": title, "markdown": markdown_text}
    else:
        parser = SimpleHTMLToMarkdownParser()
        parser.feed(html_content)
        md_text = parser.get_markdown()
        title = parser.title or "Help Center Article"
        return {"title": title, "markdown": md_text}


def fetch_article_urls_from_sitemap(sitemap_url: str, limit: int = 75) -> List[str]:
    """Discover help/documentation URLs from sitemap XML."""
    print(f"🔍 Fetching sitemap: {sitemap_url}")
    xml_content = fetch_url_content(sitemap_url)
    if not xml_content:
        return []

    # Extract all <loc> tags via regex to work with or without bs4/lxml
    locs = re.findall(r"<loc>(https?://[^<]+)</loc>", xml_content)

    # Expand sub-sitemaps if any
    sub_maps = [l for l in locs if l.endswith(".xml")]
    if sub_maps:
        all_locs = []
        for sub_map in sub_maps[:2]:
            sub_xml = fetch_url_content(sub_map)
            if sub_xml:
                all_locs.extend(re.findall(r"<loc>(https?://[^<]+)</loc>", sub_xml))
        if all_locs:
            locs = all_locs

    filtered_urls = []
    for u in locs:
        if any(u.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".svg", ".pdf", ".zip", ".xml"]):
            continue
        if any(term in u for term in ["/develop/concepts/", "/develop/quick-reference/", "/deploy/", "/develop/tutorials/", "/knowledge-base/", "/guide/", "/docs/"]):
            filtered_urls.append(u)

    if len(filtered_urls) < 10:
        filtered_urls = [u for u in locs if not u.endswith(".xml")]

    return filtered_urls[:limit]


def scrape_help_center(
    sitemap_url: str = "https://docs.streamlit.io/sitemap-0.xml",
    limit: int = 60,
    output_dir: Optional[Path] = None,
    delay_sec: float = 0.25
) -> List[Dict]:
    """
    Main ingestion pipeline:
    1. Discovers article URLs.
    2. Validates against robots.txt.
    3. Fetches each page politely.
    4. Converts HTML to clean Markdown with metadata.
    5. Saves to JSON and individual Markdown files in data/raw/.
    """
    output_dir = output_dir or settings.raw_data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    urls = fetch_article_urls_from_sitemap(sitemap_url, limit=limit)
    print(f"📋 Found {len(urls)} target articles for ingestion (Limit: {limit})")

    articles: List[Dict] = []

    for idx, url in enumerate(urls, 1):
        if not is_allowed_by_robots(url):
            print(f"⛔ Skipping {url} (Disallowed by robots.txt)")
            continue

        try:
            print(f"[{idx}/{len(urls)}] Scraping: {url}")
            html_text = fetch_url_content(url)
            if not html_text:
                print(f"⚠️ Failed to fetch content for {url}")
                continue

            extracted = clean_html_to_markdown(html_text, url)
            title = extracted["title"]
            markdown_content = extracted["markdown"]

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

            # Save individual markdown file for easy inspection
            md_file = output_dir / f"{slug}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n**Source**: {url}\n\n---\n\n{markdown_content}\n")

            # Polite delay
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
