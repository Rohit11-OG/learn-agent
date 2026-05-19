"""Agent tools: web search, fetch url, deep crawl, calculator, wikipedia,
file reader, doc search (RAG). Docstrings are sharp on purpose -> the model
reads them to pick the right tool.

Optimizations:
- one shared requests.Session  -> reuses TCP connections (faster)
- in-memory caches              -> repeat searches/fetches return instantly
"""

import ast
import operator
import os
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from ddgs import DDGS

# --- shared HTTP session: one connection pool reused by every tool ---
_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (LearningAgent)"})

# --- caches: avoid repeating the same network call ---
_search_cache: dict[str, str] = {}
_page_cache: dict[str, str] = {}


def _clean_html(html: str, drop_extra=()) -> str:
    """Strip tags/noise from HTML, return squashed plain text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", *drop_extra]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _get_page(url: str) -> str:
    """Fetch a URL's clean text (cached). Returns '' on failure."""
    if url in _page_cache:
        return _page_cache[url]
    try:
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
        text = _clean_html(resp.text)
    except Exception:
        text = ""
    _page_cache[url] = text
    return text


@tool
def web_search(query: str) -> str:
    """Search the live web. USE FOR: news, recent events, current prices,
    anything after your training cutoff, or when you are unsure of a fact.
    DO NOT use for math or for reading a known URL. Returns titles + snippets + URLs."""
    if query in _search_cache:                  # cache hit -> instant
        return _search_cache[query]
    results = DDGS().text(query, max_results=3)
    if not results:
        return "No results found."
    out = "\n\n".join(
        f"{r['title']}\n{r['body']}\n{r['href']}" for r in results
    )
    _search_cache[query] = out
    return out


@tool
def fetch_url(url: str) -> str:
    """Read ONE specific webpage. USE FOR: when you already have an exact URL
    and want its text. DO NOT use to explore a site (use deep_crawl) or to
    search (use web_search). Returns the page's text."""
    text = _get_page(url)
    return text[:2500] if text else f"Could not fetch {url}."


@tool
def deep_crawl(start_url: str, max_pages: int = 8) -> str:
    """Crawl a WHOLE website or section. USE FOR: exploring many pages of a site
    at once. DO NOT use for a single page (use fetch_url). Follows same-domain
    links, respects robots.txt, max_pages capped at 20. Public pages only."""
    max_pages = min(max_pages, 20)
    domain = urlparse(start_url).netloc

    rp = RobotFileParser()
    try:
        rp.set_url(f"{urlparse(start_url).scheme}://{domain}/robots.txt")
        rp.read()
    except Exception:
        rp = None

    queue = deque([start_url])
    seen = {start_url}
    pages = []

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        if rp and not rp.can_fetch("*", url):
            continue
        try:
            resp = _session.get(url, timeout=15)
            resp.raise_for_status()
        except Exception:
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        _page_cache[url] = text                 # crawled pages feed the cache
        pages.append(f"=== {url} ===\n{text[:1500]}")

        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"]).split("#")[0]
            if urlparse(link).netloc == domain and link not in seen:
                seen.add(link)
                queue.append(link)

        time.sleep(1)  # polite delay -> don't hammer the server

    if not pages:
        return f"Could not crawl {start_url} (blocked, offline, or no pages)."
    return f"Crawled {len(pages)} pages:\n\n" + "\n\n".join(pages)


# --- safe calculator (no python eval -> eval is a security hole) ---
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Do EXACT arithmetic. USE FOR: any math, always — never compute numbers
    in your head. Input like '12*(3+4)/2 - 5**2'. Returns the exact result."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Could not compute '{expression}': {e}"


@tool
def wikipedia_lookup(topic: str) -> str:
    """Look up an established fact on Wikipedia. USE FOR: definitions, history,
    science, well-known people/places. DO NOT use for recent news (use
    web_search). Returns a short summary."""
    try:
        # step 1: find the best-matching page title
        search = _session.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": topic,
                    "limit": 1, "format": "json"},
            timeout=10,
        ).json()
        if not search[1]:
            return f"No Wikipedia page for '{topic}'."
        title = search[1][0].replace(" ", "_")
        # step 2: fetch that page's summary
        summary = _session.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}",
            timeout=10,
        ).json()
        return summary.get("extract", f"No summary available for '{topic}'.")
    except Exception as e:
        return f"Wikipedia lookup failed for '{topic}': {e}"


@tool
def read_file(path: str) -> str:
    """Read one local text file by path. USE FOR: reading a specific file in
    the project folder. Only files inside the project are allowed."""
    base = os.path.abspath(os.getcwd())
    full = os.path.abspath(os.path.join(base, path))
    if not full.startswith(base):
        return "Access denied: path is outside the project folder."
    if not os.path.isfile(full):
        return f"No such file: {path}"
    with open(full, encoding="utf-8", errors="replace") as f:
        return f.read()[:4000]


# --- RAG: search the local knowledge base (files in the 'knowledge' folder) ---
_store = None  # built once, on first use (lazy -> fast import)


def _get_store():
    global _store
    if _store is not None:
        return _store
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    kdir = os.path.join(os.getcwd(), "knowledge")
    store = InMemoryVectorStore(FastEmbedEmbeddings())  # local, free embeddings
    if os.path.isdir(kdir):
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = []
        for fn in os.listdir(kdir):
            fp = os.path.join(kdir, fn)
            if os.path.isfile(fp):
                text = open(fp, encoding="utf-8", errors="replace").read()
                chunks += splitter.split_text(text)
        if chunks:
            store.add_texts(chunks)  # embed every chunk into the vector store
    _store = store
    return _store


@tool
def search_docs(query: str) -> str:
    """Search the project's own knowledge base (files in the 'knowledge'
    folder) by MEANING. USE FOR: project-specific notes, owner preferences,
    internal docs. DO NOT use for general or web facts. Returns top matches."""
    hits = _get_store().similarity_search(query, k=3)
    if not hits:
        return "No relevant documents in the knowledge base."
    return "\n\n---\n\n".join(h.page_content for h in hits)


if __name__ == "__main__":
    print("search :", web_search.invoke({"query": "LangGraph"})[:100])
    print("math   :", calculator.invoke({"expression": "12*(3+4)/2 - 5**2"}))
    print("wiki   :", wikipedia_lookup.invoke({"topic": "Mount Everest"})[:100])
    print("docs   :", search_docs.invoke({"query": "which model does the project use"})[:150])
