"""Research Agent — model config + tools + reflection agent + web UI, in one file.

Run:  venv\\Scripts\\python agent.py      -> launches the web UI at http://127.0.0.1:7860

Sections:
  1. Model config   — which LLM (NVIDIA NIM)
  2. Tools          — the 7 things the agent can do
  3. Agent          — the reflection agent graph (LangGraph)
  4. Web UI         — Gradio chat app (only built when run directly)
"""

import ast
import atexit
import operator
import os
import sys
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

sys.stdout.reconfigure(encoding="utf-8")  # let Windows console print Unicode
load_dotenv()  # loads NVIDIA_API_KEY from .env


# ============================================================
# 1. MODEL CONFIG  — swap the model here, in one place
# ============================================================
MODEL = "nvidia/nemotron-3-super-120b-a12b"  # NVIDIA NIM — Nemotron Super


def get_llm(temperature: float = 0):
    """Return the project's chat model. Change MODEL above to swap everywhere."""
    return ChatNVIDIA(model=MODEL, temperature=temperature)


# ============================================================
# 2. TOOLS
# ============================================================
# shared HTTP session: one connection pool reused by every tool
_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (LearningAgent)"})

# caches: avoid repeating the same network call
_search_cache: dict[str, str] = {}
_page_cache: dict[str, str] = {}

# shared headless browser, launched lazily and reused across calls
# -> Chromium startup is the slow part (~3s); reusing it makes JS rendering fast
_browser = None
_playwright_ctx = None


def _clean_html(html: str, drop_extra=()) -> str:
    """Strip tags/noise from HTML, return squashed plain text (fallback path)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", *drop_extra]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def _get_page(url: str) -> str:
    """Fetch a URL's main content as clean text (cached). Returns '' on failure.

    Uses trafilatura, the best-in-class article extractor — drops nav, ads,
    cookie banners, footers. Falls back to BeautifulSoup if trafilatura fails.
    Network failures are NOT cached -> transient errors get a retry next time.
    """
    if url in _page_cache:
        return _page_cache[url]
    try:
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return ""  # don't cache network/HTTP failures
    text = (trafilatura.extract(resp.text, include_comments=False,
                                include_tables=True)
            or _clean_html(resp.text)
            or "")
    _page_cache[url] = text
    return text


def _get_browser():
    """Lazily launch Chromium once; reuse across all JS-render calls."""
    global _browser, _playwright_ctx
    if _browser is None:
        from playwright.sync_api import sync_playwright
        _playwright_ctx = sync_playwright().start()
        _browser = _playwright_ctx.chromium.launch(headless=True)
    return _browser


def _render_js(url: str, timeout_ms: int = 30000) -> tuple[str, list[str]]:
    """Render a URL in the shared browser. Returns (visible text, links)."""
    browser = _get_browser()
    page = browser.new_page()
    try:
        page.goto(url, timeout=timeout_ms, wait_until="networkidle")
        text = page.inner_text("body")
        links = [a.get_attribute("href") for a in page.query_selector_all("a")]
        links = [link for link in links if link]
    finally:
        page.close()
    return text, links


def _cleanup_browser():
    """Close the shared browser on Python exit."""
    global _browser, _playwright_ctx
    try:
        if _browser is not None:
            _browser.close()
    except Exception:
        pass
    try:
        if _playwright_ctx is not None:
            _playwright_ctx.stop()
    except Exception:
        pass


atexit.register(_cleanup_browser)


@tool
def web_search(query: str) -> str:
    """Search the live web. USE FOR: news, recent events, current prices,
    anything after your training cutoff, or when you are unsure of a fact.
    DO NOT use for math or for reading a known URL. Returns titles + snippets + URLs."""
    if query in _search_cache:
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
    search (use web_search). Auto-falls back to a headless browser if the
    static fetch returns too little content (so JS-rendered sites still work).
    Returns the page's text."""
    text = _get_page(url)
    # too little content -> probably JS-rendered. Auto-escalate.
    if len(text) < 300:
        try:
            js_text, _ = _render_js(url)
            if len(js_text) > len(text):
                text = js_text
                _page_cache[url] = text
        except Exception:
            pass
    return text[:3000] if text else f"Could not fetch {url}."


@tool
def js_fetch(url: str) -> str:
    """Force a JavaScript render. Returns the page's visible text + its links.
    USE FOR: when you explicitly need browser-rendered output, or when
    fetch_url's auto-fallback was not enough. Slightly slower than fetch_url."""
    cache_key = f"js::{url}"
    if cache_key in _page_cache:
        return _page_cache[cache_key]
    try:
        text, links = _render_js(url)
    except Exception as e:
        return f"Could not render {url}: {e}"
    links = [link for link in links if link.startswith("http")][:20]
    out = f"{text[:3000]}\n\nLinks found:\n" + "\n".join(links)
    _page_cache[cache_key] = out
    return out


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
        _page_cache[url] = text
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


@tool
def js_crawl(start_url: str, max_pages: int = 5) -> str:
    """Crawl a JavaScript-rendered website (React, Next.js, Vercel, SPAs).
    USE FOR: multi-page exploration of a JS site where deep_crawl returns
    nothing. Same-domain only, max_pages capped at 10. Slower than deep_crawl
    but sees what a real browser sees."""
    max_pages = min(max_pages, 10)
    domain = urlparse(start_url).netloc
    queue = deque([start_url])
    seen = {start_url}
    pages = []

    while queue and len(pages) < max_pages:
        url = queue.popleft()
        try:
            text, links = _render_js(url, timeout_ms=20000)
        except Exception:
            continue
        pages.append(f"=== {url} ===\n{text[:1500]}")

        for link in links:
            full = urljoin(url, link).split("#")[0]
            if (full.startswith("http") and urlparse(full).netloc == domain
                    and full not in seen):
                seen.add(full)
                queue.append(full)
        time.sleep(0.5)  # polite delay -> don't hammer the server

    if not pages:
        return f"Could not js_crawl {start_url} (blocked or offline)."
    return f"JS-crawled {len(pages)} pages:\n\n" + "\n\n".join(pages)


# safe calculator (no python eval -> eval is a security hole)
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
        search = _session.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "opensearch", "search": topic,
                    "limit": 1, "format": "json"},
            timeout=10,
        ).json()
        if not search[1]:
            return f"No Wikipedia page for '{topic}'."
        title = search[1][0].replace(" ", "_")
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


# RAG: search the local knowledge base (files in the 'knowledge' folder)
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
            store.add_texts(chunks)
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


# ============================================================
# 3. AGENT  — reflection agent: answers, reviews itself, revises
# ============================================================
tools = [web_search, fetch_url, js_fetch, deep_crawl, js_crawl, calculator,
         wikipedia_lookup, read_file, search_docs]
llm = get_llm()
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(
    "You are a research assistant. Tools: web_search, fetch_url (auto-handles "
    "JS sites), js_fetch (force JS render), deep_crawl (multi-page static), "
    "js_crawl (multi-page JS), calculator, wikipedia_lookup, read_file, "
    "search_docs (project knowledge base). Pick the right tool per its "
    "description. Prefer fetch_url first; if a site needs many pages, choose "
    "deep_crawl or js_crawl. "
    "STRICT LIMITS: use at most 4 tool calls per question, then ANSWER. Do "
    "NOT repeat the same tool call. If a tool returned content, USE that "
    "content — do not refetch. After you have enough information, write the "
    "final answer immediately. Cite what you found."
)

MAX_REVIEWS = 2  # how many times the agent may revise


class State(MessagesState):
    review_count: int


def agent_node(state: State):
    """Call the model; retry on a failed tool call; give up gracefully."""
    msgs = [SYSTEM] + state["messages"]
    for attempt in range(3):
        try:
            return {"messages": [llm_with_tools.invoke(msgs)]}
        except Exception as e:
            if attempt < 2:
                continue
            return {"messages": [AIMessage(f"I had trouble using my tools: {e}")]}


def reflect(state: State):
    """Review the latest answer. If weak, push feedback back to the agent."""
    count = state.get("review_count", 0)
    answer = state["messages"][-1].content
    question = next((m.content for m in state["messages"] if m.type == "human"), "")

    if count >= MAX_REVIEWS:
        print("  [reflection: max reached, accepting]")
        return {"review_count": count}

    critique = llm.invoke([
        SystemMessage(
            "You are a strict reviewer. Does the answer fully and correctly "
            "answer the question? If yes, reply EXACTLY 'GOOD'. If not, give "
            "short specific instructions to fix it."),
        HumanMessage(f"Question: {question}\n\nAnswer: {answer}"),
    ]).content.strip()

    if critique.upper().startswith("GOOD"):
        print(f"  [reflection {count + 1}: GOOD]")
        return {"review_count": count}

    print(f"  [reflection {count + 1}: revising -> {critique[:70]}...]")
    return {
        "messages": [HumanMessage(f"Revise your answer. Reviewer feedback: {critique}")],
        "review_count": count + 1,
    }


def route_after_agent(state: State) -> str:
    return "tools" if state["messages"][-1].tool_calls else "reflect"


def route_after_reflect(state: State) -> str:
    return "agent" if state["messages"][-1].type == "human" else END


_graph = StateGraph(State)
_graph.add_node("agent", agent_node)
_graph.add_node("tools", ToolNode(tools))
_graph.add_node("reflect", reflect)
_graph.add_edge(START, "agent")
_graph.add_conditional_edges("agent", route_after_agent, ["tools", "reflect"])
_graph.add_edge("tools", "agent")
_graph.add_conditional_edges("reflect", route_after_reflect, ["agent", END])

app = _graph.compile(checkpointer=InMemorySaver())  # the runnable agent


def ask(question: str, thread_id: str = "default") -> str:
    """Run the agent on one question and return the final answer text."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    result = app.invoke({"messages": [HumanMessage(question)], "review_count": 0},
                        config)
    return result["messages"][-1].content


# ============================================================
# 4. CLI fallback  — for quick testing without the web UI
# ============================================================
# For the Claude-style web UI, run `python app.py` instead.
if __name__ == "__main__":
    print("Quick CLI mode. For the web UI, run:  python app.py")
    print("Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            print("\nAgent:", ask(q), "\n")
