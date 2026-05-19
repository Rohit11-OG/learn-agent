"""Research agent — a strong, source-grounded research pipeline.

Flow:  plan -> research -> synthesize -> review -> (revise | done)

It breaks a question into sub-questions, searches the web for each, reads the
actual source pages, then writes a structured report that cites every claim.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from agent import get_llm, _get_page  # reuse model config + cached fetcher

sys.stdout.reconfigure(encoding="utf-8")

# model is configured in agent.py (NVIDIA NIM — Nemotron Super)
llm = get_llm()

MAX_SUBQUESTIONS = 5
MAX_SOURCES_FETCHED = 6   # how many source pages to read in full
MAX_REVISIONS = 1         # research is slow -> at most one rewrite


class State(TypedDict):
    question: str
    subquestions: list[str]
    sources: list[dict]    # each: {title, url, snippet, content}
    report: str
    revisions: int
    feedback: str


# --- node 1: plan -> break the question into sub-questions ---
def plan(state: State):
    out = llm.invoke([
        SystemMessage(
            "Break the research question into 3-5 specific sub-questions that "
            "together fully cover it. One per line, numbered. Output ONLY them."),
        HumanMessage(state["question"]),
    ]).content
    subs = []
    for line in out.splitlines():
        line = line.strip().lstrip("0123456789.)-* ").strip()
        if line:
            subs.append(line)
    subs = subs[:MAX_SUBQUESTIONS]
    print("PLAN — sub-questions:")
    for s in subs:
        print(f"  - {s}")
    return {"subquestions": subs, "revisions": 0, "feedback": ""}


# --- node 2: research -> search each sub-question, read source pages ---
def research(state: State):
    sources, seen = [], set()
    for sq in state["subquestions"]:
        print(f"[search] {sq}")
        try:
            results = DDGS().text(sq, max_results=3)
        except Exception:
            results = []
        for r in results:
            url = r.get("href")
            if not url or url in seen:
                continue
            seen.add(url)
            sources.append({"title": r.get("title", ""), "url": url,
                            "snippet": r.get("body", ""), "content": ""})

    # read the full text of the first few sources -- IN PARALLEL
    to_read = sources[:MAX_SOURCES_FETCHED]
    print(f"[reading {len(to_read)} sources in parallel...]")
    with ThreadPoolExecutor(max_workers=6) as pool:
        texts = pool.map(lambda s: _get_page(s["url"]), to_read)
    for s, text in zip(to_read, texts):
        s["content"] = text[:3000]

    print(f"RESEARCH — collected {len(sources)} sources")
    return {"sources": sources}


# --- node 3: synthesize -> write a cited, structured report ---
def synthesize(state: State):
    numbered = ""
    for i, s in enumerate(state["sources"], 1):
        body = s["content"] or s["snippet"]
        numbered += f"[{i}] {s['title']} — {s['url']}\n{body[:1500]}\n\n"

    extra = state.get("feedback", "")
    report = llm.invoke([
        SystemMessage(
            "You are a research analyst. Write a structured report answering "
            "the question using ONLY the numbered sources below. Cite every "
            "claim inline like [1] or [2,4]. Do NOT invent facts or sources. "
            "Structure the report as:\n"
            "## Summary\n## Key Findings (with sub-sections)\n## Conclusion\n"
            "## Sources (the numbered list with URLs)"),
        HumanMessage(f"Question: {state['question']}\n\n{extra}\n\n"
                     f"Numbered sources:\n{numbered}"),
    ]).content
    return {"report": report}


# --- node 4: review -> check coverage + citations, request a rewrite if weak ---
def review(state: State):
    n = state.get("revisions", 0)
    if n >= MAX_REVISIONS:
        print("[review] revision limit reached, accepting")
        return {"feedback": ""}

    verdict = llm.invoke([
        SystemMessage(
            "Review the research report. Check: (a) it answers every "
            "sub-question, (b) claims have inline [n] citations, (c) it has a "
            "Sources section. If all good reply EXACTLY 'GOOD'. Otherwise give "
            "short, specific fix instructions."),
        HumanMessage(f"Sub-questions: {state['subquestions']}\n\n"
                     f"Report:\n{state['report']}"),
    ]).content.strip()

    if verdict.upper().startswith("GOOD"):
        print("[review] GOOD")
        return {"feedback": ""}
    print(f"[review] revising -> {verdict[:80]}...")
    return {"revisions": n + 1,
            "feedback": f"A reviewer found issues. Fix them: {verdict}"}


def route_after_review(state: State) -> str:
    # feedback set -> rewrite. empty -> done.
    return "synthesize" if state.get("feedback") else END


# --- build the graph ---
graph = StateGraph(State)
graph.add_node("plan", plan)
graph.add_node("research", research)
graph.add_node("synthesize", synthesize)
graph.add_node("review", review)

graph.add_edge(START, "plan")
graph.add_edge("plan", "research")
graph.add_edge("research", "synthesize")
graph.add_edge("synthesize", "review")
graph.add_conditional_edges("review", route_after_review, ["synthesize", END])

app = graph.compile()


def run(question: str) -> str:
    result = app.invoke({"question": question}, {"recursion_limit": 20})
    return result["report"]


if __name__ == "__main__":
    print("Research agent. Enter a research question. Type 'quit' to exit.\n")
    while True:
        q = input("Research question: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        if not q:
            continue
        print("\n...researching (this takes a minute)...\n")
        print("=" * 60)
        print(run(q))
        print("=" * 60, "\n")
