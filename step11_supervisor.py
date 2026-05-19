"""Step 11 — supervisor pattern: a boss LLM dynamically routes to workers."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from operator import add
from typing import Annotated, TypedDict

from llm_config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from step4_tool import web_search, wikipedia_lookup

llm = get_llm()

WORKERS = ["searcher", "wiki", "writer"]


class State(TypedDict):
    task: str
    gathered: Annotated[list[str], add]  # reducer 'add': every worker appends here
    next: str                            # which worker the supervisor chose
    final: str


# --- supervisor: decide who works next ---
def supervisor(state: State):
    done = "\n".join(state.get("gathered", [])) or "(nothing yet)"
    prompt = [
        SystemMessage(
            "You manage a research team. Workers:\n"
            "- searcher: web search for current info\n"
            "- wiki: Wikipedia for established facts\n"
            "- writer: writes the final answer\n"
            "Look at what's gathered. If enough info, choose 'writer'. "
            "Otherwise choose a worker that adds NEW info. Don't repeat a worker "
            "that already ran unless truly needed. Reply with ONE word only."
        ),
        HumanMessage(f"Task: {state['task']}\n\nGathered so far:\n{done}"),
    ]
    choice = llm.invoke(prompt).content.strip().lower()
    choice = next((w for w in WORKERS if w in choice), "writer")  # safe parse
    print(f"[supervisor -> {choice}]")
    return {"next": choice}


def route(state: State) -> str:
    return state["next"]  # conditional edge reads this


# --- workers ---
def searcher(state: State):
    res = web_search.invoke({"query": state["task"]})
    return {"gathered": [f"[web search]\n{res}"]}


def wiki(state: State):
    res = wikipedia_lookup.invoke({"topic": state["task"]})
    return {"gathered": [f"[wikipedia]\n{res}"]}


def writer(state: State):
    prompt = [
        SystemMessage("Write a clear, complete answer using the gathered research."),
        HumanMessage(f"Task: {state['task']}\n\n" + "\n\n".join(state["gathered"])),
    ]
    return {"final": llm.invoke(prompt).content}


# --- graph ---
graph = StateGraph(State)
graph.add_node("supervisor", supervisor)
graph.add_node("searcher", searcher)
graph.add_node("wiki", wiki)
graph.add_node("writer", writer)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route, WORKERS)  # dynamic routing
graph.add_edge("searcher", "supervisor")   # workers report back to the boss
graph.add_edge("wiki", "supervisor")
graph.add_edge("writer", END)

team = graph.compile()


if __name__ == "__main__":
    print("Supervisor team. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        if not q:
            continue
        result = team.invoke({"task": q}, {"recursion_limit": 15})
        print("\n" + "=" * 50)
        print("FINAL ANSWER:\n")
        print(result["final"], "\n")
