"""Step 12 — parallel researchers: fan out subtasks with Send, run them at once."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from operator import add
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from llm_config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send  # the fan-out primitive

from step9_agent import app as research_agent

load_dotenv()
llm = get_llm()


class State(TypedDict):
    task: str
    plan: list[str]
    # reducer 'add' -> parallel branches each append; lists get merged, not overwritten
    findings: Annotated[list[str], add]
    final: str


# --- planner: make the subtasks ---
def planner(state: State):
    prompt = [
        SystemMessage(
            "Break the question into exactly 2-3 independent research subtasks. "
            "One per line, numbered. Output ONLY the subtasks."
        ),
        HumanMessage(state["task"]),
    ]
    text = llm.invoke(prompt).content
    plan = []
    for line in text.splitlines():
        line = line.strip().lstrip("0123456789.)- ").strip()
        if line:
            plan.append(line)
    print(f"PLAN: {plan}")
    return {"plan": plan}


# --- fan-out: one Send per subtask -> researchers run in PARALLEL ---
def fan_out(state: State):
    return [Send("researcher", {"subtask": s}) for s in state["plan"]]


# --- researcher: handles ONE subtask (many run at once) ---
def researcher(state: dict):
    subtask = state["subtask"]
    print(f"[researching: {subtask}]")
    result = research_agent.invoke(
        {"messages": [HumanMessage(subtask)]},
        {"configurable": {"thread_id": f"p-{abs(hash(subtask))}"},
         "recursion_limit": 12},
    )
    text = result["messages"][-1].content
    print(f"[DONE: {subtask[:50]}...]")  # show progress as each finishes
    return {"findings": [f"Subtask: {subtask}\nFinding: {text}"]}


# --- writer: merge all findings ---
def writer(state: State):
    prompt = [
        SystemMessage("Combine the findings into one clear answer. Cite facts."),
        HumanMessage(f"Question: {state['task']}\n\n" + "\n\n".join(state["findings"])),
    ]
    return {"final": llm.invoke(prompt).content}


graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)

graph.add_edge(START, "planner")
# conditional edge returns a list of Sends -> parallel branches
graph.add_conditional_edges("planner", fan_out, ["researcher"])
graph.add_edge("researcher", "writer")  # writer waits for ALL researchers
graph.add_edge("writer", END)

team = graph.compile()


if __name__ == "__main__":
    print("Parallel research team. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        if not q:
            continue
        import time
        t0 = time.time()
        result = team.invoke({"task": q})
        print("\n" + "=" * 50)
        print(f"FINAL ANSWER ({time.time() - t0:.1f}s):\n")
        print(result["final"], "\n")
