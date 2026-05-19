"""Step 10 — multi-agent: planner -> researcher -> writer."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from step9_agent import app as research_agent  # reuse the tool-agent as a worker

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)


# --- custom state: more than just messages now ---
class State(TypedDict):
    task: str            # the user's question
    plan: list[str]      # subtasks the planner produced
    findings: list[str]  # what the researcher found per subtask
    final: str           # the writer's combined answer


# --- node 1: planner — break the task into subtasks ---
def planner(state: State):
    prompt = [
        SystemMessage(
            "Break the user's question into 2-4 focused research subtasks. "
            "Output ONLY the subtasks, one per line, numbered. No other text."
        ),
        HumanMessage(state["task"]),
    ]
    text = llm.invoke(prompt).content
    # parse numbered lines into a clean list
    plan = []
    for line in text.splitlines():
        line = line.strip().lstrip("0123456789.)- ").strip()
        if line:
            plan.append(line)
    print(f"PLAN ({len(plan)} subtasks):")
    for i, p in enumerate(plan, 1):
        print(f"  {i}. {p}")
    return {"plan": plan}


# --- node 2: researcher — run the tool-agent on each subtask ---
def researcher(state: State):
    findings = []
    for i, subtask in enumerate(state["plan"], 1):
        print(f"\n[researching {i}: {subtask}]")
        # each subtask gets its own thread_id -> separate memory
        result = research_agent.invoke(
            {"messages": [HumanMessage(subtask)]},
            {"configurable": {"thread_id": f"sub-{i}"}, "recursion_limit": 12},
        )
        findings.append(result["messages"][-1].content)
    return {"findings": findings}


# --- node 3: writer — merge findings into one answer ---
def writer(state: State):
    joined = "\n\n".join(
        f"Subtask: {t}\nFinding: {f}"
        for t, f in zip(state["plan"], state["findings"])
    )
    prompt = [
        SystemMessage(
            "Combine the research findings into one clear, well-structured "
            "answer to the original question. Cite facts found."
        ),
        HumanMessage(f"Original question: {state['task']}\n\n{joined}"),
    ]
    return {"final": llm.invoke(prompt).content}


# --- build the graph: simple linear pipeline ---
graph = StateGraph(State)
graph.add_node("planner", planner)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_edge(START, "planner")
graph.add_edge("planner", "researcher")
graph.add_edge("researcher", "writer")
graph.add_edge("writer", END)
team = graph.compile()


if __name__ == "__main__":
    print("Multi-agent team. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        if not q:
            continue
        result = team.invoke({"task": q})
        print("\n" + "=" * 50)
        print("FINAL ANSWER:\n")
        print(result["final"], "\n")
