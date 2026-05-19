"""Step 14 — agent with reflection: it reviews its own answer and revises if weak.
Uses all 7 tools (incl. search_docs RAG)."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from groq import BadRequestError

from step4_tool import (web_search, fetch_url, deep_crawl, calculator,
                        wikipedia_lookup, read_file, search_docs)

load_dotenv()

tools = [web_search, fetch_url, deep_crawl, calculator,
         wikipedia_lookup, read_file, search_docs]
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(
    "You are a research assistant. Tools: web_search, fetch_url, deep_crawl, "
    "calculator, wikipedia_lookup, read_file, search_docs (project knowledge base). "
    "Pick the right tool per its description. Answer clearly, citing what you found."
)

MAX_REVIEWS = 2  # how many times the agent may revise


# --- state: messages + a counter for how many revisions happened ---
class State(MessagesState):
    review_count: int


def agent(state: State):
    """Call the model; retry on a broken tool call; give up gracefully."""
    msgs = [SYSTEM] + state["messages"]
    for attempt in range(3):
        try:
            return {"messages": [llm_with_tools.invoke(msgs)]}
        except BadRequestError as e:
            if "tool_use_failed" in str(e) and attempt < 2:
                continue
            return {"messages": [AIMessage("I had trouble using my tools.")]}


def reflect(state: State):
    """Review the latest answer. If weak, push feedback back to the agent."""
    count = state.get("review_count", 0)
    answer = state["messages"][-1].content
    question = next((m.content for m in state["messages"] if m.type == "human"), "")

    if count >= MAX_REVIEWS:                 # out of revisions -> accept
        print(f"  [reflection: max reached, accepting]")
        return {"review_count": count}

    critique = llm.invoke([
        SystemMessage(
            "You are a strict reviewer. Does the answer fully and correctly "
            "answer the question? If yes, reply EXACTLY 'GOOD'. If not, give "
            "short specific instructions to fix it."
        ),
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


# --- routing ---
def route_after_agent(state: State) -> str:
    # model asked for a tool? -> tools. else -> reflect on the answer.
    return "tools" if state["messages"][-1].tool_calls else "reflect"


def route_after_reflect(state: State) -> str:
    # reflect appended feedback (a human message)? -> back to agent. else -> done.
    return "agent" if state["messages"][-1].type == "human" else END


# --- graph ---
graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_node("reflect", reflect)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route_after_agent, ["tools", "reflect"])
graph.add_edge("tools", "agent")
graph.add_conditional_edges("reflect", route_after_reflect, ["agent", END])

app = graph.compile(checkpointer=InMemorySaver())


def ask(question: str, thread_id: str = "chat-1") -> str:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}
    final = None
    try:
        for step in app.stream({"messages": [HumanMessage(question)],
                                "review_count": 0}, config, stream_mode="values"):
            last = step["messages"][-1]
            if last.type == "ai" and last.tool_calls:
                for tc in last.tool_calls:
                    print(f"  [{tc['name']}: {list(tc['args'].values())}]")
            final = last
    except Exception as e:
        return f"(stopped: {e})"
    return final.content


if __name__ == "__main__":
    print("Reflection agent — 7 tools, self-reviewing. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        if not q:
            continue
        print("...thinking...")
        print("\nAgent:", ask(q), "\n")
