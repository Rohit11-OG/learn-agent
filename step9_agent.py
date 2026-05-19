"""Step 9 — final research agent: 5 tools, error handling, loop guard, memory."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from llm_config import get_llm

from step4_tool import (web_search, fetch_url, calculator, wikipedia_lookup,
                        read_file, deep_crawl)

tools = [web_search, fetch_url, calculator, wikipedia_lookup, read_file, deep_crawl]
llm = get_llm()
llm_with_tools = llm.bind_tools(tools)

SYSTEM = SystemMessage(
    "You are a research assistant with tools: web_search (current info), "
    "fetch_url (read one page), deep_crawl (read many pages of a whole site), "
    "calculator (exact math), wikipedia_lookup (facts), read_file (local files). "
    "Pick the right tool. Use fetch_url for one page, deep_crawl to explore a "
    "whole site. Search at most twice per question, then answer clearly citing "
    "what you found. You remember earlier turns."
)


def agent(state: MessagesState):
    """Call the model; retry on broken tool calls; give up gracefully."""
    msgs = [SYSTEM] + state["messages"]
    for attempt in range(3):
        try:
            return {"messages": [llm_with_tools.invoke(msgs)]}
        except Exception as e:
            if attempt < 2:
                continue
            return {"messages": [AIMessage(
                "I had trouble using my tools. Try rephrasing the question."
            )]}


graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", tools_condition)
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=InMemorySaver())


def ask(question: str, thread_id: str = "chat-1") -> str:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}
    final = None
    try:
        for step in app.stream({"messages": [HumanMessage(question)]},
                               config, stream_mode="values"):
            last = step["messages"][-1]
            if last.type == "ai" and last.tool_calls:
                for tc in last.tool_calls:
                    print(f"  [{tc['name']}: {list(tc['args'].values())}]")
            final = last
    except Exception as e:
        return f"(stopped: {e})"
    return final.content


if __name__ == "__main__":
    print("Research agent v3 — 5 tools. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        if not q:
            continue
        print("...thinking...")
        print("\nAgent:", ask(q), "\n")
