"""Step 13 — stream the agent's answer token-by-token (like ChatGPT typing)."""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage

from step9_agent import app  # reuse the single ReAct agent from step 9


def ask_streaming(question: str, thread_id: str = "chat-1"):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 12}
    inputs = {"messages": [HumanMessage(question)]}

    # two stream modes at once:
    #   "messages" -> token chunks as the model writes them
    #   "updates"  -> fires once per node, so we can flag tool runs
    for mode, data in app.stream(inputs, config,
                                 stream_mode=["updates", "messages"]):
        if mode == "messages":
            chunk, meta = data
            # only print text from the 'agent' node (the answer), not tool noise
            if meta["langgraph_node"] == "agent" and chunk.content:
                print(chunk.content, end="", flush=True)  # flush -> instant print
        elif mode == "updates":
            if "tools" in data:               # the tools node just ran
                print("\n  [tool ran, thinking more...]")
    print()  # final newline


if __name__ == "__main__":
    print("Streaming agent. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        if not q:
            continue
        print("\nAgent: ", end="", flush=True)
        ask_streaming(q)
        print()
