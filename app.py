"""Web UI for the research agent — a Gradio chat app.

Run:  venv\\Scripts\\python app.py
Then open the local URL it prints (http://127.0.0.1:7860).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr
from langchain_core.messages import HumanMessage

from step14_reflect import app as agent  # the reflection agent (7 tools)


def chat(message, history):
    """Gradio calls this per user message. Yields partial output -> live updates."""
    config = {"configurable": {"thread_id": "gradio-session"},  # fixed id -> memory
              "recursion_limit": 30}
    log = ""        # running list of tool/reflection activity
    answer = ""     # latest answer text

    try:
        for step in agent.stream(
            {"messages": [HumanMessage(message)], "review_count": 0},
            config, stream_mode="values",
        ):
            last = step["messages"][-1]
            if last.type == "ai" and last.tool_calls:
                names = ", ".join(tc["name"] for tc in last.tool_calls)
                log += f"🔧 {names}\n"
                yield f"{log}\n_thinking..._"
            elif last.type == "ai" and last.content:
                answer = last.content
                yield f"{log}\n\n{answer}" if log else answer
    except Exception as e:
        yield f"⚠️ Error: {e}"
        return

    yield f"{log}\n\n{answer}" if log else (answer or "(no answer)")


demo = gr.ChatInterface(
    fn=chat,
    title="🔎 Research Agent",
    description=("LangGraph agent — web search, Wikipedia, calculator, web crawler, "
                 "and a local knowledge base (RAG). It self-reviews its answers."),
    examples=[
        "What is the latest version of Python?",
        "What is 17% of 4830?",
        "Tell me about black holes",
        "What model does this project use?",
    ],
)

if __name__ == "__main__":
    demo.launch()  # add share=True for a public link
