"""Web UI for the research agent — a Gradio chat app.

Run:  venv\\Scripts\\python app.py
Then open the local URL it prints (http://127.0.0.1:7860).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import uuid

import gradio as gr
from langchain_core.messages import HumanMessage

from step14_reflect import app as agent, tools  # reflection agent + its 7 tools

# build the tools panel text from each tool's name + first docstring sentence
TOOL_PANEL = "### 🛠️ Tools\n" + "\n".join(
    f"- **{t.name}** — {t.description.split('.')[0].strip()}." for t in tools
)


def bot_reply(chat_history, session_id):
    """Stream the agent's response into the last (empty) assistant message."""
    message = chat_history[-1]["content"]
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 30}
    chat_history = chat_history + [{"role": "assistant", "content": ""}]

    log, answer = "", ""
    try:
        for step in agent.stream(
            {"messages": [HumanMessage(message)], "review_count": 0},
            config, stream_mode="values",
        ):
            last = step["messages"][-1]
            if last.type == "ai" and last.tool_calls:
                names = ", ".join(tc["name"] for tc in last.tool_calls)
                log += f"🔧 {names}\n"
                chat_history[-1]["content"] = f"{log}\n_thinking..._"
                yield chat_history
            elif last.type == "ai" and last.content:
                answer = last.content
                chat_history[-1]["content"] = f"{log}\n\n{answer}".strip()
                yield chat_history
    except Exception as e:
        chat_history[-1]["content"] = f"⚠️ Error: {e}"
        yield chat_history
        return

    chat_history[-1]["content"] = f"{log}\n\n{answer}".strip() or "(no answer)"
    yield chat_history


def user_submit(message, chat_history):
    """Add the user message; clear the input box."""
    if not message.strip():
        return "", chat_history
    return "", chat_history + [{"role": "user", "content": message}]


def new_chat():
    """Reset the conversation and give it a fresh memory thread."""
    return [], str(uuid.uuid4())


with gr.Blocks(title="Research Agent") as demo:
    gr.Markdown(
        "# 🔎 Research Agent\n"
        "A LangGraph agent that searches the web, reads pages, does math, "
        "checks Wikipedia, and reviews its own answers."
    )
    # per-browser-session memory id; New Chat makes a fresh one
    session = gr.State(lambda: str(uuid.uuid4()))

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=460)
            msg = gr.Textbox(placeholder="Ask anything...", show_label=False,
                             autofocus=True)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("🔄 New Chat")
        with gr.Column(scale=1):
            gr.Markdown(TOOL_PANEL)

    gr.Examples(
        examples=["What is the latest version of Python?",
                  "What is 17% of 4830?",
                  "Tell me about black holes",
                  "What model does this project use?"],
        inputs=msg,
    )

    # submit (Enter) and Send button -> add user msg, then stream the reply
    for trigger in (msg.submit, send.click):
        trigger(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_reply, [chatbot, session], chatbot)
    clear.click(new_chat, None, [chatbot, session])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())  # add share=True for a public link
