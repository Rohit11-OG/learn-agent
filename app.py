"""Web UI for the research agent — Claude-style Gradio app.

Run:  venv\\Scripts\\python app.py
Then open the local URL it prints (http://127.0.0.1:7860).
"""

import uuid

import gradio as gr
from langchain_core.messages import HumanMessage

from agent import app as agent_graph, tools  # the compiled LangGraph + tools list


TOOL_PANEL = "### 🛠️ Tools\n" + "\n".join(
    f"- **{t.name}** — {t.description.split('.')[0].strip()}." for t in tools
)


def bot_reply(chat_history):
    """Stream the agent's response into a new assistant message.

    Fresh thread_id per send -> no memory contamination between
    unrelated questions.
    """
    if not chat_history or chat_history[-1].get("role") != "user":
        yield chat_history
        return
    message = chat_history[-1]["content"]
    thread_id = str(uuid.uuid4())  # fresh memory each send
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    chat_history = chat_history + [{"role": "assistant", "content": ""}]

    log, answer = "", ""
    try:
        for step in agent_graph.stream(
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
    if not message.strip():
        return "", chat_history
    return "", chat_history + [{"role": "user", "content": message}]


def new_chat():
    return []  # clear chatbot display; memory is already fresh per send


# --- Modern indigo/slate design ---
CLAUDE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

body, .gradio-container, gradio-app {
    background:
        radial-gradient(circle at 15% -10%, rgba(99,102,241,0.12), transparent 45%),
        radial-gradient(circle at 110% 0%, rgba(236,72,153,0.08), transparent 50%),
        #f8fafc !important;
    min-height: 100vh !important;
}

.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 0 32px 32px 32px !important;
}
* { box-sizing: border-box; }

/* ===== top nav ===== */
.nav {
    display: flex; justify-content: space-between; align-items: center;
    padding: 18px 24px; margin: 0 -32px 28px -32px;
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
    border-bottom: 1px solid #e2e8f0;
    position: sticky; top: 0; z-index: 50;
}
.nav-brand { display: flex; align-items: center; gap: 12px;
    font-weight: 700; font-size: 17px; color: #0f172a; letter-spacing: -0.01em; }
.nav-logo {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 18px;
    box-shadow: 0 4px 12px rgba(99,102,241,0.35);
}
.nav-badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.badge {
    background: #ffffff; color: #475569;
    padding: 6px 12px; border-radius: 999px;
    font-size: 12px; font-weight: 500;
    border: 1px solid #e2e8f0;
}
.badge-dot {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: #10b981; margin-right: 7px;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.18);
}
.badge-mono { font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }

/* ===== hero ===== */
.hero { text-align: center; padding: 56px 24px 40px 24px; }
.hero-eyebrow {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: #6366f1; letter-spacing: 0.18em; text-transform: uppercase;
    background: rgba(99,102,241,0.08);
    padding: 6px 14px; border-radius: 999px;
    margin-bottom: 18px;
    border: 1px solid rgba(99,102,241,0.2);
}
.hero h1 {
    font-size: 52px !important; font-weight: 700 !important;
    color: #0f172a !important; margin: 0 0 16px 0 !important;
    letter-spacing: -0.03em !important; line-height: 1.05 !important;
}
.hero h1 .grad {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: #475569; font-size: 18px; max-width: 660px;
    margin: 0 auto !important; line-height: 1.6;
}

/* headings + text */
.gradio-container h1, .gradio-container h2, .gradio-container h3 {
    color: #0f172a !important; font-weight: 600 !important;
    letter-spacing: -0.01em !important;
}
.gradio-container p, .gradio-container li, .gradio-container label {
    color: #475569 !important;
}

/* buttons */
button.primary, .gr-button-primary {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important; border-radius: 12px !important;
    font-weight: 600 !important; padding: 13px 22px !important;
    font-size: 14px !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
    transition: transform 0.1s ease, box-shadow 0.2s ease !important;
}
button.primary:hover, .gr-button-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
}
.gr-button-secondary, button.secondary {
    background: #ffffff !important; color: #334155 !important;
    border: 1px solid #cbd5e1 !important; border-radius: 12px !important;
    font-weight: 500 !important;
}
.gr-button-secondary:hover, button.secondary:hover {
    border-color: #6366f1 !important; color: #6366f1 !important;
}

/* chat card */
.chat-card {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(8px);
    border: 1px solid #e2e8f0 !important;
    border-radius: 20px !important; padding: 12px !important;
    box-shadow: 0 4px 24px rgba(15,23,42,0.05) !important;
}
.chatbot, .gr-chatbot, .message-wrap {
    background: transparent !important; border: none !important;
}
.message.user, [data-testid="user"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important; border-radius: 14px !important;
}
.message.user *, [data-testid="user"] * { color: white !important; }
.message.bot, [data-testid="bot"] {
    background: #ffffff !important; border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
}

/* inputs */
input, textarea, .gr-textbox textarea, .gr-textbox input {
    border-radius: 12px !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    font-family: inherit !important;
    font-size: 15px !important;
}
input:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}

/* sidebar cards */
.side-card {
    background: rgba(255, 255, 255, 0.85) !important;
    backdrop-filter: blur(8px);
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 22px !important;
    margin-bottom: 16px !important;
    box-shadow: 0 4px 24px rgba(15,23,42,0.04) !important;
}
.side-card h3 { margin-top: 0 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.12em !important;
    color: #6366f1 !important; font-weight: 700 !important;
    margin-bottom: 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.side-card ul { padding-left: 18px !important; line-height: 1.85 !important;
    margin: 0 !important; }
.side-card strong { color: #0f172a !important; font-weight: 600 !important; }
.side-card code {
    background: #f1f5f9; color: #6366f1; padding: 2px 7px;
    border-radius: 6px; font-size: 13px;
    font-family: 'JetBrains Mono', monospace !important;
}

/* examples */
.examples, .gr-examples { background: transparent !important; }
.gr-examples button {
    background: #ffffff !important; border: 1px solid #e2e8f0 !important;
    color: #475569 !important; border-radius: 10px !important;
    font-weight: 500 !important;
}
.gr-examples button:hover {
    border-color: #6366f1 !important; color: #6366f1 !important;
    background: rgba(99,102,241,0.04) !important;
}

/* footer */
.footer {
    text-align: center; padding: 36px 0 16px 0;
    color: #64748b; font-size: 13px;
    border-top: 1px solid #e2e8f0; margin-top: 48px;
}
.footer a { color: #6366f1; text-decoration: none; font-weight: 500; }
.footer a:hover { text-decoration: underline; }

footer { display: none !important; }
"""

NAV_HTML = """
<div class="nav">
  <div class="nav-brand">
    <div class="nav-logo">⌘</div>
    <div>Research Agent</div>
  </div>
  <div class="nav-badges">
    <span class="badge"><span class="badge-dot"></span>Online</span>
    <span class="badge badge-mono">nemotron-super-120b</span>
    <span class="badge">9 tools</span>
  </div>
</div>
"""

HERO_HTML = """
<div class="hero">
  <div class="hero-eyebrow">⚡ LangGraph · NVIDIA NIM</div>
  <h1>Ask anything. Get <span class="grad">grounded answers</span>.</h1>
  <p>A research agent that searches the web, reads pages, crawls JS sites,
  does math, checks Wikipedia, and reviews its own work before replying.</p>
</div>
"""

FOOTER_HTML = """
<div class="footer">
  Built with LangGraph + NVIDIA NIM ·
  <a href="https://github.com/Rohit11-OG/learn-agent" target="_blank">View source on GitHub</a>
</div>
"""

SIDE_TIPS = """
### Tips
- Use `js_fetch` for React / Next.js sites
- Click **New chat** between unrelated questions
- Agent reviews its own answers before replying
- Try the example prompts to start
"""

with gr.Blocks(title="Research Agent") as demo:
    gr.HTML(NAV_HTML)
    gr.HTML(HERO_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=3, elem_classes="chat-card"):
            chatbot = gr.Chatbot(height=560, show_label=False,
                                 avatar_images=(None, None))
            msg = gr.Textbox(placeholder="Ask anything...",
                             show_label=False, autofocus=True, lines=2)
            with gr.Row():
                send = gr.Button("Send", variant="primary", scale=4)
                clear = gr.Button("New chat", scale=1)

            gr.Examples(
                examples=[
                    "What is the latest version of Python?",
                    "What is 17% of 4830?",
                    "Tell me about black holes",
                    "Crawl example.com and summarize it",
                ],
                inputs=msg,
                label="Try one",
            )

        with gr.Column(scale=1):
            with gr.Column(elem_classes="side-card"):
                gr.Markdown(TOOL_PANEL)
            with gr.Column(elem_classes="side-card"):
                gr.Markdown(SIDE_TIPS)

    gr.HTML(FOOTER_HTML)

    for trigger in (msg.submit, send.click):
        trigger(user_submit, [msg, chatbot], [msg, chatbot]).then(
            bot_reply, chatbot, chatbot)
    clear.click(new_chat, None, chatbot)


if __name__ == "__main__":
    demo.launch(css=CLAUDE_CSS)
