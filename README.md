# learn-agent

A **research agent** built with [LangGraph](https://www.langchain.com/langgraph)
and the [NVIDIA NIM](https://build.nvidia.com) API.

Give it a question — it decides which tools to use (web search, web crawler,
Wikipedia, a calculator, a local knowledge base), gathers information, reviews
its own answer, and replies. It also has a dedicated **research pipeline** that
produces structured, source-cited reports.

---

## What is an AI agent?

```
AI agent = LLM (brain) + tools (hands) + a loop
```

The model thinks, calls a tool, sees the result, and repeats until the task is
done. Unlike a plain chatbot (one question → one answer), an agent pursues a
goal over many steps and corrects itself along the way.

---

## Features

- **7 tools** the agent can choose from
- **Reflection** — the agent reviews and revises its own answers
- **RAG** — semantic search over a local knowledge base
- **Research pipeline** — plan → search → read sources → cited report
- **Web UI** — a Gradio chat app
- **Memory** within a conversation (per session)
- Caching, connection pooling, parallel fetching for speed
- Error handling, retries, and loop guards

---

## The tools

| Tool | What it does |
|------|--------------|
| `web_search` | Live web search (DuckDuckGo) for current info |
| `fetch_url` | Read the text of one specific webpage |
| `deep_crawl` | Crawl a whole site — follows same-domain links, respects `robots.txt` |
| `calculator` | Exact arithmetic via a safe expression evaluator |
| `wikipedia_lookup` | Summaries from the Wikipedia REST API |
| `read_file` | Read a local text file (sandboxed to the project folder) |
| `search_docs` | RAG — semantic search over files in `knowledge/` |

---

## The files

| File | What it is |
|------|------------|
| `agent.py` | Model config + the 9 tools + the reflection agent (library + CLI) |
| `app.py` | Claude-style Gradio web UI on top of `agent.py` |
| `research_agent.py` | Research pipeline — produces a cited, structured report |
| `knowledge/` | Documents the `search_docs` (RAG) tool searches |

---

## Setup

**Requirements:** Python 3.11+

```bash
# 1. clone
git clone https://github.com/Rohit11-OG/learn-agent.git
cd learn-agent

# 2. create a virtual environment
python -m venv venv

# 3. install dependencies
venv\Scripts\pip install -r requirements.txt
```

**4. Add your NVIDIA NIM API key.** Create a file named `.env` in the project root:

```
NVIDIA_API_KEY=nvapi-your-key-here
```

Get a free key (no credit card) at [build.nvidia.com](https://build.nvidia.com)
— sign up for the NVIDIA Developer Program and generate an API key.

> The `.env` file is git-ignored — your key never gets committed.

---

## Usage

**Web UI** (chat in the browser — Claude-style design):

```bash
venv\Scripts\python app.py
```

Then open the local URL it prints (http://127.0.0.1:7860).

**Research pipeline** (cited report in the terminal):

```bash
venv\Scripts\python research_agent.py
```

**Quick CLI** (terminal chat, no UI):

```bash
venv\Scripts\python agent.py
```

---

## How it works

The reflection agent is a **graph** in LangGraph:

```
START → agent ──(needs a tool?)──► tools ──► agent
              └──(done?)──► reflect ──(weak?)──► agent
                                  └──(good?)──► END
```

- **State** — a shared dict holding the conversation
- **Nodes** — functions: call the model, run a tool, review the answer
- **Edges** — fixed or conditional arrows that route between nodes

The research pipeline runs a different graph:
`plan → research → synthesize → review → (revise | done)`.

---

## Model

Configured in `agent.py` — change one line to swap:

```python
MODEL = "nvidia/nemotron-3-super-120b-a12b"  # NVIDIA NIM — Nemotron Super
```

---

## Tech stack

- **LangGraph** — agent orchestration (the graph and loop)
- **LangChain** — model wrappers, tools, messages
- **NVIDIA NIM** — LLM inference (Nemotron Super)
- **FastEmbed** — local, free embeddings for RAG
- **Gradio** — the web UI
- **BeautifulSoup / requests** — web fetching and crawling

---

## Project structure

```
learn-agent/
├── agent.py            # model + tools + reflection agent
├── app.py              # Claude-style Gradio web UI
├── research_agent.py   # research pipeline (cited reports)
├── knowledge/          # docs for the RAG tool
├── requirements.txt
└── .env                # your API key (git-ignored)
```

---

## License

Personal learning project — free to use and learn from.
