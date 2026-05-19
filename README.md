# learn-agent

A hands-on project for learning how **AI agents** work, built step by step with
[LangGraph](https://www.langchain.com/langgraph) and the free [Groq](https://groq.com) API.

It is a **research assistant** agent: give it a question, and it decides which
tools to use (web search, Wikipedia, a calculator, a web crawler, and more),
gathers information, and writes an answer.

The repo grows in numbered steps — each file is a slightly more advanced agent,
so the code reads like a tutorial.

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
- **Single-agent** ReAct loop and **multi-agent** architectures
- **Reflection** — the agent reviews and revises its own answers
- **RAG** — searches a local knowledge base by meaning
- **Streaming** — answers print token-by-token
- **Memory** within a conversation (per-thread)
- Error handling, retry on bad tool calls, and loop guards
- 100% free stack — no paid API, no credit card

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

## Agent variants

Each file is runnable on its own and shows a different architecture.

| File | Architecture |
|------|--------------|
| `step9_agent.py` | Single ReAct agent — the core loop |
| `step10_multiagent.py` | Linear multi-agent — planner → researcher → writer |
| `step11_supervisor.py` | Supervisor — a boss LLM dynamically routes to workers |
| `step12_parallel.py` | Parallel — subtasks fan out and run at the same time |
| `step13_streaming.py` | Streaming — token-by-token output |
| `step14_reflect.py` | **Reflection agent** — 7 tools + self-review (most complete) |
| `step4_tool.py` | Shared tool definitions used by all of the above |

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

**4. Add your Groq API key.** Create a file named `.env` in the project root:

```
GROQ_API_KEY=your-key-here
```

Get a free key (no credit card) at [console.groq.com](https://console.groq.com).

> The `.env` file is git-ignored — your key never gets committed.

---

## Usage

Run any agent variant. The most complete one:

```bash
venv\Scripts\python step14_reflect.py
```

Then ask questions, for example:

```
You: What is the latest version of Python?
You: What is 17% of 4830?
You: Crawl https://example.com and summarize it
```

Type `quit` to exit.

---

## How it works

The agent is built as a **graph** in LangGraph:

```
START → agent ──(needs a tool?)──► tools ──► agent
              └──(done?)──► reflect ──(weak?)──► agent
                                  └──(good?)──► END
```

- **State** — a shared dict holding the conversation
- **Nodes** — functions: call the model, run a tool, review the answer
- **Edges** — fixed or conditional arrows that route between nodes

---

## Tech stack

- **LangGraph** — agent orchestration (the graph and loop)
- **LangChain** — model wrappers, tools, messages
- **Groq** — free, fast LLM inference (`openai/gpt-oss-20b`)
- **FastEmbed** — local, free embeddings for RAG
- **BeautifulSoup / requests** — web fetching and crawling

---

## Project structure

```
learn-agent/
├── step4_tool.py          # all 7 tool definitions
├── step9_agent.py         # single ReAct agent
├── step10_multiagent.py   # linear multi-agent
├── step11_supervisor.py   # supervisor routing
├── step12_parallel.py     # parallel fan-out
├── step13_streaming.py    # streaming output
├── step14_reflect.py      # reflection agent (main)
├── knowledge/             # docs for the RAG tool
├── requirements.txt
└── .env                   # your API key (git-ignored)
```

---

## License

Personal learning project — free to use and learn from.
