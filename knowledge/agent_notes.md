# Agent Project Notes

## Project
The Learn Agent project is a research assistant built with LangGraph and the
NVIDIA NIM API. It uses the model `nvidia/nemotron-3-super-120b-a12b`
(Nemotron Super) — NVIDIA's agentic-tuned flagship. The owner built it to
learn how AI agents work.

## Tools available
The agent has 9 tools: web_search, fetch_url (with auto JS-render fallback),
js_fetch (force JS render), deep_crawl (static multi-page), js_crawl
(JavaScript multi-page), calculator, wikipedia_lookup, read_file, and
search_docs (RAG over this knowledge folder).

## Architecture decisions
- Model provider is NVIDIA NIM because it is free, no credit card, and the
  Nemotron Super model is built specifically for agentic tool calling.
- Web scraping uses a shared headless Chromium (Playwright) plus trafilatura
  for clean article extraction — drops nav, ads, footers.
- Memory uses InMemorySaver per send (fresh thread_id each turn) — avoids
  cross-question memory contamination. No SQLite, no database setup.
- Embeddings use FastEmbed because it is local, free, and does not need
  PyTorch.

## Owner preferences
The owner prefers free tools, no paid APIs, and minimal setup.
The owner is learning step by step and wants simple explanations.
