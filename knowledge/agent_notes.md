# Agent Project Notes

## Project
The Learn Agent project is a research assistant built with LangGraph and the Groq API.
It uses the model openai/gpt-oss-20b. The owner built it to learn how AI agents work.

## Tools available
The agent has these tools: web_search, fetch_url, deep_crawl, calculator,
wikipedia_lookup, read_file, and search_docs.

## Architecture decisions
- Model provider is Groq because it is free and needs no credit card.
- Memory uses InMemorySaver (not SQLite) because the owner did not want database setup.
- Embeddings use FastEmbed because it is local, free, and does not need PyTorch.

## Owner preferences
The owner prefers free tools, no paid APIs, and minimal setup.
The owner is learning step by step and wants simple explanations.
