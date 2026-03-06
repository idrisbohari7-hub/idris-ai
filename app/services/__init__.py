"""
SERVICES PACKAGE
=================

Business logic lives here. The API layer (app.main) calls these services;
they do not handle HTTP, only chat flow, LLM calls, and data.

MODULES:
chat_service
groq_service
realtime_service - Realtime chat: Tavily search first, then same as groq (inherits GroqService).
- Load learning_data + chats_data, chunk, embed, FAISS index; provide retriever for context.

- Sessions (get/create, load from disk), message list, format history for LLM, save to disk.
- General chat: retrieve context from vector store, build prompt, call Groq LLM.

vector store
"""