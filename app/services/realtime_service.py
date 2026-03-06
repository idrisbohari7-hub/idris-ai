from typing import List, Optional, Iterator, Tuple, Any 
from tavily import TavilyClient
import logging
import os
import time
from app.services.groq_service import GroqService,  ALLGroqApisFailedError ,escape_curly_braces 
from app.services.vector_store import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry
from config import REALTIME_CHAT_ADDENDUM, GROQ_API_KEYS, GROQ_MODEL


logger = logging.getLogger("I.D.R.I.S")

_QUERY_EXTRACTION_PROMPT = (
    "You are a search query optimizer. Given the user's message and recent conversation, "
    "produce a single short, focused web search query (max 12 words) that will find the "
    "information the user needs. Resolve any references (like 'that website', 'him', 'it') "
    "using the conversation history. Output ONLY the search query, nothing else."
)


class RealtimeGroqService(GroqService):
    def __init__(self, vector_store_service: VectorStoreService):
        super().__init__(vector_store_service)

        tavily_api_key = os.getenv("TAVILY_API_KEY", "")
        if tavily_api_key:
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
            logger.info("✅ Tavily client initialized successfully.")
        else:
            self.tavily_client = None
            logger.warning("❌ TAVILY_API_KEY not found. Tavily disabled.")

    def _extract_search_query(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None
        ) -> str:

    # If fast LLM is not available, just use the question
        if not self.llms:
            return question

        try:
            t0 = time.perf_counter()

            history_context = ""

            if chat_history:
                recent = chat_history[-3:]
                parts = []

                for h, a in recent:
                    parts.append(f"User: {h[:200]}")
                    parts.append(f"Assistant: {a[:200]}")

                history_context = "\n".join(parts)

            if history_context:
                full_prompt = (
                    f"{_QUERY_EXTRACTION_PROMPT}\n\n"
                    f"Recent conversation:\n{history_context}\n\n"
                    f"User's latest message: {question}\n\n"
                    f"Search query:"
                )
            else:
                full_prompt = (
                    f"{_QUERY_EXTRACTION_PROMPT}\n\n"
                    f"User's message: {question}\n\n"
                    f"Search query:"
                )

            extracted = self.llms[0].invoke(full_prompt).content.strip()

            if extracted and 3 <= len(extracted) <= 200:
                logger.info(
                    "[REALTIME] Query extraction: '%s' -> '%s' (%.3fs)",
                    question[:80],
                    extracted[:80],
                    time.perf_counter() - t0,
                )
                return extracted

            logger.warning(
                "[REALTIME] Query extraction returned unusable result, using raw question"
            )
            return question

        except Exception as e:
            logger.warning(
            "[REALTIME] Query extraction failed (%s), using raw question",e)
            return question

    def search_tavily(self, query: str, num_results: int = 7) -> str:
        if not self.tavily_client:
            return ("", None)

        try:
            t0 = time.perf_counter()

            response = with_retry(
                lambda: self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=num_results,
                include_answer=True,
                include_raw_content=False,
            ),
            max_retries=3,
            initial_delay=1.0,
            )

            results = response.get("results", [])
            ai_answer = response.get("answer", "")

            if not results and not ai_answer:
                logger.warning("No tavily search result for query:%s", query)
                return ("",None)
            
            payload: Optional[dict] = {
                "query": query,
                "answer": ai_answer,
                "results": [
                    {
                        "title": r.get("title", "No title"),
                        "content": (r.get("content") or "")[:500],
                        "url": r.get("url", ""),
                        "score": round(float(r.get("score", 0)), 2),
                    }
                    for r in results[:num_results]
                ],
                           
           }
            parts = [f" === WEB SEARCH RESULTS FOR: {query} === \n"]
            if ai_answer:
                parts.append(f"AI-SYNTHESIZED ANSWER (use this as your primary source):\n{ai_answer}\n")
            if results:
                parts.append("INDIVIDUAL SOURCES:")
            for i, result in enumerate(results[:num_results], 1):
                title = result.get("title", "No title")
                content = result.get("content", "")
                url = result.get("url", "")
                score = result.get("score", 0)
                parts.append(f"\n[Source {i}] (relevance: {score :.2f})")
                parts.append(f"Title: {title}")
                if content:
                    parts.append(f"Content: {content}")
                if url:
                    parts.append(f"URL: {url}")
            parts.append("\n === END SEARCH RESULTS === ")
            formatted = "\n".join(parts)

            logger.info(
            "[TAVILY] %d results, AI answer: %s, formatted: %d chars (%.3fs)",  
            len(results), "yes" if ai_answer else "no",
            len(formatted), time.perf_counter() - t0,
            )
            return (formatted, payload)
        except Exception as e:
            logger.error("erorr performing tavily search:%s, e")
            return ("", None)
    

    def get_response(self, question: str, chat_history=None) -> str:
        try:
            search_query = self._extract_search_query(question, chat_history)
            logger.info(f"🔍 Tavily searching: {question}")
            formatted_results = self.search_tavily(question)

            if formatted_results:
                logger.info("[REALTIME] Tavily returned results (length: %d chars)",len(formatted_results))
            else:
                logger.warning("[REALTIME] Tavily returned no results for: %s", search_query)

            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None
            prompt, messages = self._build_prompt_and_messages(
            question,chat_history,
            extra_system_parts=extra_parts,
            mode_addendum=REALTIME_CHAT_ADDENDUM,
            )

            t0 = time.perf_counter()
            response_content = self._invoke_llm(prompt, messages, question)

            logger.info(
            "[TIMING] groq_api: %.3fs",
            time.perf_counter() - t0
            )

            logger.info(
            "[RESPONSE] Realtime chat | Length: %d chars | Preview: %.120s",
            len(response_content),
            response_content,
            )

            return response_content


        except ALLGroqApisFailedError :
            raise

        except Exception as e:
            logger.error(
            "Error in realtime get_response: %s", e, exc_info=True)
            raise

    def stream_response(
    self,
    question: str,
    chat_history: Optional[List[tuple]] = None
    ) -> Iterator[Any]:

        try:
            search_query = self._extract_search_query(question, chat_history)
            logger.info("[REALTIME] Searching Tavily for: %s", search_query)

            formatted_results, payload = self.search_tavily(search_query, num_results=7)

            if formatted_results:
                logger.info(
                "[REALTIME] Tavily returned results (length: %d chars)",
                len(formatted_results)
                )
            else:
                logger.warning(
                "[REALTIME] Tavily returned no results for: %s",
                search_query
                )

            if payload:
                yield {"_search_results": payload}

            extra_parts = [escape_curly_braces(formatted_results)] if formatted_results else None

            prompt, messages = self._build_prompt_and_messages(
            question,
            chat_history,
            extra_system_parts=extra_parts,
            mode_addendum=REALTIME_CHAT_ADDENDUM,
            )

            yield from self._stream_llm(prompt, messages, question)

            logger.info("[REALTIME] Stream completed for: %s", search_query)

        except ALLGroqApisFailedError:
            raise

        except Exception as e:
            logger.error("Error in realtime stream_response: %s", e, exc_info=True)
            raise