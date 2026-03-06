import logging
import time
from typing import List, Optional, Iterator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from config import GROQ_API_KEYS, GROQ_MODEL, IDRIS_SYSTEM_PROMPT, GENERAL_CHAT_ADDENDUM
from app.services.vector_store  import VectorStoreService
from app.utils.time_info import get_time_information
from app.utils.retry import with_retry

logger = logging.getLogger("I.D.R.I.S")

GROQ_REQUEST_TIMEOUT = 60
ALL_APIS_FAILED_MESSAGE = (
    "I'm unable to process your rerquest at the movement. all API sevices are"
    "temoerarrily unavailable. please try again in a few minutes"
)

class ALLGroqApisFailedError(Exception):
    pass


# ------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------

def escape_curly_braces(text: str) -> str:
    """
    Escape { and } so LangChain prompt templates do not break.
    """
    if not text:
        return text
    return text.replace("{", "{{").replace("}", "}}")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """
    Detect if error is a rate limit error (429).
    """
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "tokens per day" in msg

def _log_timing(label: str, elapssed: float, extra: str = ""):
    msg = f"[TIMING] {label}: {elapssed:.3f}s"
    if extra:
        msg+= f" ({extra})"
    logger.info(msg)           


def _mask_api_key(key: str) -> str:
    """
    Mask API key for safe logging.
    Example: gsk_12345678...abcd
    """
    if not key or len(key) < 12:
        return "***masked***"
    return f"{key[:8]}...{key[-4:]}"


# ------------------------------------------------------
# GROQ SERVICE CLASS
# ------------------------------------------------------

class GroqService:
    """
    General chat Groq service:
    - Retrieves context from vector store
    - Calls Groq LLM
    - Supports multiple API keys with round-robin + fallback
    """

    shared_key_index = 0  # shared across all instances

    def __init__(self, vector_store_service: VectorStoreService):
        if not GROQ_API_KEYS:
            raise ValueError(
                "No Groq API keys configured. Set GROQ_API_KEY in .env"
            )

        self.vector_store_service = vector_store_service

        # Create one LLM client per API key
        self.llms = [
            ChatGroq(
                groq_api_key=key,
                model_name=GROQ_MODEL,
                temperature=0.9,
            )
            for key in GROQ_API_KEYS
        ]

        logger.info(
            f"Initialized GroqService with {len(GROQ_API_KEYS)} API key(s)"
        )

    # ------------------------------------------------------
    # INTERNAL INVOKE WITH FALLBACK
    # ------------------------------------------------------

    def _invoke_llm(
        self,
        prompt: ChatPromptTemplate,
        messages: list,
        question: str
    ) -> str:
        """
        Invoke Groq using round-robin key rotation + fallback.
        """

        n = len(self.llms)
        last_exc  =None
        keys_tried = []

        for i in range(n):
            keys_tried.append(i)
            masked_key = _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"trying API key #{i+1}/{n}: {masked_key}")
            def _invoke_with_key():
                chain = prompt | self.llms[i]
                return chain.invoke({"history": messages, "question": question })
            
            try:
                response = with_retry(
                    _invoke_with_key,\
                    max_retries=2,
                    initial_delay=0.5
                )
                if i>0:
                    logger.info(f"Fa;;back successful : API key #{i+1}/{n} succeeded: {masked_key}")
                return response.content
            
            except Exception as e:
                last_exc = e
                if  _is_rate_limit_error(e):
                    logger.warning(f"APUIkey #{i + 1}/{n} rate limited: {masked_key}")
                else:
                    logger.warning(f"API key #{i + 1}/{n} failed: {masked_key} - {str(e)[:100]}")
                if i<n-1:
                    logger.info(f"Falling back to next api key........")
                    continue
                break

        maked_all = ",".join ([_mask_api_key(GROQ_API_KEYS[j])for j in keys_tried])
        logger.error(f"All{n} API key(s) failed. tried: {masked_key}")
        raise ALLGroqApisFailedError(ALLGroqApisFailedError) from last_exc
    
    def _stream_llm(
            self,
            prompt: ChatPromptTemplate,
            messages: list,
            question: str,
    ) -> Iterator[str]:
        
        n = len(self.llms)
        last_exc = None

        for i in range(n):
            masked_key = _mask_api_key(GROQ_API_KEYS[i])
            logger.info(f"Streaming with api key #{i+1}/{n}: {masked_key}") 

            try:
                chain = prompt | self.llms[i]
                chunk_count = 0 
                first_chunk_time = None
                stream_start = time.perf_counter()

                for chunk in chain.stream({"history": messages, "question": question}):
                    content   =  ""
                    if hasattr(chunk, "content"):
                        content =  chunk.content or ""
                    elif isinstance(chunk, dict) and "content" in chunk:
                        content = chunk.get("content","")or ""

                    if isinstance(content, str) and content:
                        if first_chunk_time is None:
                            first_chunk_time = time.perf_counter() - stream_start
                            _log_timing("groq_first_token", first_chunk_time)
                        chunk_count += 1
                        yield content

                total_stream =time.perf_counter() - stream_start
                _log_timing("groq_stream_total", total_stream, f"chunks: {chunk_count}")

                if i > 0 and chunk_count > 0:
                    logger.info(f"Fallback successful: API key #{i+1}/{n} streamed :{masked_key}")
                return


            except Exception as e:
                last_exc = e
                if _is_rate_limit_error(e):
                    logger.warning(f"API key #{i+1}/{n} rate limited: {masked_key}")
                else:
                    logger.warning(f"API key #{i+1}/{n} failed : {masked_key} - {str(e)[:100]}")
                if i<n - 1 :
                    logger.info("failling back to next API key for stream....")
                    continue
                    break

            logger.error(f"ALL {n}API key(s) failed during stream.")
            raise ALLGroqApisFailedError(ALL_APIS_FAILED_MESSAGE) from last_exc
        
    def _build_prompt_and_messages(
                self,
                question: str,
                chat_history: Optional[List[tuple]] = None,
                extra_system_parts: Optional[List[str]] = None,
                mode_addendum: str = "",
                ) -> tuple:
            
            content = ""
            context_sources = []
            t0 = time.perf_counter()
            try:
    
                retriever = self.vector_store_service.get_retriever(k=10) 
                context_docs = retriever.invoke(question)

                if context_docs:
                    context = "\n".join([doc.page_content for doc in context_docs])
                    context_sources = [doc.metadata.get("source", "unknown") for doc in context_docs]
                    logger.info("[CONTEXT] Retreved %d chunks frrom source: %s", len(context_docs), context_sources)

                else:
                    logger.info("[CONTEXT] No relevent chunks found for query ")

            except Exception as retriever_err:
                logger.warning("vector store retrieval failed , using empty context: %s", retriever_err)

            finally:
                _log_timing("vector_db", time.perf_counter() - t0)

            time_info =  get_time_information()
            system_message  = IDRIS_SYSTEM_PROMPT

            system_message += f"\n\ncurent time and date: {time_info}"

            if context:
                system_message += f"\n\nRelevent context from your learning data and past conversation:\n{escape_curly_braces(context)}"

            if extra_system_parts :
                system_message += "\n\n" + "\n\n".join(extra_system_parts)

            if mode_addendum:
                system_message += f"\n\n{mode_addendum}"

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_message),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ])

            messages =[]
            if chat_history:
                for human_msg, ai_msg in chat_history:
                    messages.append(HumanMessage(content=human_msg))
                    messages.append(AIMessage(content=ai_msg))

            logger.info("[prompt] System lenght: %d chars | History pairs:%d | Question: %.100s",
                        len(system_message), len(chat_history) if chat_history else 0 , question)
            return prompt, messages
        
    def get_response(
            self,
            question: str,
            chat_history: Optional[List[tuple]] = None        
        ) -> str:
            
            try:
                prompt, messages = self._build_prompt_and_messages(
                    question, chat_history, mode_addendum = GENERAL_CHAT_ADDENDUM,
                )
                t0 = time.perf_counter()
                result = self._invoke_llm(prompt, messages, question)
                _log_timing("groq_api", time.perf_counter() - t0)
                logger.info("[response] General chat | Length:%d chars | preview: %.120",len(result), result)
                return result
            except ALLGroqApisFailedError:
                raise
            except Exception as e:
                raise Exception (f"Error getting response from groq : {str(e)}") from e
            
    def stream_response(
                self,
                question: str,
                chat_history: Optional[List[tuple]] = None,
        ) -> Iterator[str]:
            try :
                prompt, messages = self._build_prompt_and_messages(
                    question, chat_history, mode_addendum = GENERAL_CHAT_ADDENDUM,
                )
                yield from self._stream_llm(prompt, messages, question)
            except ALLGroqApisFailedError:
                raise
            except Exception as e:
                raise Exception (f"Error getting response from groq : {str(e)}") from e
