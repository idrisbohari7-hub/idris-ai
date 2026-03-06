import logging
import time
from typing import TypeVar, Callable

logger = logging.getLogger("I.D.R.I.S.")
T = TypeVar("T")

def with_retry(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,) -> T:
    
    last_exception = None
    delay = initial_delay

    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e

            if attempt == max_retries - 1:
                raise

            logger.warning(
                "attempt %s/%s failed (%s). retrying in %.1fs: %s",
                attempt + 1,
                max_retries,
                e,
                delay,
                func.__name__
            )

            time.sleep(delay)
            delay *= 2
            
    raise last_exception