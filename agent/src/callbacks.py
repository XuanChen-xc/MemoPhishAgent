from collections import defaultdict
import logging
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler, UsageMetadataCallbackHandler

logger = logging.getLogger(__name__)


class ToolUsageTracker(BaseCallbackHandler):
    """Count how many times each tool is invoked."""

    def __init__(self):
        self.counts = defaultdict(int)

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs) -> None:
        tool_name = serialized.get("name")
        logging.debug(f"Trying to get tool name: {tool_name}")
        if tool_name:
            self.counts[tool_name] += 1


class LLMCounter(BaseCallbackHandler):
    """Count how many times the LLM is called (across all prompts)."""

    def __init__(self):
        self.count = 0

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs) -> None:
        self.count += 1


class LLMTimingCallback(BaseCallbackHandler):
    """Record wall-clock time of every chat-model call."""

    def __init__(self):
        self.call_times: list[float] = []
        self._starts: dict[str, float] = {}

    def on_chat_model_start(
        self,
        serialized: dict,
        messages: list,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._starts[str(run_id)] = time.perf_counter()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        start = self._starts.pop(str(run_id), None)
        if start is not None:
            self.call_times.append(time.perf_counter() - start)

    @property
    def total(self) -> float:
        return sum(self.call_times)

    @property
    def avg(self) -> float:
        return self.total / len(self.call_times) if self.call_times else 0.0


def get_default_callbacks() -> list[BaseCallbackHandler]:
    return [ToolUsageTracker(), LLMCounter()]


def get_token_usage_callbacks() -> UsageMetadataCallbackHandler:
    return UsageMetadataCallbackHandler()


def get_llm_timing_callback() -> LLMTimingCallback:
    return LLMTimingCallback()
