import logging
import uuid
from collections import Counter
from typing import Any, Optional

from langgraph.store.memory import InMemoryStore

from state import ReactURLState
from tools import AgentTools
from utils import extract_and_fix, get_memory_embeddings, get_provider_from_llm

logger = logging.getLogger(__name__)


class AgenticMemorySystem:
    def __init__(
        self,
        llm: Any,
        evo_threshold: int = 100,
        k: int = 5,
        threshold: float = 0.60,
    ):
        embeddings, dims = get_memory_embeddings(get_provider_from_llm(llm))
        self.memory_store = InMemoryStore(index={"embed": embeddings, "dims": dims})
        self.k = k
        self.threshold = threshold
        self.llm_controller = llm
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    async def summarize_keywords(self, text: str, screenshot_b64: str) -> list[str]:
        system = {
            "role": "system",
            "content": (
                """Given the following text content and visual artifacts of a webpage, generate up to 10 keywords that best capture its content, using comma as the separator.
                Format your response as a JSON object with a 'keywords' field containing the selected text.
                Example response format:{"keywords": "keyword1, keyword2, keyword3"}"""
            ),
        }
        user = {
            "role": "user",
            "content": (
                f"Page text:\n{text}\n\n"
                f"Screenshot base64 (for context, do NOT output raw data):\n"
                f"{screenshot_b64}…[truncated]"
            ),
        }
        resp = await self.llm_controller.ainvoke([system, user])
        return [kw.strip() for kw in resp.content.split(",") if kw.strip()]

    async def search_by_keywords(self, keywords: list[str]) -> Optional[str]:
        query = " ".join(keywords)
        namespace = ("agent_memory",)
        hits = self.memory_store.search(namespace, query=query, limit=self.k)

        case_summaries = []
        for hit in hits:
            if hit.score < self.threshold:
                continue
            mem = hit.value
            url = mem["url"]
            kws = mem["keywords"]
            verdict = mem["verdict"]
            trace = mem["trace"]

            label = "Malicious" if verdict.get("malicious") else "Benign"
            conf = verdict.get("confidence", 0)
            reason = verdict.get("reason", "")

            case_summaries.append(
                "• **URL:** "
                + url
                + "\n"
                + f"  - **Keywords:** {kws}\n"
                + f"  - **Verdict:** {label} (confidence {conf}/5)\n"
                + f"  - **Reason:** {reason}\n"
                + f"  - **Tool calling trace:** {trace}\n"
            )

        if case_summaries:
            return (
                "Previously, for similar URLs, we had these memories:\n\n"
                + "\n\n".join(case_summaries)
            )
        return None

    async def search_by_keywords_w_majority(
        self, keywords: list[str]
    ) -> tuple[Optional[str], Optional[bool]]:
        query = " ".join(keywords)
        namespace = ("agent_memory",)
        hits = self.memory_store.search(namespace, query=query, limit=self.k)

        verdicts = []
        case_summaries = []
        for hit in hits:
            if hit.score < self.threshold:
                continue
            mem = hit.value
            url = mem["url"]
            kws = mem["keywords"]
            verdict = mem["verdict"]
            trace = mem["trace"]

            is_mal = bool(verdict.get("malicious", False))
            verdicts.append(is_mal)

            label = "Malicious" if is_mal else "Benign"
            conf = verdict.get("confidence", 0)
            reason = verdict.get("reason", "")

            case_summaries.append(
                f"• **URL:** {url}\n"
                f"  - **Keywords:** {kws}\n"
                f"  - **Verdict:** {label} (confidence {conf}/5)\n"
                f"  - **Reason:** {reason}\n"
                f"  - **Tool calling trace:** {trace}\n"
            )

        if not case_summaries:
            return None, None

        snippet = (
            "Previously, for similar URLs, we had these memories:\n\n"
            + "\n".join(case_summaries)
        )

        counts = Counter(verdicts)
        majority = False
        if counts[True] > counts[False] and len(case_summaries) >= self.k:
            majority = True

        return snippet, majority

    async def store_memory(
        self,
        keywords: list[str],
        trace: list[str],
        verdict: dict[str, Any],
        url: str,
    ) -> None:
        content = {
            "url": url,
            "keywords": [", ".join(keywords)],
            "verdict": verdict,
            "trace": trace,
        }
        namespace = ("agent_memory",)
        memory_id = uuid.uuid4().hex
        self.memory_store.put(namespace=namespace, key=memory_id, value=content)


class MemoryNodes:
    def __init__(self, tools: AgentTools, agent_memory: AgenticMemorySystem):
        self.tools = tools
        self.agent_memory = agent_memory

    async def prepare_memory(self, state: ReactURLState) -> dict[str, Any]:
        page = await self.tools.crawl.arun({"url": state.url, "screenshot": True})

        keywords = await self.agent_memory.summarize_keywords(
            page["text"], page["screenshot"]
        )
        logging.info(f"keywords for current url: {keywords} \n {state.url}")

        retrieved_mem, majority = await self.agent_memory.search_by_keywords_w_majority(
            keywords
        )
        mem_case = "memory_reuse" if retrieved_mem else "full_reasoning"

        return {
            "memory_snippet": retrieved_mem,
            "keywords": keywords,
            "memory_majority": majority,
            "memory_case": mem_case,
        }

    async def store_memory(self, state: ReactURLState) -> dict[str, Any]:
        final_msg = state.messages[-1].content
        final_json = extract_and_fix(final_msg)
        try:
            for verdict in final_json[0]["verdicts"]:
                if verdict["confidence"] > 4:
                    logging.info(
                        f"Save memory {state.keywords}, {state.tool_sequence}, {verdict}"
                    )
                    await self.agent_memory.store_memory(
                        keywords=state.keywords,
                        trace=state.tool_sequence,
                        verdict=verdict,
                        url=state.url,
                    )
        except Exception as exc:
            logging.info(f"Error {exc}, model response: {final_msg}")

        return {}
