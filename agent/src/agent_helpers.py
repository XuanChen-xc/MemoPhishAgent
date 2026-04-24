import asyncio
import json
import logging
import time
from typing import Any, Literal, cast

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from prompts import SYSTEM_NO_IMG, SYSTEM_REACT, SYSTEM_REACT_MEM
from state import ReactURLState, URLState
from tools import AgentTools
from utils import ai_overview_preprocess, extract_and_fix, is_rate_limit_error

logger = logging.getLogger(__name__)


class ReactNodes:
    def __init__(
        self,
        llm: Any,
        tools: list[BaseTool],
        token_callback: Any,
        config: RunnableConfig,
        args: Any,
    ):
        self.llm = llm
        self.my_tools = tools
        self.token_callback = token_callback
        self.config = config
        self.args = args
        self.SYSTEM_REACT = SYSTEM_REACT
        self.SYSTEM_REACT_MEM = SYSTEM_REACT_MEM
        self.react_agent = None
        self._sem = asyncio.Semaphore(5)

    async def call_model(self, state: ReactURLState) -> dict[str, list[AIMessage]]:
        model = self.llm.bind_tools(self.my_tools)
        system_message = self.SYSTEM_REACT
        logger.info(f"Find related memory: {state.memory_snippet}")

        if state.memory_majority:
            logging.info("Majority of the memories are malicious")
            verdict = {
                "url": state.url,
                "malicious": True,
                "confidence": 5,
                "reason": "Reused majority-vote from past similar URLs (>50% malicious).",
            }
            answer = AIMessage(content=json.dumps({"verdicts": [verdict]}))
            return {"messages": [answer]}

        if state.memory_snippet:
            system_message = (
                f"{state.memory_snippet}\n"
                "First leverage those past-case summaries ('memory'), only invoke tools if you need more evidence.\n"
                + self.SYSTEM_REACT_MEM
            )

        if not state.messages:
            human = {
                "role": "user",
                "content": f"Judge if this URL {state.url} is malicious or phishing site.",
            }
            prompt = [{"role": "system", "content": system_message}, human]
        else:
            prompt = [{"role": "system", "content": system_message}, *state.messages]

        response = cast(AIMessage, await model.ainvoke(prompt))
        if response.tool_calls and response.tool_calls[0]["name"] == "crawl_content":
            response.tool_calls[0]["args"]["screenshot"] = False

        if state.is_last_step and response.tool_calls:
            return {
                "messages": [
                    AIMessage(
                        id=response.id,
                        content="Sorry, I could not find an answer to your question in the specified number of steps.",
                    )
                ]
            }
        return {"messages": [response]}

    def route_model_output(
        self, state: ReactURLState
    ) -> Literal["store_memory", "tools", "__end__"]:
        last_msg = state.messages[-1]
        if not isinstance(last_msg, AIMessage):
            raise ValueError(f"Expected AIMessage, got {type(last_msg).__name__}")

        if last_msg.tool_calls:
            return "tools"
        return "store_memory" if state.use_memory else "__end__"

    async def _process_one_url(self, url: str, idx: int) -> dict[str, Any]:
        async with self._sem:
            if self.args.use_ai_overview:
                ai_overview_res = ai_overview_preprocess(url, self.llm)
                if ai_overview_res and ai_overview_res["malicious"]:
                    ai_overview_res["memory_case"] = "google_ai_overview"
                    return {"type": "ai_overview", "data": ai_overview_res}

            url_config = {
                **self.config,
                "configurable": {
                    **self.config.get("configurable", {}),
                    "thread_id": f"{self.config.get('configurable', {}).get('thread_id', 'default')}-{idx}",
                },
            }
            agent_input = {
                "messages": [
                    HumanMessage(
                        content=f"Judge if this URL: {url} is malicious or a phishing website."
                    )
                ],
                "url": url,
                "use_memory": self.args.use_memory,
                "tool_sequence": [],
                "keywords": [],
            }
            async for step in self.react_agent.astream(
                agent_input, config=url_config, stream_mode="values"
            ):
                last_msg = step["messages"][-1]
                last_msg.pretty_print()

            return {
                "type": "full",
                "url": url,
                "final_msg": last_msg.content,
                "memory_case": step.get("memory_case", " "),
            }

    async def react_judge_node(self, state: URLState) -> dict[str, Any]:
        if self.react_agent is None:
            raise RuntimeError("react_agent not set on ReactNodes")

        raw_results = await asyncio.gather(
            *[self._process_one_url(url, i) for i, url in enumerate(state["urls"])],
            return_exceptions=True,
        )

        verdicts, jsons, failed_urls = [], [], []
        for url, raw in zip(state["urls"], raw_results):
            if isinstance(raw, Exception):
                if is_rate_limit_error(raw):
                    logging.warning(f"Rate limit for {url}: {raw}")
                else:
                    logging.warning(f"Unexpected error for {url}: {raw}")
                failed_urls.append(url)
                continue

            if raw["type"] == "ai_overview":
                jsons.append(raw["data"])
                continue

            logging.info("===" * 50)
            verdicts.append({"url": url, "reason": raw["final_msg"]})
            final_json = extract_and_fix(raw["final_msg"])
            try:
                for verdict in final_json[0]["verdicts"]:
                    verdict["memory_case"] = raw["memory_case"]
                    jsons.append(verdict)
            except Exception as exc:
                logging.info(f"Error {exc}")
                failed_urls.append(url)

        with open(self.args.output, "w") as f:
            json.dump(jsons, f, indent=2)
        if failed_urls:
            with open(f"{self.args.output.rsplit('.', 1)[0]}_failed_urls.txt", "w") as f:
                for line in failed_urls:
                    f.write(line + "\n")

        logging.info(f"Token usage: {self.token_callback.usage_metadata}")
        return {"result": verdicts, "json_result": jsons, "failed_urls": failed_urls}


class DeterministicNodes:
    """Deterministic multi-tool pipeline for classifying URLs."""

    def __init__(self, tools: AgentTools, token_callback: UsageMetadataCallbackHandler):
        self.tools = tools
        self.token_callback = token_callback

    async def process(self, state: URLState) -> dict[str, list[str]]:
        page_malicious = []
        screenshot_malicious = []
        failed_urls = []
        for url in state["urls"]:
            try:
                page = await self.tools.crawl.arun({"url": url, "screenshot": True})
                content_judge = await self.tools.judge_crawled_page.arun(
                    {"url": url, "text": page["text"]}
                )
                if content_judge["malicious"]:
                    page_malicious.append(url)
                    logging.info("Page content is malicious.")
                    continue

                ss_judge = await self.tools.check_screenshot.arun(url)
                if ss_judge["malicious"]:
                    screenshot_malicious.append(url)
                    logging.info("Page screenshot is malicious.")
                    continue

                targets = await self.tools.extract_targets.arun(
                    {"url": url, "text": page["text"]}
                )

                async def _check_img(img_url: str) -> dict[str, Any]:
                    img_desp = await self.tools.check_img.arun(img_url)
                    return await self.tools.judge_img.arun(
                        {
                            "image_url": img_desp["image_url"],
                            "description": img_desp["description"],
                        }
                    )

                img_results = await asyncio.gather(
                    *[_check_img(candidate) for candidate in targets["to_check_images"]],
                    return_exceptions=True,
                )
                for result in img_results:
                    if isinstance(result, Exception):
                        logging.info(f"Error when call image tools: {result}, continue")
                        continue
                    if result.get("malicious"):
                        screenshot_malicious.append(url)
                        logging.info("Inside image is malicious.")
                        break

                async def _check_sub_url(inside_url: str) -> dict[str, Any]:
                    page = await self.tools.crawl.arun(inside_url)
                    return await self.tools.judge_crawled_page.arun(
                        {"url": inside_url, "text": page["text"]}
                    )

                sub_results = await asyncio.gather(
                    *[_check_sub_url(candidate) for candidate in targets["to_crawl"]],
                    return_exceptions=True,
                )
                for result in sub_results:
                    if isinstance(result, Exception):
                        logging.info(f"Error when checking sub-URL: {result}, continue")
                        continue
                    if result.get("malicious"):
                        page_malicious.append(url)
                        logging.info("Inside URL is malicious.")
                        break

            except Exception as exc:
                if is_rate_limit_error(exc):
                    logging.warning("Throttling detected: sleeping 5 seconds...")
                    await asyncio.sleep(5)
                else:
                    logging.warning(f"Error when judging {url}: {exc}, continue")
                failed_urls.append(url)

        final_malicious = page_malicious + screenshot_malicious
        logging.info(f"Token usage: {self.token_callback.usage_metadata}")

        return {
            "page_malicious": page_malicious,
            "screenshot_malicious": screenshot_malicious,
            "final_malicious": final_malicious,
            "failed_urls": failed_urls,
        }


class NoImgNodes:
    """Reactive URL classification node that operates without image tools."""

    def __init__(
        self,
        react_agent: CompiledStateGraph,
        config: RunnableConfig,
        token_callback: UsageMetadataCallbackHandler,
    ):
        self.react_agent = react_agent
        self.config = config
        self.token_callback = token_callback

    async def react_judge_node(self, state: URLState) -> dict[str, Any]:
        verdicts, jsons, failed = [], [], []
        start = time.time()

        for i, url in enumerate(state["urls"]):
            agent_input = {
                "messages": [
                    SYSTEM_NO_IMG,
                    HumanMessage(
                        content=f"Judge if this URL: {url} is malicious or a phishing website."
                    ),
                ]
            }

            try:
                async for step in self.react_agent.astream(
                    agent_input, config=self.config, stream_mode="values"
                ):
                    last_msg = step["messages"][-1]

                final_msg = last_msg.content
                logging.info(final_msg)
                logging.info("===" * 50)

                verdicts.append({"url": url, "reason": final_msg})
                final_json = extract_and_fix(final_msg)
                for verdict in final_json[0]["verdicts"]:
                    jsons.append(verdict)

                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start
                    avg = elapsed / (i + 1)
                    logging.info(f"Total elapsed: {elapsed:.2f}s")
                    logging.info(f"Avg time/URL: {avg:.2f}s")

            except Exception as exc:
                if is_rate_limit_error(exc):
                    logging.warning("Throttling detected: sleeping 3 seconds...")
                    failed.append(url)
                    await asyncio.sleep(3)
                    continue
                logging.warning(f"Error when judging {url}: {exc}, continue")
                failed.append(url)

        return {"result": verdicts, "json_result": jsons, "failed_urls": failed}
