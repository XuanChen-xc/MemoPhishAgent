import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent

from agent_helpers import DeterministicNodes, NoImgNodes, ReactNodes
from callbacks import (
    get_default_callbacks,
    get_llm_timing_callback,
    get_token_usage_callbacks,
)
from memory import AgenticMemorySystem, MemoryNodes
from state import ReactURLState, URLState
from tools import AgentTools, _timing
from utils import get_llm

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("langchain_aws").setLevel(logging.WARNING)


def build_deterministic_agent(
    provider: str,
    model: Optional[str] = None,
    callbacks: Optional[list[BaseCallbackHandler]] = None,
) -> CompiledStateGraph:
    token_callback = get_token_usage_callbacks()
    if callbacks is None:
        callbacks = []
    callbacks.append(token_callback)
    llm = get_llm(provider=provider, model=model, callbacks=callbacks)

    tools = AgentTools(llm)
    deterministic_nodes = DeterministicNodes(tools, token_callback)

    graph = StateGraph(URLState)
    graph.add_node("deterministic_judge", deterministic_nodes.process)
    graph.add_edge(START, "deterministic_judge")
    graph.add_edge("deterministic_judge", END)
    return graph.compile()


def build_noimg_agent(
    provider: str,
    model: Optional[str] = None,
    callbacks: Optional[list[BaseCallbackHandler]] = None,
) -> CompiledStateGraph:
    llm = get_llm(provider=provider, model=model, callbacks=callbacks)
    tools = AgentTools(llm)

    token_callback = get_token_usage_callbacks()
    if callbacks is None:
        callbacks = []
    callbacks.append(token_callback)
    config = RunnableConfig(callbacks=callbacks, recursion_limit=80)
    react_agent = create_react_agent(model=llm, tools=[tools.crawl, tools.extract_links])

    nodes = NoImgNodes(
        react_agent=react_agent,
        config=config,
        token_callback=token_callback,
    )

    graph = StateGraph(URLState)
    graph.add_node("judge", nodes.react_judge_node)
    graph.add_edge(START, "judge")
    graph.add_edge("judge", END)
    return graph.compile()


def build_full_agent(
    provider: str,
    model: Optional[str] = None,
    callbacks: Optional[list[BaseCallbackHandler]] = None,
    use_memory: bool = True,
    memory_kwargs: Optional[dict[str, Any]] = None,
    args: Any = None,
) -> CompiledStateGraph:
    llm = get_llm(provider=provider, model=model, callbacks=callbacks)

    tools = AgentTools(llm)
    tool_list = [
        tools.crawl,
        tools.extract_targets,
        tools.check_img,
        tools.check_screenshot,
        tools.serpapi_search,
    ]

    token_callback = get_token_usage_callbacks()
    if callbacks is None:
        callbacks = []
    callbacks.append(token_callback)
    config = RunnableConfig(callbacks=callbacks, recursion_limit=80)

    if memory_kwargs is None:
        memory_kwargs = {}
    logging.info(f"Using memory: {use_memory}, Agent memory kwargs: {memory_kwargs}.")
    react_nodes = ReactNodes(
        llm=llm,
        tools=tool_list,
        token_callback=token_callback,
        config=config,
        args=args,
    )

    react_builder = StateGraph(ReactURLState, input=ReactURLState)
    if use_memory:
        logging.info("Building graph with memory.")
        agent_memory = AgenticMemorySystem(llm, **memory_kwargs)
        memory_nodes = MemoryNodes(tools, agent_memory)
        react_builder.add_node("prepare_memory", memory_nodes.prepare_memory)
        react_builder.add_node("store_memory", memory_nodes.store_memory)
        react_builder.add_node(react_nodes.call_model)
        react_builder.add_node("tools", ToolNode(tool_list))
        react_builder.add_edge("__start__", "prepare_memory")
        react_builder.add_edge("prepare_memory", "call_model")
        react_builder.add_conditional_edges(
            "call_model", react_nodes.route_model_output
        )
        react_builder.add_edge("tools", "call_model")
        react_builder.add_edge("store_memory", "__end__")
    else:
        logging.info("Building graph without memory.")
        react_builder.add_node("tools", ToolNode(tool_list))
        react_builder.add_node(react_nodes.call_model)
        react_builder.add_edge("__start__", "call_model")
        react_builder.add_conditional_edges(
            "call_model", react_nodes.route_model_output
        )
        react_builder.add_edge("tools", "call_model")

    react_agent = react_builder.compile(name="ReAct Agent")
    react_nodes.react_agent = react_agent

    graph = StateGraph(URLState)
    graph.add_node("judge", react_nodes.react_judge_node)
    graph.add_edge(START, "judge")
    graph.add_edge("judge", END)
    return graph.compile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["determine", "noimg_agent", "full_agent"])
    parser.add_argument(
        "--provider",
        choices=["openai", "bedrock"],
        default=os.environ.get("MEMOPHISH_PROVIDER", "openai"),
        help="LLM provider to use for the run.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MEMOPHISH_MODEL"),
        help="Optional model override. If omitted, provider-specific defaults are used.",
    )
    parser.add_argument("--input", default="test.txt")
    parser.add_argument("--output", default="data.json")
    parser.add_argument(
        "--use-ai-overview",
        default=True,
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        help="whether to use google ai overview in serpAPI (default: True)",
    )
    parser.add_argument(
        "--use-memory",
        default=True,
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        help="whether to enable the memory-augmented reasoning (default: True)",
    )
    parser.add_argument("-k", default=5, type=int, help="maximum similar memory returned")
    parser.add_argument(
        "--threshold",
        default=0.60,
        type=float,
        help="similarity threshold for retrieving memories",
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        urls = [u.strip().strip('",') for u in f if u.strip()]

    callbacks = get_default_callbacks()
    tracker = callbacks[0]
    counter = callbacks[1]
    llm_timer = get_llm_timing_callback()
    callbacks.append(llm_timer)

    logging.info("Running MemoPhishAgent with provider=%s model=%s", args.provider, args.model or "default")

    if args.agent == "determine":
        agent = build_deterministic_agent(
            provider=args.provider, model=args.model, callbacks=callbacks
        )
    elif args.agent == "noimg_agent":
        agent = build_noimg_agent(
            provider=args.provider, model=args.model, callbacks=callbacks
        )
    elif args.agent == "full_agent":
        agent = build_full_agent(
            provider=args.provider,
            model=args.model,
            callbacks=callbacks,
            use_memory=args.use_memory,
            memory_kwargs={"k": args.k, "threshold": args.threshold},
            args=args,
        )
    else:
        raise NotImplementedError(f"{args.agent} is not supported.")

    start_time = time.time()
    output = asyncio.run(agent.ainvoke({"urls": urls}))
    total_time = time.time() - start_time
    avg_time = total_time / len(urls) if urls else 0
    logging.info(
        "Total running time: %.2fs. Avg time: %.2fs per URL.", total_time, avg_time
    )

    result = output.get("result", [])
    json_result = output.get("json_result", [])
    failed_urls = output.get("failed_urls", [])

    logging.info("Tool usage: %s", dict(tracker.counts))
    logging.info("LLM calls: %d", counter.count)

    num_urls = max(len(urls), 1)
    llm_total = llm_timer.total
    crawl_total = sum(_timing["crawl_content_network"])
    ss_crawl = sum(_timing["check_screenshot_crawl"])
    ss_llm = sum(_timing["check_screenshot_llm"])
    extract_llm = sum(_timing["extract_targets_llm"])
    tool_crawl_total = crawl_total + ss_crawl
    tool_llm_total = ss_llm + extract_llm
    agent_llm_total = max(llm_total - tool_llm_total, 0.0)

    logging.info("=" * 70)
    logging.info("Component latency breakdown (cumulative across all URLs)")
    logging.info(
        "  crawl_content  - network (Crawl4AI): %.2fs  (%d calls, avg %.2fs)",
        crawl_total,
        len(_timing["crawl_content_network"]),
        crawl_total / max(len(_timing["crawl_content_network"]), 1),
    )
    logging.info(
        "  check_screenshot - crawl:           %.2fs  (%d calls, avg %.2fs)",
        ss_crawl,
        len(_timing["check_screenshot_crawl"]),
        ss_crawl / max(len(_timing["check_screenshot_crawl"]), 1),
    )
    logging.info(
        "  check_screenshot - vision LLM:      %.2fs  (%d calls, avg %.2fs)",
        ss_llm,
        len(_timing["check_screenshot_llm"]),
        ss_llm / max(len(_timing["check_screenshot_llm"]), 1),
    )
    logging.info(
        "  extract_targets  - LLM:             %.2fs  (%d calls, avg %.2fs)",
        extract_llm,
        len(_timing["extract_targets_llm"]),
        extract_llm / max(len(_timing["extract_targets_llm"]), 1),
    )
    logging.info(
        "  Agent ReAct reasoning LLM:          %.2fs  (%d calls, avg %.2fs)",
        agent_llm_total,
        len(llm_timer.call_times),
        llm_timer.avg,
    )
    logging.info("  -- Subtotals --")
    logging.info(
        "  All Crawl4AI network I/O:           %.2fs  (%.1f%% of total wall time)",
        tool_crawl_total,
        100 * tool_crawl_total / max(total_time, 1e-9),
    )
    logging.info(
        "  All LLM API calls:                  %.2fs  (%.1f%% of total wall time)",
        llm_total,
        100 * llm_total / max(total_time, 1e-9),
    )
    logging.info(
        "  Per-URL avg - Crawl4AI I/O:         %.2fs",
        tool_crawl_total / num_urls,
    )
    logging.info(
        "  Per-URL avg - LLM API:              %.2fs",
        llm_total / num_urls,
    )
    logging.info("=" * 70)

    output_base = args.output.rsplit(".", 1)[0]
    with open(f"{output_base}_raw.json", "w") as f:
        json.dump(result, f, indent=2)

    if json_result:
        with open(args.output, "w") as f:
            json.dump(json_result, f, indent=2)
    if failed_urls:
        with open(f"{output_base}_failed_urls.txt", "w") as file:
            for line in failed_urls:
                file.write(line + "\n")
