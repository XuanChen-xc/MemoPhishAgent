import base64
import io
import json
import logging
import os
import time
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from furl import furl
from langchain_community.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from PIL import Image
from pydantic import BaseModel, Field

from prompts import SYSTEM_EXTRACT, SYSTEM_JUDGE, SYSTEM_JUDGE_IMG, SYSTEM_SCREEN
from state import URLState
from utils import (
    build_image_message,
    find_all_link_urls,
    find_image_urls,
    get_provider_from_llm,
    get_serpapi_api_key,
)

logger = logging.getLogger(__name__)

MAX_CHARS = 5_000
MAX_IMG = 90_000
MAX_LINKS = 2
MAX_IMAGES = 2

_crawl_cache: dict[tuple[str, bool], dict[str, Any]] = {}
_timing: dict[str, list[float]] = defaultdict(list)

api_key = get_serpapi_api_key()
serpapi = SerpAPIWrapper(serpapi_api_key=api_key)
serpapi_tool = Tool.from_function(
    func=serpapi.run,
    coroutine=serpapi.arun,
    name="serpapi_search",
    description=(
        "Use this to look up facts on the web. "
        "Input: a search query (string). "
        "Output: the combined text of the top search results."
    ),
)


class CrawlContentInput(BaseModel):
    url: str = Field(..., description="The URL of the page to fetch.")
    screenshot: bool = Field(
        False, description="Whether to take a screenshot of the page."
    )


class CrawlContentTool(BaseTool):
    name: str = "crawl_content"
    description: str = (
        "Fetch the page at `url`, return up to the first 5000 chars of its text."
    )
    args_schema: type = CrawlContentInput

    async def _arun(self, url: str, screenshot: bool = False) -> Dict[str, Any]:
        if (url, screenshot) in _crawl_cache:
            logging.info(f"Crawl cache hit for {url}")
            return _crawl_cache[(url, screenshot)]
        if not screenshot and (url, True) in _crawl_cache:
            logging.info(f"Crawl cache hit (screenshot superset) for {url}")
            return _crawl_cache[(url, True)]

        prefixes = ("http://", "https://")
        if any(url.startswith(prefix) for prefix in prefixes):
            candidates = [url]
        else:
            candidates = [f"{prefix}{url}" for prefix in prefixes]

        async with AsyncWebCrawler() as crawler:
            js_code = [
                """
                    (() => {
                        const loadMoreButton = Array.from(document.querySelectorAll('button'))
                            .find(button => button.textContent.includes('Load More'));
                        if (loadMoreButton) loadMoreButton.click();
                    })();
                    """
            ]
            config = CrawlerRunConfig(
                cache_mode=None,
                js_code=js_code,
                verbose=False,
                screenshot=screenshot,
            )

            for candidate in candidates:
                try:
                    start = time.perf_counter()
                    result = await crawler.arun(candidate, config=config)
                    _timing["crawl_content_network"].append(time.perf_counter() - start)
                    snippet = result.markdown.raw_markdown[:MAX_CHARS]
                    if screenshot:
                        out: Dict[str, Any] = {
                            "url": candidate,
                            "text": snippet,
                            "screenshot": result.screenshot[:MAX_IMG],
                        }
                    else:
                        out = {"url": candidate, "text": snippet}
                    _crawl_cache[(url, screenshot)] = out
                    return out
                except Exception as exc:
                    logging.info(f"Error crawling {candidate}: {exc}")
                    continue

        if screenshot:
            fallback: Dict[str, Any] = {
                "url": url,
                "text": "No text.",
                "screenshot": "No image.",
            }
        else:
            fallback = {"url": url, "text": "No text."}
        _crawl_cache[(url, screenshot)] = fallback
        return fallback

    def _run(self, *args, **kwargs):
        raise NotImplementedError(
            "CrawlContentTool only supports async invocation via `_arun`."
        )


def make_extract_urls_no_images(chat: Any):
    @tool(parse_docstring=True)
    async def extract_urls_no_images(
        url: str,
        text: Optional[str] = "",
    ) -> Dict[str, List[str]]:
        """Given a url and its corresponding content, select which non-image links
        to be crawled next. Image URLs are ignored entirely.

        Args:
            url: The page URL.
            text: The truncated page text.

        Returns:
            A dict with a single key:
            * **to_crawl**: list of URLs (excluding any image URLs) to crawl next.
        """
        logging.info(f"extract_urls_no_images received page: {url}")

        if text == "":
            async with AsyncWebCrawler() as crawler:
                js_code = [
                    """
                    (() => {
                        const loadMoreButton = Array.from(document.querySelectorAll('button'))
                            .find(button => button.textContent.includes('Load More'));
                        if (loadMoreButton) loadMoreButton.click();
                    })();
                    """
                ]
                config = CrawlerRunConfig(
                    cache_mode=None,
                    js_code=js_code,
                    verbose=False,
                    page_timeout=30000,
                )
                try:
                    result = await crawler.arun(url, config=config)
                    snippet = result.markdown.raw_markdown[:MAX_CHARS]
                except Exception as exc:
                    logging.info(f"Error crawling {url}: {exc}")
                    snippet = "No text"
        else:
            snippet = text

        img_urls = find_image_urls(snippet)
        all_urls = find_all_link_urls(snippet)

        seen = set()
        non_image_urls = []
        for link in all_urls:
            if link not in img_urls and link not in seen:
                seen.add(link)
                non_image_urls.append(link)

        payload = {
            "url": url,
            "text": text[:4000],
            "non_image_links": non_image_urls,
        }
        logging.info(f"extract_urls_no_images invoke llm for: {url}")
        resp = await chat.ainvoke(
            [
                SYSTEM_EXTRACT,
                HumanMessage(
                    content=(
                        "The page's URL, text snippet, and list of non-image hyperlinks are provided: "
                        f"{payload}"
                    )
                ),
            ]
        )

        try:
            parsed = json.loads(resp.content)
            to_crawl = parsed.get("to_crawl", [])[:MAX_LINKS]
        except Exception as exc:
            logging.info("extract_urls_no_images JSON parse failed: %s", exc)
            to_crawl = non_image_urls[:MAX_LINKS]

        return {"to_crawl": to_crawl}

    return extract_urls_no_images


def make_judge_crawled_page(chat: Any):
    @tool
    async def judge_crawled_page(
        url: str,
        text: str,
    ) -> Dict[str, Any]:
        """
        Args:
            url: The page URL.
            text: The truncated page text.

        Returns:
            A dict:
            {
              "url": str,
              "malicious": bool,
              "confidence": int,
              "reason": str
            }
        """
        logging.info("judge_page received URL: %s", url)
        resp = await chat.ainvoke(
            [
                SYSTEM_JUDGE,
                HumanMessage(content=f"Judge if this URL: {url} is malicious or not."),
            ]
        )
        try:
            return json.loads(resp.content)
        except Exception as exc:
            logging.info("judge_page JSON parse failed: %s", exc)
            return {
                "url": url,
                "malicious": False,
                "confidence": 0.0,
                "reason": "Failed to parse model response",
            }

    return judge_crawled_page


class ExtractTargetsInput(BaseModel):
    url: str = Field(..., description="The URL of the page to analyze.")
    text: Optional[str] = Field(
        "",
        description="The page text (truncated) if already fetched; leave empty to auto-crawl.",
    )


class ExtractTargetsTool(BaseTool):
    name: str = "extract_targets_tool"
    description: str = (
        "Given a URL and its page text, select a small subset of links (`to_crawl`) "
        "and images (`to_check_images`) for deeper inspection."
    )
    args_schema: type = ExtractTargetsInput
    chat: Any

    def __init__(self, chat: Any):
        super().__init__(chat=chat)

    async def _arun(self, url: str, text: Optional[str] = "") -> Dict[str, List[str]]:
        snippet = text
        if not snippet:
            async with AsyncWebCrawler() as crawler:
                js_code = [
                    """
                            (() => {
                                const loadMoreButton = Array.from(document.querySelectorAll('button'))
                                    .find(button => button.textContent.includes('Load More'));
                                if (loadMoreButton) loadMoreButton.click();
                            })();
                            """
                ]
                config = CrawlerRunConfig(
                    cache_mode=None,
                    js_code=js_code,
                    verbose=False,
                )
                try:
                    result = await crawler.arun(url, config=config)
                    snippet = result.markdown.raw_markdown[:MAX_CHARS]
                except Exception as exc:
                    logging.info(f"Error crawling {url}: {exc}")
                    snippet = "No text"

        img_urls = find_image_urls(snippet)
        all_urls = find_all_link_urls(snippet)

        non_image_urls = []
        seen = set()
        for link in all_urls:
            if link not in img_urls and link not in seen:
                seen.add(link)
                non_image_urls.append(link)

        cleaned_img_urls = []
        for image_url in img_urls:
            try:
                cleaned_img_urls.append(furl(image_url).remove(query=True).url)
            except ValueError as exc:
                logging.info(f"Skipping invalid URL: {image_url} - Error: {exc}.")
                continue

        payload = {
            "url": url,
            "text": snippet,
            "non image links": non_image_urls,
            "images links": cleaned_img_urls,
        }
        start = time.perf_counter()
        resp = await self.chat.ainvoke(
            [
                SYSTEM_EXTRACT,
                HumanMessage(
                    content=(
                        "The page's URL, text snippet, list of hyperlinks, and list of image URLs are attached here: "
                        f"{payload}"
                    )
                ),
            ]
        )
        _timing["extract_targets_llm"].append(time.perf_counter() - start)

        try:
            parsed = json.loads(resp.content)
            to_crawl = parsed.get("to_crawl", [])[:MAX_LINKS]
            to_check_images = parsed.get("to_check_images", [])[:MAX_IMAGES]
        except Exception as exc:
            logging.info(f"extract_targets JSON parse failed: {exc}\n{resp.content}")
            to_crawl = non_image_urls[:MAX_LINKS]
            to_check_images = cleaned_img_urls[:MAX_IMAGES]

        return {"to_crawl": to_crawl, "to_check_images": to_check_images}

    def _run(self, *args, **kwargs):
        raise NotImplementedError("`extract_targets_tool` only supports async invocation.")


class CheckImageInput(BaseModel):
    img_url: str = Field(..., description="The URL of the image to fetch and describe.")


class CheckImageTool(BaseTool):
    name: str = "check_image"
    description: str = (
        "Fetch an image from a URL, send it to the LLM, and return a one-sentence description."
    )
    args_schema: type = CheckImageInput
    chat: Any

    def __init__(self, chat: Any):
        super().__init__(chat=chat)

    async def _arun(self, img_url: str) -> Dict[str, str]:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(img_url)
        image_data = base64.b64encode(response.content).decode("utf-8")
        img_type = response.headers.get("Content-Type", "image/jpeg")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    build_image_message(
                        image_b64=image_data,
                        media_type=img_type,
                        provider=get_provider_from_llm(self.chat),
                    ),
                ],
            }
        ]
        resp = await self.chat.ainvoke(messages)
        return {"image_url": img_url, "description": resp.content}

    def _run(self, *args, **kwargs):
        raise NotImplementedError(
            "`check_image` only supports async invocation via `_arun`."
        )


def make_judge_image(chat: Any):
    @tool(parse_docstring=True)
    async def judge_image(image_url: str, description: str) -> Dict[str, Any]:
        """
        Given an image URL and its textual description, decide if it is part of a phishing site.

        Args:
            image_url: The source URL of the image.
            description: A natural-language description of what appears in the image.

        Returns:
            A dict with url, malicious, confidence, and reason.
        """
        human_msg = HumanMessage(
            content=json.dumps({"image_url": image_url, "description": description})
        )
        resp = await chat.ainvoke([SYSTEM_JUDGE_IMG, human_msg])

        try:
            return json.loads(resp.content)
        except Exception:
            return {
                "url": image_url,
                "malicious": False,
                "confidence": 0.0,
                "reason": "Failed to parse model response",
            }

    return judge_image


class CheckScreenshotInput(BaseModel):
    url: str = Field(
        ...,
        description="URL(string) that you want to extract screenshot and analyze for phishing artifacts.",
    )


class CheckScreenshotTool(BaseTool):
    name: str = "check_screenshot"
    description: str = (
        "Analyze a base64-encoded screenshot for phishing-site artifacts. "
        "Returns a JSON dict with keys: `malicious`, `confidence`, and `notes`."
    )
    args_schema: type = CheckScreenshotInput
    chat: Any

    def __init__(self, chat: Any):
        super().__init__(chat=chat)

    async def _arun(self, url: str) -> Dict[str, Any]:
        prefixes = ["http://", "https://"]
        if any(url.startswith(prefix) for prefix in prefixes):
            candidates = [url]
        else:
            candidates = [f"{prefix}{url}" for prefix in prefixes]

        screenshot = ""
        async with AsyncWebCrawler() as crawler:
            js_code = [
                """
                        (() => {
                            const loadMoreButton = Array.from(document.querySelectorAll('button'))
                                .find(button => button.textContent.includes('Load More'));
                            if (loadMoreButton) loadMoreButton.click();
                        })();
                        """
            ]
            config = CrawlerRunConfig(
                cache_mode=None,
                js_code=js_code,
                verbose=False,
                screenshot=True,
            )
            for candidate_url in candidates:
                try:
                    start = time.perf_counter()
                    result = await crawler.arun(candidate_url, config=config)
                    _timing["check_screenshot_crawl"].append(
                        time.perf_counter() - start
                    )
                    screenshot = result.screenshot
                    break
                except Exception as exc:
                    logging.info(f"Error crawling {candidate_url}: {exc}")
                    continue

        if not screenshot:
            return {
                "url": url,
                "malicious": False,
                "confidence": 0.0,
                "reason": "Failed to capture screenshot",
            }

        try:
            raw = base64.b64decode(screenshot)
            img = Image.open(BytesIO(raw)).resize((256, 256), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=60, optimize=True)
            compressed_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            messages = [
                {"role": "system", "content": SYSTEM_SCREEN.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze whether this screenshot is a phishing site.",
                        },
                        build_image_message(
                            image_b64=compressed_b64,
                            media_type="image/jpeg",
                            provider=get_provider_from_llm(self.chat),
                        ),
                    ],
                },
            ]
        except Exception as exc:
            logging.info("read image failed: %s", exc)
            return {
                "url": url,
                "malicious": False,
                "confidence": 0.0,
                "reason": "Failed to parse screenshot",
            }

        start = time.perf_counter()
        resp = await self.chat.ainvoke(messages)
        _timing["check_screenshot_llm"].append(time.perf_counter() - start)

        try:
            return json.loads(resp.content)
        except (json.JSONDecodeError, AttributeError):
            return {
                "malicious": False,
                "confidence": 0,
                "notes": "Failed to parse model response.",
            }

    def _run(self, *args, **kwargs):
        raise NotImplementedError("Please use async invocation (_arun).")


class AgentTools:
    def __init__(self, llm: Any):
        self.crawl = CrawlContentTool()
        self.extract_links = make_extract_urls_no_images(llm)
        self.judge_crawled_page = make_judge_crawled_page(llm)
        self.check_screenshot = CheckScreenshotTool(llm)
        self.extract_targets = ExtractTargetsTool(llm)
        self.check_img = CheckImageTool(llm)
        self.judge_img = make_judge_image(llm)
        self.serpapi_search = serpapi_tool
