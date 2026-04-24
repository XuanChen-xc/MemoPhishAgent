import argparse
import asyncio
import json
import os
import time

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from langchain_core.messages import HumanMessage, SystemMessage

from callbacks import get_token_usage_callbacks
from utils import get_llm, is_rate_limit_error


async def crawl_and_judge(urls, llm, max_text_chars=2000, max_screenshot_chars=100000):
    """Crawls a list of URLs and uses an LLM to determine if each URL is malicious."""
    malicious_urls = []

    async with AsyncWebCrawler() as crawler:
        for idx, url in enumerate(urls):
            if not url:
                print(f"Skipping record {idx + 1}: Empty URL")
                continue

            print(f"[{idx + 1}/{len(urls)}] Processing: {url}")
            try:
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
                    cache_mode=CacheMode.ENABLED,
                    js_code=js_code,
                    verbose=False,
                    screenshot=True,
                )
                result = await crawler.arun(url=url, config=config)

                try:
                    raw_markdown = result.markdown.raw_markdown or ""
                except AttributeError:
                    raw_markdown = ""

                screenshot_b64 = result.screenshot or ""
                text_snippet = raw_markdown[:max_text_chars]
                screenshot_snippet = screenshot_b64[:max_screenshot_chars]

                messages = [
                    SystemMessage(
                        content="""
                                You are a cybersecurity analyst. You will receive for each URL:
                                1. The URL string
                                2. A snippet of the page text (up to 2000 chars)
                                3. A base64 screenshot of the page

                                Based on all of this, decide if the site is an phishing attack.
                                Return JSON exactly:

                                {
                                "url": "<the URL>",
                                "malicious": <true|false>,
                                "confidence": <0/1/2/3/4/5>,
                                "reason": "<one-sentence rationale>"
                                }
                            """
                    ),
                    HumanMessage(
                        content=f"""
                                    Judge this URL with the following crawled data:
                                    URL: {url}
                                    Page text snippet:
                                        {text_snippet}
                                    Screenshot (base64, truncated):
                                        {screenshot_snippet}
                                """
                    ),
                ]

                ai_msg = llm.invoke(messages)
                print(ai_msg.content)

                try:
                    verdict = json.loads(ai_msg.content)
                    if verdict.get("malicious"):
                        malicious_urls.append(url)
                except json.JSONDecodeError:
                    print(f"Failed to parse LLM response for {url}")

            except Exception as exc:
                if is_rate_limit_error(exc):
                    print(f"Rate limit for {url}: {exc}. Sleeping 3 seconds...")
                    await asyncio.sleep(3)
                    continue
                raise

    return malicious_urls


async def main_async(input_path, output_path, provider, model):
    with open(input_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    token_callback = get_token_usage_callbacks()
    llm = get_llm(provider=provider, model=model, callbacks=[token_callback])

    start_all = time.perf_counter()
    malicious = await crawl_and_judge(urls, llm)
    total_time = time.perf_counter() - start_all
    avg_time = total_time / len(urls) if urls else 0

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as file:
        for line in malicious:
            file.write(line + "\n")

    print(
        f"Processed {len(urls)} URLs. Found {len(malicious)} malicious. Avg judge time: {avg_time:.2f}s URL"
    )
    print(f"Token usage: {token_callback.usage_metadata}")


def main():
    parser = argparse.ArgumentParser(
        description="Monolithic baseline for phishing detection"
    )
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
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input file with URLs, one per line",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to write the malicious URLs"
    )
    args = parser.parse_args()

    asyncio.run(main_async(args.input, args.output, args.provider, args.model))


if __name__ == "__main__":
    main()
