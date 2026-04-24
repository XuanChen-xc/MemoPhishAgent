import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from urllib.parse import urlparse

import boto3
import botocore.exceptions
from botocore.config import Config
from dotenv import load_dotenv
import openai
import tldextract
from langchain_aws.chat_models import ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from serpapi import GoogleSearch

load_dotenv()

Provider = Literal["openai", "bedrock"]

DEFAULT_PROVIDER: Provider = "openai"
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_BEDROCK_MODEL = "anthropic.claude-3-sonnet"
DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_BEDROCK_EMBED_MODEL = "amazon.titan-embed-image-v1"
SERPAPI_ENV = "SERPAPI_API_KEY"
OPENAI_ENV = "OPENAI_API_KEY"


def normalize_provider(provider: Optional[str] = None) -> Provider:
    raw = (provider or os.environ.get("MEMOPHISH_PROVIDER") or DEFAULT_PROVIDER).lower()
    if raw not in {"openai", "bedrock"}:
        raise ValueError(
            f"Unsupported provider '{raw}'. Expected one of: openai, bedrock."
        )
    return raw  # type: ignore[return-value]


def get_model_id(provider: Optional[str] = None, model: Optional[str] = None) -> str:
    selected = normalize_provider(provider)
    if model:
        return model
    if selected == "openai":
        return os.environ.get("MEMOPHISH_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    return os.environ.get("MEMOPHISH_BEDROCK_MODEL", DEFAULT_BEDROCK_MODEL)


def get_provider_from_llm(llm: Any) -> Provider:
    return normalize_provider(getattr(llm, "memo_provider", DEFAULT_PROVIDER))


def get_serpapi_api_key() -> str:
    api_key = os.environ.get(SERPAPI_ENV)
    if api_key:
        return api_key

    for candidate in (Path("serpAPI_key.txt"), Path("../serpAPI_key.txt")):
        if candidate.exists():
            return candidate.read_text().strip()
    return ""


def get_bedrock_client(region: Optional[str] = None) -> Any:
    """Instantiate the AWS Bedrock client and fail early if credentials are absent."""
    region_name = region or AWS_REGION
    try:
        session = boto3.Session(region_name=region_name)
        credentials = session.get_credentials()
    except botocore.exceptions.ProfileNotFound as exc:
        raise ValueError(
            "Bedrock provider selected, but the configured AWS profile could not be found."
        ) from exc

    if credentials is None:
        raise ValueError(
            "Bedrock provider selected, but no AWS credentials were found. "
            "Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, AWS_PROFILE, or mount a valid ~/.aws configuration."
        )

    config = Config(
        read_timeout=2000,
        retries={
            "max_attempts": 50,
            "mode": "adaptive",
        },
    )
    return session.client("bedrock-runtime", region_name=region_name, config=config)


def get_llm(
    provider: Optional[str] = None,
    callbacks: Optional[List[Any]] = None,
    model: Optional[str] = None,
    region: Optional[str] = None,
) -> Any:
    """Instantiate the configured chat model for the selected provider."""
    callbacks = callbacks or []
    selected = normalize_provider(provider)
    model_id = get_model_id(selected, model)

    if selected == "openai":
        api_key = os.environ.get(OPENAI_ENV)
        if not api_key:
            raise ValueError(
                "OpenAI provider selected, but OPENAI_API_KEY is not set."
            )
        llm = ChatOpenAI(
            model=model_id,
            api_key=api_key,
            callbacks=callbacks,
            temperature=0,
        )
    else:
        client = get_bedrock_client(region=region)
        llm = ChatBedrock(
            client=client,
            model_id=model_id,
            callbacks=callbacks,
            beta_use_converse_api=False,
        )

    setattr(llm, "memo_provider", selected)
    setattr(llm, "memo_model_id", model_id)
    return llm


def get_memory_embeddings(provider: Optional[str] = None) -> Tuple[Any, int]:
    """Return provider-aligned embeddings plus their vector dimension."""
    selected = normalize_provider(provider)
    if selected == "openai":
        api_key = os.environ.get(OPENAI_ENV)
        if not api_key:
            raise ValueError(
                "OpenAI embeddings requested, but OPENAI_API_KEY is not set."
            )
        model_name = os.environ.get(
            "MEMOPHISH_OPENAI_EMBED_MODEL", DEFAULT_OPENAI_EMBED_MODEL
        )
        return OpenAIEmbeddings(model=model_name, api_key=api_key), 1536

    client = get_bedrock_client()
    model_name = os.environ.get(
        "MEMOPHISH_BEDROCK_EMBED_MODEL", DEFAULT_BEDROCK_EMBED_MODEL
    )
    return BedrockEmbeddings(model_id=model_name, client=client), 1024


def build_image_message(
    image_b64: str,
    media_type: str,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    selected = normalize_provider(provider)
    if selected == "openai":
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{image_b64}"},
        }
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": image_b64,
        },
    }


def is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    if isinstance(exc, botocore.exceptions.ClientError):
        error = exc.response.get("Error", {})
        return error.get("Code") == "ThrottlingException"
    return False


def extract_json_from_llm_output(output: str) -> Dict:
    """
    Extracts and parses a JSON object from the output of an LLM.
    """
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output, re.DOTALL)
    parsed = {
        "url": "None",
        "malicious": False,
        "confidence": 0.0,
        "reason": "Could not parse model response",
    }
    if not match:
        match = re.search(r"(\{.*\})", output, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as exc:
            print(f"JSON parsing error: {exc}")
            return parsed
    print("No JSON found in the output.")
    return parsed


def extract_and_fix(text: str) -> Any:
    """
    Extract all JSON snippets from a text, attempt to parse each.
    """
    results = []
    i = 0
    total_len = len(text)
    while True:
        start = text.find("{", i)
        if start < 0:
            break
        depth = 0
        for j in range(start, total_len):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : j + 1].replace("\n", " ")
                    try:
                        results.append(json.loads(snippet))
                    except json.JSONDecodeError:
                        pass
                    i = j + 1
                    break
        else:
            break
    return results


def find_image_urls(markdown_text: str) -> set[str]:
    """
    Extract markdown image URLs ending in .jpg/.png so we can exclude them.
    """
    img_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
    urls = img_pattern.findall(markdown_text)
    return {u for u in urls if re.search(r"\.(jpg|png)(?:\?|$)", u, re.IGNORECASE)}


def find_all_link_urls(markdown_text: str) -> List[str]:
    link_pattern = re.compile(r"\[.*?\]\((.*?)\)")
    return link_pattern.findall(markdown_text)


SKIP_DOMAINS = {
    "sites.google.com",
    "github.io",
    "gitlab.io",
    "netlify.app",
}


def should_skip(url: str) -> bool:
    hostname = urlparse(url).netloc.lower()
    return any(hostname.endswith(domain) for domain in SKIP_DOMAINS)


def extract_domain_and_brand(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    hostname = parsed.netloc or parsed.path
    ext = tldextract.extract(hostname)
    brand = ext.domain.replace("-", " ").replace("_", " ").title()
    return hostname, brand


def make_queries(domain: str, brand: str) -> List[str]:
    return [
        f"{domain} overview",
        f"{brand} site:{domain}",
        f"site:{domain} overview",
        f'"{domain} phishing"',
        f"info:{domain}",
        f"related:{domain}",
        f"info:{domain} scam",
        f"related:{domain} scam",
        f"link:{domain} scam",
        f"link:{domain} phishing",
    ]


def google_search_with_retry(q: str, api_key: str, max_retries: int = 5, delay_seconds: int = 3) -> Dict[str, Any]:
    """Perform a Google Search, retrying on transient failures."""
    params = {
        "engine": "google",
        "q": q,
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "api_key": api_key,
    }

    for attempt in range(1, max_retries + 1):
        try:
            return GoogleSearch(params).get_dict()
        except Exception as exc:
            print(f"Attempt {attempt} failed: {exc!r}")
            time.sleep(delay_seconds)

    print("All retries exhausted.")
    return {"ai_overview": None}


def fetch_ai_overview_for_query(q: str) -> Optional[Dict[str, Any]]:
    api_key = get_serpapi_api_key()
    if not api_key:
        return None

    resp = google_search_with_retry(q, api_key)
    ai_block = resp.get("ai_overview", {})
    token = ai_block.get("page_token")
    if not token:
        return None

    resp2 = GoogleSearch(
        {"engine": "google_ai_overview", "page_token": token, "api_key": api_key}
    ).get_dict()

    try:
        return resp2.get("ai_overview", None)
    except Exception:
        print(resp2)
        print("No ai overview available.")
        return None


def extract_text(ai_overview: Dict[str, Any]) -> str:
    snippets = []
    for block in ai_overview.get("text_blocks", []):
        block_type = block.get("type")
        if block_type == "paragraph":
            snippets.append(block.get("snippet", ""))
        elif block_type == "list":
            for item in block.get("list", []):
                if item.get("snippet"):
                    snippets.append(item["snippet"])
    return "\n".join(snippets)


def analyze_with_llm(llm: Any, text: str, url: str) -> Optional[Dict[str, Any]]:
    system = {
        "role": "system",
        "content": (
            """
           You are an AI security analyst. The following text snippets were obtained by performing Google searches on the domain of the target URL.
           Using only this information, determine whether the URL is benign or malicious.
           Output only the raw JSON object, with no markdown, no code fences.
           Respond in JSON format:
                - "url": the original url (provided by the user)
                - "malicious": true or false
                - "confidence": integer from 0 (lowest) to 5 (highest)
                - "reason": one concise sentence explaining your judgment
        """
        ),
    }
    user = {
        "role": "user",
        "content": (f"URL is: {url}. \n Google serach text snippet:\n{text}"),
    }
    try:
        resp = llm.invoke([system, user])
        return json.loads(resp.content)
    except Exception as exc:
        print(f"Error {exc}, return None")
        return None


def ai_overview_preprocess(url: str, llm: Any) -> Optional[Dict[str, Any]]:
    if should_skip(url):
        return None

    domain, brand = extract_domain_and_brand(url)
    for q in make_queries(domain, brand):
        overview = fetch_ai_overview_for_query(q)
        if overview:
            text = extract_text(overview)
            return analyze_with_llm(llm, text, url)
    return None
