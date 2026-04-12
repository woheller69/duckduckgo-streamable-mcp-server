import argparse
import asyncio
import logging
import sys
import traceback
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List
import httpx
import uvicorn
from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests
from fastmcp import FastMCP
from markdownify import markdownify as md
from mcp.server.fastmcp import Context
from starlette.middleware.cors import CORSMiddleware

# Configure logging once at module level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("duckduckgo_mcp")

@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30):
        self.requests_per_minute = requests_per_minute
        self.requests = []

    async def acquire(self):
        now = datetime.now()
        # Remove requests older than 1 minute
        self.requests = [
            req for req in self.requests if now - req < timedelta(minutes=1)
        ]

        if len(self.requests) >= self.requests_per_minute:
            # Wait until we can make another request
            wait_time = 60 - (now - self.requests[0]).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class DuckDuckGoSearcher:
    BASE_URL = "https://html.duckduckgo.com/html"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.logger = logging.getLogger("duckduckgo_mcp.searcher")

    def format_results_for_llm(self, results: List[SearchResult]) -> str:
        """Format results in a natural language style"""
        if not results:
            return "No results were found for your search query..."
        output = []
        output.append(f"Found {len(results)} search results:\n")
        for result in results:
            output.append(f"{result.position}. {result.title}")
            output.append(f"   URL: {result.link}")
            output.append(f"   Summary: {result.snippet}")
            output.append("")
        return "\n".join(output)

    async def search(
        self, query: str, ctx: Context, max_results: int = 10
    ) -> List[SearchResult]:
        try:
            await self.rate_limiter.acquire()
            data = {"q": query, "b": "", "kl": ""}
            self.logger.info(f"Searching DuckDuckGo for: {query}")

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.BASE_URL, data=data, headers=self.HEADERS, timeout=30.0
                )
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            if not soup:
                self.logger.error("Failed to parse HTML response")
                return []

            results = []
            for result in soup.select(".result"):
                title_elem = result.select_one(".result__title")
                if not title_elem:
                    continue

                link_elem = title_elem.find("a")
                if not link_elem:
                    continue

                title = link_elem.get_text(strip=True)
                link = link_elem.get("href", "")

                if "y.js" in link:
                    continue

                if link.startswith("//duckduckgo.com/l/?uddg="):
                    link = urllib.parse.unquote(link.split("uddg=")[1].split("&")[0])

                snippet_elem = result.select_one(".result__snippet")
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(
                    SearchResult(
                        title=title,
                        link=link,
                        snippet=snippet,
                        position=len(results) + 1,
                    )
                )

                if len(results) >= max_results:
                    break

            self.logger.info(f"Successfully found {len(results)} results")
            return results

        except httpx.TimeoutException:
            self.logger.error("Search request timed out")
            return []
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP error occurred: {str(e)}")
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error during search: {str(e)}", exc_info=True)
            return []


class WebContentFetcher:
    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_minute=20)
        self.logger = logging.getLogger("duckduckgo_mcp.fetcher")
        # Cache for fetched content: {url: full_text}
        self.content_cache: dict[str, str] = {}

    def find_content(self, soup, url=None):
        # Parse URL parts for domain/path-based logic
        domain = ""
        path = ""
        if url:
            try:
                parsed = urllib.parse.urlparse(url)
                domain = parsed.netloc.lower()
                path = parsed.path.lower()
            except Exception:
                pass
        # Special case for Android Developer reference pages
        if url and "developer.android.com" in domain and path.startswith("/reference"):
            article = soup.find('article', class_='devsite-article')
            if article:
                return article

        if url and "wikipedia.org" in domain and path.startswith("/wiki"):
            article = soup.find(class_="mw-body-content")
            if article:
                return article

        # For other URLs, use your existing heuristics
        content_selectors = [
            {'tag': 'div', 'attr': {'class': 'main-content'}},
            {'tag': 'main', 'attr': {}},
            {'tag': 'article', 'attr': {}}
        ]

        for selector in content_selectors:
            content = soup.find(selector['tag'], attrs=selector['attr'])
            if content:
                return content

        # As a fallback, return the body or None
        return soup.find('body') or None
            
    async def fetch_and_parse(
        self, url: str, ctx: Context, page: int = 1, chunk_size: int = 20000
    ) -> str:
        """
        Fetch and parse content from a webpage URL with pagination support.

        Args:
            url: The webpage URL to fetch content from
            ctx: MCP context for logging
            page: Page number for pagination (default: 1)
            chunk_size: Size of each chunk in characters (default: 20000)
        """
        try:
            await self.rate_limiter.acquire()
            self.logger.info(f"Fetching content from: {url}")

            # Check cache first
            if url in self.content_cache:
                full_text = self.content_cache[url]
            else:
                response = curl_requests.get(
                    url,
                    impersonate="chrome120",  # or "firefox110", "safari15_3", etc.
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                    timeout=30
                )

                soup = BeautifulSoup(response.text, "html.parser")

                # Find content using heuristics
                content_div = self.find_content(soup, url)

                if content_div:
                    # Converting to Markdown
                    full_text = md(str(content_div), heading_style="ATX")
                    self.content_cache[url] = full_text
                else:
                    return "Content not found."

            total_chars = len(full_text)
            pages_total = (total_chars + chunk_size - 1) // chunk_size
            # Calculate range for pagination
            start_idx = (page - 1) * chunk_size
            end_idx = min(start_idx + chunk_size, total_chars)

            page_text = full_text[start_idx:end_idx]

            # Add header info
            if pages_total > 0:
                # Check if page is out of range
                if page > pages_total:
                    return (
                        f"[CONTENT PAGINATION: Page {page} of {pages_total}] "
                        f"out of range (total characters: {total_chars})"
                    )
                else:
                    page_info = (
                        f"[CONTENT PAGINATION: Page {page} of {pages_total}] "
                        f"(characters {start_idx+1}-{end_idx} of {total_chars})\n\n"
                    )
            else:
                page_info = ""

            return page_info + page_text

        except Exception as e:
            self.logger.error(f"Error fetching content from {url}: {str(e)}", exc_info=True)
            return f"Error: An unexpected error occurred while fetching the webpage ({str(e)})"
                


# Initialize FastMCP server
mcp = FastMCP("ddg-search")
searcher = DuckDuckGoSearcher()
fetcher = WebContentFetcher()


@mcp.tool()
async def search(query: str, ctx: Context, max_results: int = 10) -> str:
    """
    Search DuckDuckGo and return formatted results.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 10)
        ctx: MCP context for logging
    """
    try:
        results = await searcher.search(query, ctx, max_results)
        return searcher.format_results_for_llm(results)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return f"An error occurred while searching: {str(e)}"


@mcp.tool()
async def fetch_content(
    url: str, ctx: Context, page: int = 1, chunk_size: int = 20000
) -> str:
    """
    Fetch and parse content from a webpage URL with pagination support.

    Args:
        url: The webpage URL to fetch content from
        ctx: MCP context for logging
        page: Page number for pagination (default: 1). Subsequent pages are read from cache.
        chunk_size: Size of each page in characters (default: 20000)

    Returns:
        Cleaned and formatted text content from the webpage formatted as markdown.

    Note:
        This tool uses HTTP requests to fetch page content (using curl_cffi),
        so it does not execute JavaScript. Dynamic content loaded via JavaScript
        may not be visible in the results.
    """
    return await fetcher.fetch_and_parse(url, ctx, page, chunk_size)

# Create ASGI app (NOT using mcp.run())
app = mcp.http_app(path="/mcp")
print(f"✓ Created ASGI app: {type(app)}")  # Should show <class 'starlette.applications.Starlette'>


# Add CORS middleware - critical for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify exact origins in production!
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE"],  # Must include OPTIONS for preflight
    allow_headers=[
        "Content-Type",
        "Authorization",
        "mcp-protocol-version",
        "mcp-session-id",
    ],
    expose_headers=["mcp-session-id"],  # Required for session management in browsers
)
print("✓ CORS middleware added")

# Add argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="DuckDuckGo MCP Server - Search & Fetch Tools"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Server port (default: 3000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"🚀 Starting MCP server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

