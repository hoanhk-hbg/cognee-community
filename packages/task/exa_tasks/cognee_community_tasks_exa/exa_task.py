import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Literal

import cognee
from cognee.shared.logging_utils import get_logger
from exa_py import Exa

logger = get_logger("ExaTask")

SearchType = Literal["auto", "neural", "fast", "deep-lite", "deep", "deep-reasoning", "instant"]
Category = Literal[
    "company",
    "research paper",
    "news",
    "personal site",
    "financial report",
    "pdf",
    "github",
    "tweet",
    "movie",
    "song",
    "linkedin profile",
]


@dataclass
class ExaSearchResult:
    """Typed representation of a single Exa search result."""

    url: str
    title: str | None = None
    text: str | None = None
    highlights: list[str] = field(default_factory=list)
    summary: str | None = None
    score: float | None = None
    published_date: str | None = None
    author: str | None = None

    def best_content(self) -> str:
        """Return the most informative content field that is available.

        Exa may return any combination of text/highlights/summary depending on
        the request. Cascade through fields rather than assuming one is set.
        """
        if self.text:
            return self.text
        if self.summary:
            return self.summary
        if self.highlights:
            return "\n".join(self.highlights)
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.best_content(),
            "text": self.text,
            "highlights": self.highlights,
            "summary": self.summary,
            "score": self.score,
            "published_date": self.published_date,
            "author": self.author,
        }


def _parse_result(raw: Any) -> ExaSearchResult:
    """Parse an exa-py Result object (or dict-like) into an ExaSearchResult."""
    get = (lambda k: raw.get(k)) if isinstance(raw, dict) else (lambda k: getattr(raw, k, None))
    highlights = get("highlights") or []
    if not isinstance(highlights, list):
        highlights = [highlights]
    return ExaSearchResult(
        url=get("url") or "",
        title=get("title"),
        text=get("text"),
        highlights=list(highlights),
        summary=get("summary"),
        score=get("score"),
        published_date=get("published_date") or get("publishedDate"),
        author=get("author"),
    )


def _build_exa_client(api_key: str | None) -> Exa:
    if api_key is None:
        api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("Exa API key is required. Set the EXA_API_KEY environment variable.")
    client = Exa(api_key=api_key)
    client.headers["x-exa-integration"] = "cognee-community"
    return client


async def search_web(
    query: str,
    num_results: int = 10,
    search_type: SearchType = "auto",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    category: Category | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
    include_text: bool = True,
    include_highlights: bool = False,
    include_summary: bool = False,
    summary_query: str | None = None,
    api_key: str | None = None,
) -> list[dict]:
    """Search the web with Exa and return structured results.

    Parameters
    ----------
    query : str
        Natural language search query.
    num_results : int
        Number of results to return (1-100, default 10).
    search_type : str
        One of ``auto``, ``neural``, ``fast``, ``deep-lite``, ``deep``,
        ``deep-reasoning``, ``instant``.
    include_domains, exclude_domains : Optional[List[str]]
        Domain allow/deny lists.
    category : Optional[str]
        Category filter (e.g. ``company``, ``research paper``, ``news``).
    start_published_date, end_published_date : Optional[str]
        ISO 8601 date bounds on publication date.
    include_text : bool
        Return the full page text content.
    include_highlights : bool
        Return query-relevant highlights.
    include_summary : bool
        Return an LLM-generated summary. Combine with ``summary_query`` for
        a targeted summary.
    summary_query : Optional[str]
        Natural language prompt used to generate each result's summary.
    api_key : Optional[str]
        Exa API key. Falls back to the ``EXA_API_KEY`` environment variable.

    Returns
    -------
    List[dict]
        Result dicts with keys ``url``, ``title``, ``content``, ``text``,
        ``highlights``, ``summary``, ``score``, ``published_date``, ``author``.
    """
    client = _build_exa_client(api_key)

    contents: dict[str, Any] = {}
    if include_text:
        contents["text"] = True
    if include_highlights:
        contents["highlights"] = True
    if include_summary:
        contents["summary"] = {"query": summary_query} if summary_query else True

    kwargs: dict[str, Any] = {
        "query": query,
        "num_results": num_results,
        "type": search_type,
    }
    if include_domains:
        kwargs["include_domains"] = include_domains
    if exclude_domains:
        kwargs["exclude_domains"] = exclude_domains
    if category:
        kwargs["category"] = category
    if start_published_date:
        kwargs["start_published_date"] = start_published_date
    if end_published_date:
        kwargs["end_published_date"] = end_published_date

    logger.info(f"Running Exa search: {query!r} (type={search_type}, n={num_results})")

    def _do_search():
        if contents:
            return client.search_and_contents(**kwargs, **contents)
        return client.search(**kwargs)

    try:
        response = await asyncio.to_thread(_do_search)
    except Exception as e:
        logger.error(f"Exa search failed: {e!s}")
        raise

    raw_results = getattr(response, "results", response)
    parsed = [_parse_result(r) for r in raw_results]
    logger.info(f"Exa returned {len(parsed)} results")
    return [r.to_dict() for r in parsed]


async def search_and_add(
    query: str,
    num_results: int = 10,
    search_type: SearchType = "auto",
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    category: Category | None = None,
    start_published_date: str | None = None,
    end_published_date: str | None = None,
    summary_query: str | None = None,
    api_key: str | None = None,
    dataset_name: str = "exa",
) -> Any:
    """Run an Exa search and add the returned content to a cognee dataset.

    Each result is combined into a single text document and passed to
    ``cognee.add`` followed by ``cognee.cognify``.

    Parameters mirror :func:`search_web`. Content retrieval is forced on so
    that there is something to ingest.
    """
    results = await search_web(
        query=query,
        num_results=num_results,
        search_type=search_type,
        include_domains=include_domains,
        exclude_domains=exclude_domains,
        category=category,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        include_text=True,
        include_highlights=True,
        include_summary=summary_query is not None,
        summary_query=summary_query,
        api_key=api_key,
    )

    usable = [r for r in results if r.get("content")]
    if not usable:
        raise RuntimeError("No Exa results returned any content to ingest.")

    combined_text = "\n\n".join(
        f"Source: {r['url']}\nTitle: {r.get('title') or ''}\n{r['content']}" for r in usable
    )

    await cognee.add(combined_text, dataset_name=dataset_name)
    result = await cognee.cognify()

    logger.info(
        f"Added {len(usable)} Exa results for query {query!r} to cognee dataset '{dataset_name}'"
    )
    return result
