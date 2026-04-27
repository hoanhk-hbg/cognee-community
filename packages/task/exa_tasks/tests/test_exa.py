import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cognee_community_tasks_exa import search_web
from cognee_community_tasks_exa.exa_task import (
    ExaSearchResult,
    _parse_result,
    search_and_add,
)


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "test-api-key")


def _fake_result(**overrides):
    """Build a SimpleNamespace mimicking an exa-py Result."""
    data = {
        "url": "https://example.com",
        "title": "Example",
        "text": None,
        "highlights": None,
        "summary": None,
        "score": 0.9,
        "published_date": "2026-01-01",
        "author": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def _fake_exa_instance(results):
    inst = MagicMock()
    inst.headers = {}
    inst.search_and_contents.return_value = SimpleNamespace(results=results)
    inst.search.return_value = SimpleNamespace(results=results)
    return inst


class TestSearchWeb:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("EXA_API_KEY", raising=False)
        with pytest.raises(ValueError, match="EXA_API_KEY"):
            asyncio.run(search_web("anything", api_key=None))

    def test_returns_parsed_results(self):
        results = [
            _fake_result(url="https://a.example", title="A", text="body A"),
            _fake_result(url="https://b.example", title="B", text="body B"),
        ]
        inst = _fake_exa_instance(results)

        with patch("cognee_community_tasks_exa.exa_task.Exa", return_value=inst):
            out = asyncio.run(search_web("hello world", num_results=2))

        assert len(out) == 2
        assert out[0]["url"] == "https://a.example"
        assert out[0]["title"] == "A"
        assert out[0]["content"] == "body A"
        inst.search_and_contents.assert_called_once()
        call_kwargs = inst.search_and_contents.call_args.kwargs
        assert call_kwargs["query"] == "hello world"
        assert call_kwargs["num_results"] == 2
        assert call_kwargs["text"] is True

    def test_calls_plain_search_when_no_contents_requested(self):
        inst = _fake_exa_instance([_fake_result()])
        with patch("cognee_community_tasks_exa.exa_task.Exa", return_value=inst):
            asyncio.run(
                search_web(
                    "q",
                    include_text=False,
                    include_highlights=False,
                    include_summary=False,
                )
            )
        inst.search.assert_called_once()
        inst.search_and_contents.assert_not_called()

    def test_sets_integration_header(self):
        inst = _fake_exa_instance([_fake_result()])
        with patch("cognee_community_tasks_exa.exa_task.Exa", return_value=inst):
            asyncio.run(search_web("q"))
        assert inst.headers.get("x-exa-integration") == "cognee-community"

    def test_passes_filters(self):
        inst = _fake_exa_instance([])
        with patch("cognee_community_tasks_exa.exa_task.Exa", return_value=inst):
            asyncio.run(
                search_web(
                    "q",
                    include_domains=["example.com"],
                    exclude_domains=["spam.com"],
                    category="research paper",
                    start_published_date="2026-01-01",
                    end_published_date="2026-04-01",
                    include_highlights=True,
                    include_summary=True,
                    summary_query="summarize this",
                )
            )
        kwargs = inst.search_and_contents.call_args.kwargs
        assert kwargs["include_domains"] == ["example.com"]
        assert kwargs["exclude_domains"] == ["spam.com"]
        assert kwargs["category"] == "research paper"
        assert kwargs["start_published_date"] == "2026-01-01"
        assert kwargs["end_published_date"] == "2026-04-01"
        assert kwargs["highlights"] is True
        assert kwargs["summary"] == {"query": "summarize this"}

    def test_explicit_api_key_is_used(self):
        inst = _fake_exa_instance([])
        with patch("cognee_community_tasks_exa.exa_task.Exa", return_value=inst) as mock_cls:
            asyncio.run(search_web("q", api_key="explicit-key"))
        mock_cls.assert_called_once_with(api_key="explicit-key")


class TestContentFallback:
    def test_prefers_text_over_summary_and_highlights(self):
        r = _parse_result(_fake_result(text="full text", summary="summary text", highlights=["h1"]))
        assert r.best_content() == "full text"

    def test_falls_back_to_summary_when_text_missing(self):
        r = _parse_result(_fake_result(text=None, summary="summary text", highlights=["h1"]))
        assert r.best_content() == "summary text"

    def test_falls_back_to_highlights_when_text_and_summary_missing(self):
        r = _parse_result(_fake_result(text=None, summary=None, highlights=["h1", "h2"]))
        assert r.best_content() == "h1\nh2"

    def test_returns_empty_when_nothing_available(self):
        r = _parse_result(_fake_result(text=None, summary=None, highlights=None))
        assert r.best_content() == ""

    def test_parses_dict_input(self):
        r = _parse_result(
            {
                "url": "https://x.example",
                "title": "X",
                "text": "text",
                "highlights": ["h"],
                "score": 0.5,
                "publishedDate": "2026-01-01",
            }
        )
        assert isinstance(r, ExaSearchResult)
        assert r.url == "https://x.example"
        assert r.highlights == ["h"]
        assert r.published_date == "2026-01-01"


class TestSearchAndAdd:
    def test_raises_when_no_results_have_content(self):
        inst = _fake_exa_instance([_fake_result(text=None, summary=None, highlights=None)])

        with (
            patch("cognee_community_tasks_exa.exa_task.Exa", return_value=inst),
            pytest.raises(RuntimeError, match="No Exa results"),
        ):
            asyncio.run(search_and_add(query="q"))

    def test_calls_cognee_add_and_cognify(self):
        inst = _fake_exa_instance([_fake_result(text="body content")])

        mock_cognee = MagicMock()
        mock_cognee.add = AsyncMock()
        mock_cognee.cognify = AsyncMock(return_value="graph_result")

        with (
            patch("cognee_community_tasks_exa.exa_task.Exa", return_value=inst),
            patch("cognee_community_tasks_exa.exa_task.cognee", mock_cognee),
        ):
            result = asyncio.run(search_and_add(query="q", dataset_name="test_dataset"))

        mock_cognee.add.assert_called_once()
        call_kwargs = mock_cognee.add.call_args
        assert "test_dataset" in str(call_kwargs)

        mock_cognee.cognify.assert_called_once()
        assert result == "graph_result"
