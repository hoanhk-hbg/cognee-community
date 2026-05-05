import asyncio
import os

import cognee

from cognee_community_tasks_exa import search_and_add, search_web


async def main():
    # Set required API keys
    os.environ["LLM_API_KEY"] = os.getenv("LLM_API_KEY", "YOUR_OPENAI_API_KEY")
    os.environ["EXA_API_KEY"] = os.getenv("EXA_API_KEY", "YOUR_EXA_API_KEY")

    query = "How do knowledge graphs improve LLM memory?"

    # --- Example 1: search only ---
    print(f"Searching Exa for: {query!r}")
    results = await search_web(
        query=query,
        num_results=5,
        include_highlights=True,
    )
    for item in results:
        print(f"\nURL: {item['url']}")
        print(f"  Title: {item.get('title')}")
        content_preview = str(item["content"])[:300]
        print(f"  Content preview: {content_preview}...")

    # --- Example 2: search and add to cognee ---
    print("\nSearching and adding results to cognee...")
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await search_and_add(
        query=query,
        num_results=5,
        dataset_name="exa_search",
    )

    search_results = await cognee.search("What is cognee?")
    print("\nSearch results after ingestion:")
    for result in search_results:
        print(f"  - {result}")


if __name__ == "__main__":
    asyncio.run(main())
