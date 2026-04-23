"""Explicit sync helpers for the Moss adapter.

The cognee-memory plugin normally calls cognee.improve() via its
SessionEnd hook to promote per-session memory into the shared graph
and apply feedback weights. In non-interactive modes like `claude -p`,
that hook is frequently cancelled before it finishes, so cross-session
sync never actually runs.

This module exposes sync functions any caller (orchestrator script,
CLI tool, custom hook) can invoke explicitly - independent of the
Claude Code hook lifecycle. Both sync (blocking) and async variants
are provided so callers can pick whichever fits their context.

Typical uses:

    # From a synchronous orchestrator (no event loop running):
    from cognee_community_vector_adapter_moss import sync_session
    sync_session(dataset="shared_skills", session_id="agent-1")

    # From an async pipeline (event loop already running):
    from cognee_community_vector_adapter_moss import sync_session_async
    await sync_session_async(dataset="shared_skills", session_id="agent-1")

    # From a Claude Code hook script (blocking):
    #   python -c "from cognee_community_vector_adapter_moss import sync_session; \
    #              sync_session('shared_skills', 'agent-1')"
"""

import asyncio
from typing import Any, Optional
from uuid import UUID


async def sync_session_async(
    dataset: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> Any:
    """Run cognee.improve() to completion for a specific session.

    Promotes per-session findings into the permanent knowledge graph,
    applies feedback weights from scores, and runs default enrichment
    (triplet embeddings). Returns only after the pipeline has fully
    committed, so callers can trust subsequent reads see the new state.

    Args:
        dataset: Cognee dataset name (e.g., "shared_skills").
        session_id: The session whose memory should be promoted.
        user_id: Optional UUID string; falls back to the default user.

    Returns:
        Whatever cognee.improve() returns (dataset -> run info mapping).
    """
    import cognee
    from cognee.modules.users.methods import get_default_user, get_user

    user = None
    if user_id:
        try:
            user = await get_user(UUID(user_id))
        except Exception:
            user = None
    if user is None:
        user = await get_default_user()

    # run_in_background=False: block until the improve pipeline finishes.
    # This is the whole point of an explicit sync - when we return, the
    # shared graph is up to date.
    return await cognee.improve(
        dataset=dataset,
        session_ids=[session_id],
        run_in_background=False,
        user=user,
    )


def sync_session(
    dataset: str,
    session_id: str,
    user_id: Optional[str] = None,
) -> Any:
    """Blocking sync wrapper around sync_session_async.

    Use from purely synchronous contexts (CLI scripts, subprocess hooks,
    orchestrator code). Creates its own event loop via asyncio.run().

    Raises:
        RuntimeError: If called from inside a running event loop. Use
            sync_session_async instead in that case.
    """
    return asyncio.run(
        sync_session_async(dataset=dataset, session_id=session_id, user_id=user_id)
    )


async def sync_sessions_async(
    dataset: str,
    session_ids: list[str],
    user_id: Optional[str] = None,
) -> Any:
    """Bulk variant of sync_session_async for multiple sessions at once.

    cognee.improve() accepts a list of session_ids in a single call,
    which is cheaper than looping when many sessions need promotion.
    """
    import cognee
    from cognee.modules.users.methods import get_default_user, get_user

    user = None
    if user_id:
        try:
            user = await get_user(UUID(user_id))
        except Exception:
            user = None
    if user is None:
        user = await get_default_user()

    return await cognee.improve(
        dataset=dataset,
        session_ids=list(session_ids),
        run_in_background=False,
        user=user,
    )


def sync_sessions(
    dataset: str,
    session_ids: list[str],
    user_id: Optional[str] = None,
) -> Any:
    """Blocking bulk wrapper around sync_sessions_async."""
    return asyncio.run(
        sync_sessions_async(dataset=dataset, session_ids=session_ids, user_id=user_id)
    )


__all__ = [
    "sync_session",
    "sync_session_async",
    "sync_sessions",
    "sync_sessions_async",
]
