"""
FastAPI REST API — GitHub endpoints + Grok agent.
Run: uvicorn api:app --reload --port 8000
"""

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

import github_services as gh
from github_services import GitHubError, GITHUB_TOKEN
from agent_backend import (
    get_unified_agent_status,
    run_agent_chat,
    run_impact_chat,
    run_intelligence_chat,
)
from intelligence.collector import collect_repository_snapshot
from intelligence.analyzer import analyze_snapshot
from intelligence.context_export import (
    SCHEMA_PATH,
    ContextExportOptions,
    export_context_studio_bundle,
    export_context_studio_preview,
)
from dependency_graph.builder import build_dependency_graph
from dependency_graph.impact import build_change_impact

app = FastAPI(
    title="GitHub History API",
    description="REST API + Grok agent for GitHub PR & Issue history",
    version="1.1.0",
)



def _http_error(exc: GitHubError):
    raise HTTPException(status_code=exc.status_code, detail=exc.detail)


# ── Agent ────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class AgentChatRequest(BaseModel):
    messages: list[ChatMessage]
    owner: str | None = None
    repo: str | None = None
    backend: str | None = Field(
        None,
        description="ica | local | auto — default from AGENT_BACKEND env",
    )
    session_id: str | None = Field(
        None,
        description="ICA workflow session id (reuse for multi-turn chat)",
    )


class AgentChatResponse(BaseModel):
    message: str
    tool_calls: list[dict] = Field(default_factory=list)
    model: str
    provider: str | None = None
    backend: str | None = None
    session_id: str | None = None
    insight_graph: dict | None = None
    action_id: str | None = None
    quick_actions: list[dict] = Field(default_factory=list)


@app.post("/agent/chat", tags=["Agent"], response_model=AgentChatResponse)
async def agent_chat(body: AgentChatRequest):
    """Chat via ICA workflow (MCP agent) or local LLM + GitHub tools."""
    try:
        result = await run_agent_chat(
            [m.model_dump() for m in body.messages],
            owner=body.owner,
            repo=body.repo,
            backend=body.backend,
            session_id=body.session_id,
        )
        return AgentChatResponse(
            message=result["message"],
            tool_calls=result.get("tool_calls", []),
            model=result.get("model", "unknown"),
            provider=result.get("provider"),
            backend=result.get("backend"),
            session_id=result.get("session_id"),
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/agent/status", tags=["Agent"])
async def agent_status():
    return get_unified_agent_status()


# ── Repo ─────────────────────────────────────────────────────────────────────

@app.get("/repo/{owner}/{repo}", tags=["Repo"])
async def repo_info(owner: str, repo: str):
    try:
        return await gh.get_repo_info(owner, repo)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/contributors", tags=["Repo"])
async def repo_contributors(owner: str, repo: str):
    try:
        return await gh.list_repo_contributors(owner, repo)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/labels", tags=["Repo"])
async def repo_labels(owner: str, repo: str):
    try:
        return await gh.list_repo_labels(owner, repo)
    except GitHubError as e:
        _http_error(e)


# ── Pull Requests ─────────────────────────────────────────────────────────────

@app.get("/repo/{owner}/{repo}/pulls", tags=["Pull Requests"])
async def list_prs(
    owner: str,
    repo: str,
    state: str = Query("all", enum=["open", "closed", "all"]),
    sort: str = Query("created", enum=["created", "updated", "popularity", "long-running"]),
    direction: str = Query("desc", enum=["asc", "desc"]),
):
    try:
        return await gh.list_pull_requests(owner, repo, state=state, sort=sort, direction=direction)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/pulls/{pr_number}", tags=["Pull Requests"])
async def pr_detail(owner: str, repo: str, pr_number: int):
    try:
        return await gh.get_pull_request_detail(owner, repo, pr_number)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/pulls/{pr_number}/reviews", tags=["Pull Requests"])
async def pr_reviews(owner: str, repo: str, pr_number: int):
    try:
        return await gh.list_pr_reviews(owner, repo, pr_number)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/pulls/{pr_number}/comments", tags=["Pull Requests"])
async def pr_comments(owner: str, repo: str, pr_number: int):
    try:
        return await gh.list_pr_comments(owner, repo, pr_number)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/pulls/{pr_number}/commits", tags=["Pull Requests"])
async def pr_commits(owner: str, repo: str, pr_number: int):
    try:
        return await gh.list_pr_commits(owner, repo, pr_number)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/pulls/{pr_number}/files", tags=["Pull Requests"])
async def pr_files(owner: str, repo: str, pr_number: int):
    try:
        return await gh.list_pr_files(owner, repo, pr_number)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/pulls/{pr_number}/full", tags=["Pull Requests"])
async def pr_full(owner: str, repo: str, pr_number: int):
    """Full PR bundle: detail, reviews, inline comments, discussion, commits, file diffs."""
    try:
        return await gh.get_pr_full_detail(owner, repo, pr_number)
    except GitHubError as e:
        _http_error(e)


# ── Issues ────────────────────────────────────────────────────────────────────

@app.get("/repo/{owner}/{repo}/issues", tags=["Issues"])
async def list_issues(
    owner: str,
    repo: str,
    state: str = Query("all", enum=["open", "closed", "all"]),
    sort: str = Query("created", enum=["created", "updated", "comments"]),
    direction: str = Query("desc", enum=["asc", "desc"]),
    labels: str = Query("", description="Comma-separated label names"),
):
    try:
        return await gh.list_issues(owner, repo, state=state, sort=sort, direction=direction, labels=labels)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/issues/{issue_number}", tags=["Issues"])
async def issue_detail(owner: str, repo: str, issue_number: int):
    try:
        return await gh.get_issue_detail(owner, repo, issue_number)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/issues/{issue_number}/comments", tags=["Issues"])
async def issue_comments(owner: str, repo: str, issue_number: int):
    try:
        return await gh.list_issue_comments(owner, repo, issue_number)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/issues/{issue_number}/events", tags=["Issues"])
async def issue_events(owner: str, repo: str, issue_number: int):
    try:
        return await gh.list_issue_events(owner, repo, issue_number)
    except GitHubError as e:
        _http_error(e)


@app.get("/repo/{owner}/{repo}/issues/{issue_number}/full", tags=["Issues"])
async def issue_full(owner: str, repo: str, issue_number: int):
    """Full issue bundle: detail, comments, timeline events."""
    try:
        return await gh.get_issue_full_detail(owner, repo, issue_number)
    except GitHubError as e:
        _http_error(e)


# ── Search ────────────────────────────────────────────────────────────────────

@app.get("/repo/{owner}/{repo}/search", tags=["Search"])
async def search(
    owner: str,
    repo: str,
    q: str = Query(..., description="Search keywords"),
    kind: str = Query("both", enum=["issue", "pr", "both"]),
):
    try:
        return await gh.search_issues_and_prs(owner, repo, q, kind=kind)
    except GitHubError as e:
        _http_error(e)


# ── Dependency graph ──────────────────────────────────────────────────────────

@app.get("/repo/{owner}/{repo}/dependencies/graph", tags=["Dependencies"])
async def repository_dependency_graph(
    owner: str,
    repo: str,
    ref: str = Query("", description="Branch, tag, or commit SHA"),
    max_files: int = Query(400, ge=50, le=800),
    include_packages: bool = Query(True),
    focus_path: str = Query("", description="Optional: subgraph from this file path"),
    max_depth: int = Query(0, ge=0, le=8, description="BFS depth when focus_path set; 0 = full graph"),
):
    """
    Code import dependency graph as JSON (nodes, edges, file_tree, clusters).
    Use with the /dependencies Next.js page or any graph library (React Flow, D3, etc.).
    """
    try:
        return await build_dependency_graph(
            owner,
            repo,
            ref=ref or None,
            max_files=max_files,
            include_packages=include_packages,
            focus_path=focus_path or None,
            max_depth=max_depth if focus_path and max_depth > 0 else None,
        )
    except GitHubError as e:
        _http_error(e)
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
        raise HTTPException(
            status_code=504,
            detail=(
                "GitHub API timed out while fetching file contents. "
                "Try a smaller max_files (e.g. 100–150) or set GITHUB_BATCH_CONCURRENCY=3."
            ),
        ) from e


@app.get("/repo/{owner}/{repo}/dependencies/impact", tags=["Dependencies"])
async def repository_change_impact(
    owner: str,
    repo: str,
    file_path: str = Query(..., description="Repository-relative file path to analyze"),
    ref: str = Query(""),
    max_files: int = Query(400, ge=50, le=800),
    include_packages: bool = Query(True),
    max_depth_dependents: int = Query(4, ge=1, le=8),
    max_depth_dependencies: int = Query(3, ge=1, le=8),
    pr_sample_size: int = Query(40, ge=5, le=80),
):
    """
    Blast radius for changing a file: reverse dependents, forward imports, related PRs.
    Use with the /dependencies UI or MCP tool get_change_impact.
    """
    try:
        return await build_change_impact(
            owner,
            repo,
            file_path,
            ref=ref or None,
            max_files=max_files,
            include_packages=include_packages,
            max_depth_dependents=max_depth_dependents,
            max_depth_dependencies=max_depth_dependencies,
            pr_sample_size=pr_sample_size,
        )
    except GitHubError as e:
        _http_error(e)
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.PoolTimeout) as e:
        raise HTTPException(
            status_code=504,
            detail="GitHub API timed out during blast radius analysis. Try a smaller max_files.",
        ) from e


class ImpactChatRequest(BaseModel):
    messages: list[ChatMessage]
    impact: dict = Field(..., description="ChangeImpact JSON from /dependencies/impact")
    backend: str | None = None
    session_id: str | None = None
    max_context_files: int = Field(8, ge=1, le=12)
    action_id: str | None = Field(
        None,
        description="Quick action id (blast_map, refactor_risk, test_matrix, …)",
    )


@app.post(
    "/repo/{owner}/{repo}/dependencies/impact/chat",
    tags=["Dependencies"],
    response_model=AgentChatResponse,
)
async def impact_context_chat(owner: str, repo: str, body: ImpactChatRequest):
    """
    Chat about a file using blast-radius results + source from target and related paths.
    """
    try:
        result = await run_impact_chat(
            [m.model_dump() for m in body.messages],
            body.impact,
            owner,
            repo,
            backend=body.backend,
            session_id=body.session_id,
            max_context_files=body.max_context_files,
            action_id=body.action_id,
        )
        return AgentChatResponse(
            message=result["message"],
            tool_calls=result.get("tool_calls", []),
            model=result.get("model", "unknown"),
            provider=result.get("provider"),
            backend=result.get("backend"),
            session_id=result.get("session_id"),
            insight_graph=result.get("insight_graph"),
            action_id=result.get("action_id"),
            quick_actions=result.get("quick_actions") or [],
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── Repository Intelligence ───────────────────────────────────────────────────

class IntelligenceAskRequest(BaseModel):
    messages: list[ChatMessage]
    insights: dict | None = None
    backend: str | None = None
    session_id: str | None = None


@app.get("/repo/{owner}/{repo}/intelligence/insights", tags=["Intelligence"])
async def repository_insights(owner: str, repo: str):
    """
    Collect repository snapshot and return structured intelligence insights.
    May take 30–90s on large repos (samples PR file changes).
    """
    try:
        snapshot = await collect_repository_snapshot(owner, repo)
        return analyze_snapshot(snapshot)
    except GitHubError as e:
        _http_error(e)


@app.post("/repo/{owner}/{repo}/intelligence/ask", tags=["Intelligence"], response_model=AgentChatResponse)
async def intelligence_ask(owner: str, repo: str, body: IntelligenceAskRequest):
    """AI analyst with pre-computed insights + GitHub tools."""
    try:
        insights = body.insights
        if not insights:
            snapshot = await collect_repository_snapshot(owner, repo)
            insights = analyze_snapshot(snapshot)
        result = await run_intelligence_chat(
            [m.model_dump() for m in body.messages],
            insights,
            owner,
            repo,
            backend=body.backend,
            session_id=body.session_id,
        )
        return AgentChatResponse(
            message=result["message"],
            tool_calls=result.get("tool_calls", []),
            model=result.get("model", "unknown"),
            provider=result.get("provider"),
            backend=result.get("backend"),
            session_id=result.get("session_id"),
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


class ContextStudioExportRequest(BaseModel):
    ref: str = ""
    include_graph: bool = True
    graph_max_files: int = Field(300, ge=50, le=600)
    llm_enhance_docs: bool = False
    pr_file_sample: int = Field(25, ge=5, le=50)
    insights: dict | None = None


@app.get("/context-studio/schema", tags=["Context Studio"])
async def context_studio_schema_download():
    """Download the global JSON-LD ontology (import once per ICA team)."""
    if not SCHEMA_PATH.is_file():
        raise HTTPException(status_code=404, detail="Schema file not found")
    return FileResponse(
        SCHEMA_PATH,
        media_type="application/ld+json",
        filename="software-repository-archaeology-schema.jsonld",
    )


@app.post("/repo/{owner}/{repo}/context-studio/preview", tags=["Context Studio"])
async def context_studio_preview(owner: str, repo: str, body: ContextStudioExportRequest | None = None):
    """
    Build export artifacts and return manifest + summary JSON (no ZIP).
    Pass `insights` from /intelligence/insights to skip full re-collection.
    """
    opts = body or ContextStudioExportRequest()
    try:
        return await export_context_studio_preview(
            owner,
            repo,
            ContextExportOptions(
                ref=opts.ref,
                include_graph=opts.include_graph,
                graph_max_files=opts.graph_max_files,
                llm_enhance_docs=opts.llm_enhance_docs,
                pr_file_sample=opts.pr_file_sample,
                cached_insights=opts.insights,
                build_zip=False,
            ),
        )
    except GitHubError as e:
        _http_error(e)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/repo/{owner}/{repo}/context-studio/export", tags=["Context Studio"])
async def context_studio_export(owner: str, repo: str, body: ContextStudioExportRequest | None = None):
    """
    Build ICA Context Studio bundle: JSON-LD schema + per-repo instances (lab syntax),
    facts JSON, markdown docs, manifest. Returns application/zip.
    May take 60–120s on large repos.
    """
    opts = body or ContextStudioExportRequest()
    try:
        result = await export_context_studio_bundle(
            owner,
            repo,
            ContextExportOptions(
                ref=opts.ref,
                include_graph=opts.include_graph,
                graph_max_files=opts.graph_max_files,
                llm_enhance_docs=opts.llm_enhance_docs,
                pr_file_sample=opts.pr_file_sample,
                cached_insights=opts.insights,
            ),
        )
    except GitHubError as e:
        _http_error(e)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    safe = f"{owner}-{repo}".replace("/", "-")
    filename = f"{safe}-context-studio.zip"
    return Response(
        content=result.zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-BlueWings-Validation-Errors": str(len(result.validation_errors)),
        },
    )


@app.get("/repo/{owner}/{repo}/context-studio/export", tags=["Context Studio"])
async def context_studio_export_get(
    owner: str,
    repo: str,
    ref: str = "",
    include_graph: bool = True,
    graph_max_files: int = Query(300, ge=50, le=600),
    llm_enhance_docs: bool = False,
):
    """GET alias for Context Studio ZIP export (same as POST with query params)."""
    body = ContextStudioExportRequest(
        ref=ref,
        include_graph=include_graph,
        graph_max_files=graph_max_files,
        llm_enhance_docs=llm_enhance_docs,
    )
    return await context_studio_export(owner, repo, body)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health():
    status = get_unified_agent_status()
    return {
        "status": "ok",
        "token_set": bool(GITHUB_TOKEN),
        "llm_configured": status["llm_configured"],
        "agent_backend": status.get("default_backend"),
        "ica_configured": status.get("ica_configured"),
        "llm_provider": status.get("provider"),
    }
