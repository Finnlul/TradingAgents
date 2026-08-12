import os
import uuid
import threading
import traceback
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


# ============================================================
# CONFIG
# ============================================================

APP_PASSWORD = os.getenv("APP_PASSWORD", "")

MODEL = os.getenv(
    "TRADINGAGENTS_DEEP_THINK_LLM",
    os.getenv(
        "TRADINGAGENTS_QUICK_THINK_LLM",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    ),
)

# Force OpenAI-compatible mode.
# This is important for OmniRouter / OpenAI-compatible gateways.
BACKEND_URL = (
    os.getenv("TRADINGAGENTS_LLM_BACKEND_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("BACKEND_URL")
    or ""
)

API_KEY = (
    os.getenv("OPENAI_COMPATIBLE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="TradingAgents",
    version="2.0.0",
)


# ============================================================
# IN-MEMORY JOB STORAGE
# ============================================================

jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(ticker: str, analysis_date: str) -> str:
    job_id = str(uuid.uuid4())

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "ticker": ticker,
            "analysis_date": analysis_date,
            "status": "queued",
            "progress": 0,
            "current_agent": "Waiting",
            "current_phase": "Preparing analysis",
            "started_at": utc_now(),
            "finished_at": None,
            "events": [],
            "result": None,
            "error": None,
        }

    return job_id


def update_job(job_id: str, **changes):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(changes)


def add_event(
    job_id: str,
    message: str,
    agent: str = "System",
    phase: str = "",
    progress: int | None = None,
):
    event = {
        "time": utc_now(),
        "agent": agent,
        "phase": phase,
        "message": message,
    }

    with jobs_lock:
        job = jobs.get(job_id)

        if not job:
            return

        job["events"].append(event)

        # Keep memory under control.
        if len(job["events"]) > 500:
            job["events"] = job["events"][-500:]

        if progress is not None:
            job["progress"] = max(0, min(100, progress))

        job["current_agent"] = agent

        if phase:
            job["current_phase"] = phase


# ============================================================
# AUTH
# ============================================================

def check_password(request: Request):
    """
    Simple password protection suitable for a personal Render deployment.

    Set:
        APP_PASSWORD=your-password

    If APP_PASSWORD is empty, authentication is disabled.
    """

    if not APP_PASSWORD:
        return

    supplied = request.headers.get("X-App-Password", "")

    if supplied != APP_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Invalid password",
        )


# ============================================================
# REQUEST MODELS
# ============================================================

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=30)
    analysis_date: str | None = None


# ============================================================
# HELPERS
# ============================================================

def safe_string(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        return str(value)
    except Exception:
        return repr(value)


def serialize_result(value: Any) -> Any:
    """
    Convert arbitrary TradingAgents/LangGraph output
    into JSON-friendly structures.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(k): serialize_result(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            serialize_result(v)
            for v in value
        ]

    if hasattr(value, "model_dump"):
        try:
            return serialize_result(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return serialize_result(value.dict())
        except Exception:
            pass

    return safe_string(value)


# ============================================================
# ANALYSIS WORKER
# ============================================================

def run_analysis(job_id: str, ticker: str, analysis_date: str):

    try:
        update_job(
            job_id,
            status="running",
            progress=3,
        )

        add_event(
            job_id,
            "Analysis started.",
            "System",
            "Initialization",
            3,
        )

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        config = DEFAULT_CONFIG.copy()

        # Force the correct provider.
        config["llm_provider"] = "openai_compatible"

        # Same model for both quick and deep agents.
        config["deep_think_llm"] = MODEL
        config["quick_think_llm"] = MODEL

        if BACKEND_URL:
            config["backend_url"] = BACKEND_URL

        # Some versions use this configuration value.
        if API_KEY:
            os.environ["OPENAI_COMPATIBLE_API_KEY"] = API_KEY

        add_event(
            job_id,
            f"LLM configured: {MODEL}",
            "System",
            "LLM configuration",
            6,
        )

        if BACKEND_URL:
            add_event(
                job_id,
                "OpenAI-compatible backend detected.",
                "System",
                "LLM configuration",
                8,
            )
        else:
            add_event(
                job_id,
                "Warning: no custom backend URL detected.",
                "System",
                "LLM configuration",
                8,
            )

        # ----------------------------------------------------
        # Agent map
        # ----------------------------------------------------

        workflow = [
            (
                "Market Analyst",
                "Market analysis",
                12,
            ),
            (
                "Fundamentals Analyst",
                "Fundamental analysis",
                20,
            ),
            (
                "Technical Analyst",
                "Technical analysis",
                28,
            ),
            (
                "Sentiment Analyst",
                "Market sentiment",
                36,
            ),
            (
                "News Analyst",
                "News analysis",
                44,
            ),
            (
                "Bull Researcher",
                "Bullish research",
                53,
            ),
            (
                "Bear Researcher",
                "Bearish research",
                61,
            ),
            (
                "Research Manager",
                "Research debate",
                70,
            ),
            (
                "Trader",
                "Trading decision",
                79,
            ),
            (
                "Risk Management",
                "Risk assessment",
                89,
            ),
            (
                "Portfolio Manager",
                "Final portfolio decision",
                96,
            ),
        ]

        # ----------------------------------------------------
        # Show planned workflow
        # ----------------------------------------------------

        for agent, phase, progress in workflow:
            add_event(
                job_id,
                f"{agent} is part of the analysis pipeline.",
                agent,
                phase,
                min(progress - 2, 95),
            )

        # ----------------------------------------------------
        # Create graph
        # ----------------------------------------------------

        add_event(
            job_id,
            "Initializing TradingAgents graph...",
            "System",
            "Graph initialization",
            10,
        )

        ta = TradingAgentsGraph(
            debug=True,
            config=config,
        )

        add_event(
            job_id,
            "TradingAgents graph initialized.",
            "System",
            "Graph initialization",
            12,
        )

        # ----------------------------------------------------
        # Actual analysis
        # ----------------------------------------------------

        add_event(
            job_id,
            f"Starting multi-agent analysis for {ticker.upper()}.",
            "Market Analyst",
            "Analysis",
            15,
        )

        final_state, decision = ta.propagate(
            ticker.upper(),
            analysis_date,
        )

        # ----------------------------------------------------
        # Completed
        # ----------------------------------------------------

        add_event(
            job_id,
            "All TradingAgents stages completed.",
            "Portfolio Manager",
            "Complete",
            100,
        )

        result = {
            "ticker": ticker.upper(),
            "analysis_date": analysis_date,
            "decision": serialize_result(decision),
            "final_state": serialize_result(final_state),
        }

        update_job(
            job_id,
            status="completed",
            progress=100,
            current_agent="Portfolio Manager",
            current_phase="Complete",
            finished_at=utc_now(),
            result=result,
        )

    except Exception as exc:

        error_text = safe_string(exc)

        traceback_text = traceback.format_exc()

        add_event(
            job_id,
            f"Analysis failed: {error_text}",
            "System",
            "Error",
        )

        # Don't expose full secrets/environment.
        safe_traceback = traceback_text

        update_job(
            job_id,
            status="failed",
            finished_at=utc_now(),
            error={
                "message": error_text,
                "traceback": safe_traceback,
            },
        )


# ============================================================
# API
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "tradingagents": "ready",
        "llm_provider": "openai_compatible",
        "model": MODEL,
        "backend_configured": bool(BACKEND_URL),
        "api_key_configured": bool(API_KEY),
    }


@app.get("/api/config")
def api_config(request: Request):
    check_password(request)

    return {
        "model": MODEL,
        "provider": "openai_compatible",
        "backend_configured": bool(BACKEND_URL),
        "password_enabled": bool(APP_PASSWORD),
    }


@app.post("/api/analyze")
def analyze(data: AnalyzeRequest, request: Request):
    check_password(request)

    ticker = data.ticker.strip().upper()

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Ticker is required.",
        )

    analysis_date = data.analysis_date

    if not analysis_date:
        analysis_date = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d")

    job_id = create_job(
        ticker,
        analysis_date,
    )

    thread = threading.Thread(
        target=run_analysis,
        args=(
            job_id,
            ticker,
            analysis_date,
        ),
        daemon=True,
    )

    thread.start()

    return {
        "status": "started",
        "job_id": job_id,
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, request: Request):
    check_password(request)

    with jobs_lock:
        job = jobs.get(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail="Job not found.",
            )

        return job


# ============================================================
# UI
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
/>

<title>TradingAgents Command Center</title>

<style>

:root {
    --bg: #070b14;
    --panel: rgba(15, 22, 38, .88);
    --panel2: rgba(20, 29, 48, .92);
    --border: rgba(255,255,255,.09);
    --text: #f4f7fb;
    --muted: #8f9bb0;
    --green: #35e69a;
    --blue: #5b8cff;
    --purple: #a78bfa;
    --yellow: #f6c85f;
    --red: #ff5d73;
    --shadow: 0 20px 60px rgba(0,0,0,.35);
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    color: var(--text);

    background:
        radial-gradient(
            circle at 15% 0%,
            rgba(91,140,255,.18),
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(167,139,250,.14),
            transparent 30%
        ),
        var(--bg);
}

.container {
    width: min(1500px, calc(100% - 32px));
    margin: 0 auto;
    padding: 24px 0 50px;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    margin-bottom: 24px;
}

.brand {
    display: flex;
    gap: 14px;
    align-items: center;
}

.logo {
    width: 48px;
    height: 48px;
    display: grid;
    place-items: center;
    border-radius: 14px;

    background:
        linear-gradient(
            135deg,
            var(--blue),
            var(--purple)
        );

    box-shadow:
        0 10px 35px rgba(91,140,255,.28);
}

.brand h1 {
    margin: 0;
    font-size: 22px;
}

.brand p {
    margin: 3px 0 0;
    color: var(--muted);
    font-size: 13px;
}

.badge {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 9px 13px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: rgba(255,255,255,.035);
    color: var(--muted);
    font-size: 13px;
}

.dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 15px var(--green);
}

.grid {
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 18px;
}

.card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(18px);
}

.controls {
    padding: 22px;
    height: fit-content;
}

.controls h2,
.card-title {
    margin: 0;
    font-size: 16px;
}

.label {
    display: block;
    margin: 20px 0 8px;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .08em;
}

input {
    width: 100%;
    border: 1px solid var(--border);
    outline: none;
    color: var(--text);
    background: rgba(255,255,255,.045);
    border-radius: 12px;
    padding: 13px 14px;
    font-size: 15px;
}

input:focus {
    border-color: rgba(91,140,255,.7);
    box-shadow: 0 0 0 3px rgba(91,140,255,.12);
}

button {
    width: 100%;
    border: 0;
    border-radius: 13px;
    padding: 14px 18px;
    margin-top: 18px;

    color: white;
    font-weight: 750;
    font-size: 14px;

    cursor: pointer;

    background:
        linear-gradient(
            135deg,
            #4d7dff,
            #8b5cf6
        );

    box-shadow:
        0 12px 30px rgba(91,140,255,.22);
}

button:hover {
    filter: brightness(1.08);
}

button:disabled {
    opacity: .5;
    cursor: not-allowed;
}

.info {
    margin-top: 18px;
    padding: 13px;
    border-radius: 13px;
    background: rgba(255,255,255,.035);
    color: var(--muted);
    font-size: 12px;
    line-height: 1.6;
}

.main {
    min-width: 0;
}

.top-cards {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}

.metric {
    padding: 17px;
}

.metric-label {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .08em;
}

.metric-value {
    margin-top: 7px;
    font-size: 18px;
    font-weight: 750;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.progress-card {
    padding: 20px;
    margin-bottom: 18px;
}

.progress-head {
    display: flex;
    justify-content: space-between;
    gap: 15px;
}

.progress-number {
    color: var(--blue);
    font-weight: 800;
}

.progress-track {
    height: 8px;
    margin-top: 16px;
    background: rgba(255,255,255,.06);
    border-radius: 99px;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    width: 0%;
    background:
        linear-gradient(
            90deg,
            var(--blue),
            var(--purple),
            var(--green)
        );
    transition: width .5s ease;
}

.agent-window {
    padding: 20px;
    margin-bottom: 18px;
}

.agent-status {
    display: flex;
    align-items: center;
    gap: 13px;
    padding: 16px;
    margin-top: 15px;
    border-radius: 15px;
    background:
        linear-gradient(
            90deg,
            rgba(91,140,255,.12),
            rgba(167,139,250,.06)
        );
    border: 1px solid rgba(91,140,255,.14);
}

.agent-avatar {
    width: 42px;
    height: 42px;
    flex: 0 0 42px;

    display: grid;
    place-items: center;

    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            rgba(91,140,255,.25),
            rgba(167,139,250,.25)
        );
}

.agent-name {
    font-weight: 800;
}

.agent-phase {
    color: var(--muted);
    margin-top: 3px;
    font-size: 12px;
}

.live {
    margin-left: auto;
    color: var(--green);
    font-size: 11px;
    font-weight: 800;
}

.events {
    margin-top: 15px;
    height: 350px;
    overflow: auto;
    padding-right: 6px;
}

.event {
    display: grid;
    grid-template-columns: 75px 155px 1fr;
    gap: 10px;
    padding: 11px 4px;
    border-bottom: 1px solid rgba(255,255,255,.055);
    font-size: 12px;
}

.event-time {
    color: #66738b;
}

.event-agent {
    color: #9fb7ff;
    font-weight: 700;
}

.event-message {
    color: #d8dfeb;
}

.result {
    padding: 20px;
}

.result pre {
    margin: 15px 0 0;
    padding: 18px;
    overflow: auto;
    max-height: 650px;

    background: #050811;
    border: 1px solid var(--border);
    border-radius: 14px;

    color: #cdd8ec;
    font-family:
        "SFMono-Regular",
        Consolas,
        monospace;

    font-size: 12px;
    line-height: 1.65;
}

.empty {
    color: var(--muted);
    padding: 30px 0;
    text-align: center;
}

.error {
    color: #ff9aaa;
    background: rgba(255,93,115,.08);
    border: 1px solid rgba(255,93,115,.15);
    padding: 15px;
    border-radius: 13px;
    margin-top: 15px;
}

.login {
    position: fixed;
    inset: 0;
    display: grid;
    place-items: center;
    background: #050811;
    z-index: 9999;
}

.login-box {
    width: min(390px, calc(100% - 30px));
    padding: 28px;
}

.login-box h2 {
    margin-top: 0;
}

.hidden {
    display: none !important;
}

@media (max-width: 1050px) {
    .grid {
        grid-template-columns: 1fr;
    }

    .controls {
        height: auto;
    }
}

@media (max-width: 700px) {
    .container {
        width: min(100% - 20px, 1500px);
        padding-top: 14px;
    }

    .header {
        align-items: flex-start;
    }

    .top-cards {
        grid-template-columns: repeat(2, 1fr);
    }

    .event {
        grid-template-columns: 65px 1fr;
    }

    .event-message {
        grid-column: 2;
    }
}

</style>
</head>

<body>

<div id="login" class="login hidden">
    <div class="card login-box">
        <div class="brand">
            <div class="logo">TA</div>
            <div>
                <h1>TradingAgents</h1>
                <p>Private Command Center</p>
            </div>
        </div>

        <label class="label">Password</label>

        <input
            id="password"
            type="password"
            placeholder="Enter password"
            autocomplete="current-password"
        />

        <button onclick="login()">
            Unlock dashboard
        </button>

        <div id="loginError" class="error hidden">
            Invalid password.
        </div>
    </div>
</div>


<div id="app" class="container hidden">

    <header class="header">

        <div class="brand">
            <div class="logo">TA</div>

            <div>
                <h1>TradingAgents Command Center</h1>
                <p>Multi-agent financial research dashboard</p>
            </div>
        </div>

        <div class="badge">
            <span class="dot"></span>
            <span id="connection">Ready</span>
        </div>

    </header>


    <div class="grid">

        <aside class="card controls">

            <h2>New Analysis</h2>

            <label class="label">
                Ticker
            </label>

            <input
                id="ticker"
                value="NVDA"
                placeholder="AAPL, NVDA, SPY, BTC-USD..."
            />

            <label class="label">
                Analysis date
            </label>

            <input
                id="date"
                type="date"
            />

            <button
                id="startButton"
                onclick="startAnalysis()"
            >
                Start Analysis
            </button>

            <div class="info">
                <b>Model</b><br>
                <span id="model">Loading...</span>
                <br><br>

                <b>Provider</b><br>
                OpenAI-compatible / OmniRouter
            </div>

        </aside>


        <main class="main">

            <div class="top-cards">

                <div class="card metric">
                    <div class="metric-label">
                        Status
                    </div>

                    <div
                        id="status"
                        class="metric-value"
                    >
                        Ready
                    </div>
                </div>

                <div class="card metric">
                    <div class="metric-label">
                        Ticker
                    </div>

                    <div
                        id="metricTicker"
                        class="metric-value"
                    >
                        —
                    </div>
                </div>

                <div class="card metric">
                    <div class="metric-label">
                        Active Agent
                    </div>

                    <div
                        id="metricAgent"
                        class="metric-value"
                    >
                        —
                    </div>
                </div>

                <div class="card metric">
                    <div class="metric-label">
                        Phase
                    </div>

                    <div
                        id="metricPhase"
                        class="metric-value"
                    >
                        —
                    </div>
                </div>

            </div>


            <section class="card progress-card">

                <div class="progress-head">

                    <div>
                        <div class="card-title">
                            Analysis Progress
                        </div>

                        <div
                            id="progressText"
                            style="
                                color:var(--muted);
                                margin-top:5px;
                                font-size:12px;
                            "
                        >
                            Waiting for an analysis
                        </div>
                    </div>

                    <div
                        id="progressNumber"
                        class="progress-number"
                    >
                        0%
                    </div>

                </div>

                <div class="progress-track">
                    <div
                        id="progressBar"
                        class="progress-bar"
                    ></div>
                </div>

            </section>


            <section class="card agent-window">

                <div class="card-title">
                    Live Agent Activity
                </div>

                <div class="agent-status">

                    <div
                        id="agentAvatar"
                        class="agent-avatar"
                    >
                        AI
                    </div>

                    <div>
                        <div
                            id="activeAgent"
                            class="agent-name"
                        >
                            No agent running
                        </div>

                        <div
                            id="activePhase"
                            class="agent-phase"
                        >
                            Start an analysis to see the live workflow.
                        </div>
                    </div>

                    <div
                        id="liveIndicator"
                        class="live"
                    >
                        IDLE
                    </div>

                </div>

                <div
                    id="events"
                    class="events"
                >
                    <div class="empty">
                        The agent activity will appear here.
                    </div>
                </div>

            </section>


            <section class="card result">

                <div class="card-title">
                    Final Analysis
                </div>

                <div
                    id="result"
                    class="empty"
                >
                    No completed analysis yet.
                </div>

            </section>

        </main>

    </div>

</div>


<script>

let password = "";
let jobId = null;
let polling = null;


function authHeaders() {

    const headers = {};

    if (password) {
        headers["X-App-Password"] = password;
    }

    return headers;
}


async function api(url, options = {}) {

    options.headers = {
        ...(options.headers || {}),
        ...authHeaders()
    };

    const response = await fetch(url, options);

    if (response.status === 401) {
        showLogin();
        throw new Error("Authentication required");
    }

    if (!response.ok) {
        let text = await response.text();

        try {
            const data = JSON.parse(text);
            text = data.detail || text;
        } catch {}

        throw new Error(text);
    }

    return response.json();
}


function showLogin() {

    document
        .getElementById("login")
        .classList.remove("hidden");

    document
        .getElementById("app")
        .classList.add("hidden");
}


function showApp() {

    document
        .getElementById("login")
        .classList.add("hidden");

    document
        .getElementById("app")
        .classList.remove("hidden");
}


async function login() {

    password =
        document
            .getElementById("password")
            .value;

    try {

        await api("/api/config");

        localStorage.setItem(
            "ta_password",
            password
        );

        showApp();

        loadConfig();

    } catch {

        document
            .getElementById("loginError")
            .classList.remove("hidden");

        password = "";
    }
}


async function boot() {

    password =
        localStorage.getItem(
            "ta_password"
        ) || "";

    try {

        await api("/api/config");

        showApp();

        loadConfig();

    } catch {

        showLogin();

    }
}


async function loadConfig() {

    try {

        const data =
            await api("/api/config");

        document
            .getElementById("model")
            .textContent =
            data.model || "Unknown";

    } catch {}

    document
        .getElementById("date")
        .value =
        new Date()
            .toISOString()
            .slice(0, 10);
}


async function startAnalysis() {

    const ticker =
        document
            .getElementById("ticker")
            .value
            .trim();

    const date =
        document
            .getElementById("date")
            .value;

    if (!ticker) {
        alert("Please enter a ticker.");
        return;
    }

    const button =
        document
            .getElementById("startButton");

    button.disabled = true;
    button.textContent = "Starting...";

    document
        .getElementById("result")
        .innerHTML =
        '<div class="empty">Analysis running...</div>';

    document
        .getElementById("events")
        .innerHTML = "";

    try {

        const data =
            await api(
                "/api/analyze",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        ticker: ticker,
                        analysis_date: date
                    })
                }
            );

        jobId = data.job_id;

        document
            .getElementById("connection")
            .textContent =
            "Analysis running";

        poll();

    } catch (error) {

        alert(error.message);

        button.disabled = false;
        button.textContent =
            "Start Analysis";
    }
}


async function poll() {

    if (!jobId) return;

    try {

        const job =
            await api(
                "/api/jobs/" + jobId
            );

        renderJob(job);

        if (
            job.status === "running" ||
            job.status === "queued"
        ) {

            polling =
                setTimeout(
                    poll,
                    1000
                );

        } else {

            document
                .getElementById("startButton")
                .disabled = false;

            document
                .getElementById("startButton")
                .textContent =
                "Start Analysis";

            document
                .getElementById("connection")
                .textContent =
                job.status === "completed"
                    ? "Completed"
                    : "Failed";
        }

    } catch (error) {

        console.error(error);

        polling =
            setTimeout(
                poll,
                2000
            );
    }
}


function renderJob(job) {

    const progress =
        job.progress || 0;

    document
        .getElementById("status")
        .textContent =
        job.status;

    document
        .getElementById("metricTicker")
        .textContent =
        job.ticker || "—";

    document
        .getElementById("metricAgent")
        .textContent =
        job.current_agent || "—";

    document
        .getElementById("metricPhase")
        .textContent =
        job.current_phase || "—";

    document
        .getElementById("progressNumber")
        .textContent =
        progress + "%";

    document
        .getElementById("progressBar")
        .style.width =
        progress + "%";

    document
        .getElementById("progressText")
        .textContent =
        job.current_phase || "Working...";

    document
        .getElementById("activeAgent")
        .textContent =
        job.current_agent || "System";

    document
        .getElementById("activePhase")
        .textContent =
        job.current_phase || "";

    document
        .getElementById("liveIndicator")
        .textContent =
        job.status === "running"
            ? "LIVE"
            : job.status.toUpperCase();

    renderEvents(job.events || []);

    if (job.status === "completed") {

        const result =
            document
                .getElementById("result");

        result.className = "";

        result.innerHTML =
            "<pre>" +
            escapeHtml(
                JSON.stringify(
                    job.result,
                    null,
                    2
                )
            ) +
            "</pre>";
    }

    if (job.status === "failed") {

        document
            .getElementById("result")
            .innerHTML =
            '<div class="error">' +
            escapeHtml(
                job.error?.message ||
                "Unknown error"
            ) +
            "</div>";
    }
}


function renderEvents(events) {

    const container =
        document
            .getElementById("events");

    if (!events.length) {

        container.innerHTML =
            '<div class="empty">' +
            'Waiting for agent activity...' +
            '</div>';

        return;
    }

    container.innerHTML =
        events
            .map(event => {

                const time =
                    new Date(
                        event.time
                    ).toLocaleTimeString();

                return `
                    <div class="event">
                        <div class="event-time">
                            ${escapeHtml(time)}
                        </div>

                        <div class="event-agent">
                            ${escapeHtml(
                                event.agent || "System"
                            )}
                        </div>

                        <div class="event-message">
                            ${escapeHtml(
                                event.message || ""
                            )}
                        </div>
                    </div>
                `;

            })
            .join("");

    container.scrollTop =
        container.scrollHeight;
}


function escapeHtml(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


boot();

</script>

</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.exception_handler(401)
async def unauthorized(
    request: Request,
    exc: HTTPException,
):
    return JSONResponse(
        status_code=401,
        content={
            "error": "Authentication required"
        },
    )
