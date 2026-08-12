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

# ------------------------------------------------------------
# OMNIROUTER
# ------------------------------------------------------------
#
# Render Environment Variables:
#
# OMNIROUTER_BASE_URL=DEINE_OMNIROUTER_URL/v1
# OMNIROUTER_API_KEY=DEIN_OMNIROUTER_KEY
# TRADINGAGENTS_MODEL=DEIN_OPENROUTER_MODELL
#
# Beispiel:
#
# TRADINGAGENTS_MODEL=openai/gpt-4o
#
# KEIN nvidia/... Default mehr.
# ------------------------------------------------------------

MODEL = os.getenv(
    "TRADINGAGENTS_MODEL",
    "",
)

BACKEND_URL = (
    os.getenv("OMNIROUTER_BASE_URL")
    or os.getenv("TRADINGAGENTS_LLM_BACKEND_URL")
    or os.getenv("OPENAI_BASE_URL")
    or ""
)

API_KEY = (
    os.getenv("OMNIROUTER_API_KEY")
    or os.getenv("OPENAI_COMPATIBLE_API_KEY")
    or ""
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="TradingAgents OmniRouter API",
    version="3.0.0",
)


# ============================================================
# JOB STORAGE
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

        if len(job["events"]) > 500:
            job["events"] = job["events"][-500:]

        if progress is not None:
            job["progress"] = max(
                0,
                min(100, progress),
            )

        job["current_agent"] = agent

        if phase:
            job["current_phase"] = phase


# ============================================================
# AUTH
# ============================================================

def check_password(request: Request):
    if not APP_PASSWORD:
        return

    supplied = request.headers.get(
        "X-App-Password",
        "",
    )

    if supplied != APP_PASSWORD:
        raise HTTPException(
            status_code=401,
            detail="Invalid password",
        )


# ============================================================
# REQUEST
# ============================================================

class AnalyzeRequest(BaseModel):
    ticker: str = Field(
        ...,
        min_length=1,
        max_length=30,
    )

    analysis_date: str | None = None


# ============================================================
# SERIALIZATION
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

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, dict):
        return {
            str(k): serialize_result(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            serialize_result(v)
            for v in value
        ]

    if hasattr(value, "model_dump"):
        try:
            return serialize_result(
                value.model_dump()
            )
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return serialize_result(
                value.dict()
            )
        except Exception:
            pass

    return safe_string(value)


# ============================================================
# CONFIG VALIDATION
# ============================================================

def validate_llm_config():

    if not MODEL:
        raise RuntimeError(
            "TRADINGAGENTS_MODEL is missing. "
            "Set the exact OpenRouter model ID in Render."
        )

    if not BACKEND_URL:
        raise RuntimeError(
            "OMNIROUTER_BASE_URL is missing. "
            "Set the OmniRouter OpenAI-compatible /v1 URL in Render."
        )

    if not API_KEY:
        raise RuntimeError(
            "OMNIROUTER_API_KEY is missing. "
            "Set your OmniRouter API key in Render."
        )


# ============================================================
# ANALYSIS WORKER
# ============================================================

def run_analysis(
    job_id: str,
    ticker: str,
    analysis_date: str,
):

    try:

        update_job(
            job_id,
            status="running",
            progress=2,
        )

        add_event(
            job_id,
            "Analysis started.",
            "System",
            "Initialization",
            2,
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        validate_llm_config()

        add_event(
            job_id,
            "OmniRouter configuration validated.",
            "System",
            "LLM configuration",
            5,
        )

        # ----------------------------------------------------
        # Environment
        # ----------------------------------------------------
        #
        # TradingAgents openai_compatible uses:
        #
        # OPENAI_COMPATIBLE_API_KEY
        #
        # backend_url comes from config.
        # ----------------------------------------------------

        os.environ[
            "OPENAI_COMPATIBLE_API_KEY"
        ] = API_KEY

        # Also expose the key through OPENAI_API_KEY.
        # Some OpenAI-compatible client versions
        # look for this variable.
        os.environ[
            "OPENAI_API_KEY"
        ] = API_KEY

        # ----------------------------------------------------
        # TradingAgents configuration
        # ----------------------------------------------------

        config = DEFAULT_CONFIG.copy()

        config["llm_provider"] = (
            "openai_compatible"
        )

        config["deep_think_llm"] = MODEL
        config["quick_think_llm"] = MODEL

        config["backend_url"] = BACKEND_URL

        add_event(
            job_id,
            f"Model: {MODEL}",
            "System",
            "LLM configuration",
            7,
        )

        add_event(
            job_id,
            f"Gateway: {BACKEND_URL}",
            "System",
            "LLM configuration",
            9,
        )

        add_event(
            job_id,
            "Provider: openai_compatible",
            "System",
            "LLM configuration",
            10,
        )

        # ----------------------------------------------------
        # Workflow
        # ----------------------------------------------------

        workflow = [
            (
                "Market Analyst",
                "Market analysis",
                15,
            ),
            (
                "Fundamentals Analyst",
                "Fundamental analysis",
                23,
            ),
            (
                "Technical Analyst",
                "Technical analysis",
                31,
            ),
            (
                "Sentiment Analyst",
                "Market sentiment",
                39,
            ),
            (
                "News Analyst",
                "News analysis",
                47,
            ),
            (
                "Bull Researcher",
                "Bullish research",
                55,
            ),
            (
                "Bear Researcher",
                "Bearish research",
                63,
            ),
            (
                "Research Manager",
                "Research debate",
                71,
            ),
            (
                "Trader",
                "Trading decision",
                80,
            ),
            (
                "Risk Management",
                "Risk assessment",
                90,
            ),
            (
                "Portfolio Manager",
                "Final portfolio decision",
                97,
            ),
        ]

        for agent, phase, progress in workflow:

            add_event(
                job_id,
                f"{agent} added to workflow.",
                agent,
                phase,
                min(progress - 2, 95),
            )

        # ----------------------------------------------------
        # Graph
        # ----------------------------------------------------

        add_event(
            job_id,
            "Initializing TradingAgents graph...",
            "System",
            "Graph initialization",
            11,
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
            13,
        )

        # ----------------------------------------------------
        # Actual analysis
        # ----------------------------------------------------

        ticker = ticker.upper().strip()

        add_event(
            job_id,
            f"Starting analysis for {ticker}.",
            "Market Analyst",
            "Analysis",
            15,
        )

        final_state, decision = ta.propagate(
            ticker,
            analysis_date,
        )

        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {
            "ticker": ticker,
            "analysis_date": analysis_date,
            "decision": serialize_result(
                decision
            ),
            "final_state": serialize_result(
                final_state
            ),
        }

        add_event(
            job_id,
            "All TradingAgents stages completed.",
            "Portfolio Manager",
            "Complete",
            100,
        )

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

        update_job(
            job_id,
            status="failed",
            finished_at=utc_now(),
            error={
                "message": error_text,
                "traceback": traceback_text,
            },
        )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "TradingAgents API",
        "provider": "openai_compatible",
        "model": MODEL or None,
        "backend_configured": bool(
            BACKEND_URL
        ),
        "api_key_configured": bool(
            API_KEY
        ),
    }


# ============================================================
# CONFIG API
# ============================================================

@app.get("/api/config")
def api_config(request: Request):

    check_password(request)

    return {
        "model": MODEL or "Not configured",
        "provider": "openai_compatible",
        "backend_configured": bool(
            BACKEND_URL
        ),
        "api_key_configured": bool(
            API_KEY
        ),
        "password_enabled": bool(
            APP_PASSWORD
        ),
    }


# ============================================================
# START ANALYSIS
# ============================================================

@app.post("/api/analyze")
def analyze(
    data: AnalyzeRequest,
    request: Request,
):

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


# ============================================================
# JOB STATUS
# ============================================================

@app.get("/api/jobs/{job_id}")
def job_status(
    job_id: str,
    request: Request,
):

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
# SIMPLE WEB UI
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>TradingAgents Command Center</title>

<style>

:root {
    --bg:#070b14;
    --panel:#10182a;
    --border:#26324a;
    --text:#f5f7fb;
    --muted:#8995aa;
    --blue:#5b8cff;
    --purple:#9b72ff;
    --green:#35e69a;
    --red:#ff6075;
}

* {
    box-sizing:border-box;
}

body {
    margin:0;
    background:
        radial-gradient(
            circle at top left,
            #172650,
            transparent 40%
        ),
        var(--bg);
    color:var(--text);
    font-family:
        Inter,
        system-ui,
        -apple-system,
        sans-serif;
}

.container {
    width:min(1400px, calc(100% - 30px));
    margin:auto;
    padding:25px 0 50px;
}

.header {
    display:flex;
    justify-content:space-between;
    align-items:center;
    margin-bottom:20px;
}

.logo {
    width:45px;
    height:45px;
    border-radius:12px;
    display:grid;
    place-items:center;
    font-weight:900;
    background:
        linear-gradient(
            135deg,
            var(--blue),
            var(--purple)
        );
}

.brand {
    display:flex;
    align-items:center;
    gap:12px;
}

h1 {
    margin:0;
    font-size:22px;
}

.subtitle {
    color:var(--muted);
    font-size:12px;
    margin-top:3px;
}

.grid {
    display:grid;
    grid-template-columns:330px 1fr;
    gap:18px;
}

.card {
    background:rgba(16,24,42,.9);
    border:1px solid var(--border);
    border-radius:18px;
    padding:20px;
}

label {
    display:block;
    color:var(--muted);
    font-size:12px;
    margin:18px 0 7px;
}

input {
    width:100%;
    padding:13px;
    border-radius:10px;
    border:1px solid var(--border);
    background:#080e1c;
    color:white;
    outline:none;
}

button {
    width:100%;
    margin-top:18px;
    padding:14px;
    border:0;
    border-radius:10px;
    color:white;
    font-weight:800;
    cursor:pointer;
    background:
        linear-gradient(
            135deg,
            var(--blue),
            var(--purple)
        );
}

button:disabled {
    opacity:.5;
}

.metrics {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
    margin-bottom:15px;
}

.metric {
    padding:15px;
}

.metric small {
    color:var(--muted);
    display:block;
}

.metric strong {
    display:block;
    margin-top:7px;
    overflow:hidden;
    text-overflow:ellipsis;
    white-space:nowrap;
}

.progress {
    margin-bottom:15px;
}

.track {
    height:8px;
    background:#080e1c;
    border-radius:20px;
    overflow:hidden;
    margin-top:14px;
}

.bar {
    height:100%;
    width:0%;
    background:
        linear-gradient(
            90deg,
            var(--blue),
            var(--purple),
            var(--green)
        );
    transition:.4s;
}

.agent {
    display:flex;
    align-items:center;
    gap:12px;
    margin-top:15px;
    padding:15px;
    border-radius:12px;
    background:#0b1222;
}

.agent-icon {
    width:42px;
    height:42px;
    display:grid;
    place-items:center;
    border-radius:10px;
    background:#1b2946;
}

.live {
    margin-left:auto;
    color:var(--green);
    font-size:11px;
    font-weight:800;
}

.events {
    height:350px;
    overflow:auto;
    margin-top:15px;
}

.event {
    padding:10px 3px;
    border-bottom:1px solid #202b40;
    font-size:12px;
}

.event .time {
    color:#65728a;
}

.event .agent-name {
    color:#91adff;
    font-weight:800;
    margin:3px 0;
}

pre {
    background:#050912;
    padding:15px;
    border-radius:12px;
    overflow:auto;
    max-height:600px;
    font-size:12px;
    line-height:1.6;
}

.info {
    color:var(--muted);
    font-size:12px;
    line-height:1.6;
    margin-top:15px;
}

.error {
    color:#ff9baa;
    background:#2a1018;
    padding:12px;
    border-radius:10px;
    margin-top:12px;
}

.login {
    position:fixed;
    inset:0;
    display:grid;
    place-items:center;
    background:#050912;
    z-index:10;
}

.login-box {
    width:min(390px,calc(100% - 30px));
}

.hidden {
    display:none !important;
}

@media(max-width:900px) {

    .grid {
        grid-template-columns:1fr;
    }

    .metrics {
        grid-template-columns:repeat(2,1fr);
    }

}

@media(max-width:500px) {

    .metrics {
        grid-template-columns:1fr 1fr;
    }

}

</style>

</head>

<body>

<div id="login" class="login hidden">

    <div class="card login-box">

        <div class="brand">

            <div class="logo">
                TA
            </div>

            <div>
                <h1>TradingAgents</h1>
                <div class="subtitle">
                    Private Command Center
                </div>
            </div>

        </div>

        <label>Password</label>

        <input
            id="password"
            type="password"
            placeholder="Password"
        >

        <button onclick="login()">
            Unlock
        </button>

        <div
            id="loginError"
            class="error hidden"
        >
            Invalid password.
        </div>

    </div>

</div>


<div id="app" class="container hidden">

    <div class="header">

        <div class="brand">

            <div class="logo">
                TA
            </div>

            <div>
                <h1>
                    TradingAgents Command Center
                </h1>

                <div class="subtitle">
                    TradingAgents → OmniRouter → OpenRouter
                </div>
            </div>

        </div>

        <div id="connection">
            Ready
        </div>

    </div>


    <div class="grid">

        <div class="card">

            <h2>New Analysis</h2>

            <label>Ticker</label>

            <input
                id="ticker"
                value="NVDA"
                placeholder="AAPL, NVDA, SPY, BTC-USD"
            >

            <label>Analysis date</label>

            <input
                id="date"
                type="date"
            >

            <button
                id="startButton"
                onclick="startAnalysis()"
            >
                Start Analysis
            </button>

            <div class="info">

                <b>Provider</b><br>

                OpenAI-compatible

                <br><br>

                <b>Gateway</b><br>

                OmniRouter

                <br><br>

                <b>Model</b><br>

                <span id="model">
                    Loading...
                </span>

            </div>

        </div>


        <div>

            <div class="metrics">

                <div class="card metric">
                    <small>Status</small>
                    <strong id="status">
                        Ready
                    </strong>
                </div>

                <div class="card metric">
                    <small>Ticker</small>
                    <strong id="metricTicker">
                        —
                    </strong>
                </div>

                <div class="card metric">
                    <small>Agent</small>
                    <strong id="metricAgent">
                        —
                    </strong>
                </div>

                <div class="card metric">
                    <small>Phase</small>
                    <strong id="metricPhase">
                        —
                    </strong>
                </div>

            </div>


            <div class="card progress">

                <b>Analysis Progress</b>

                <span
                    id="progressNumber"
                    style="float:right;color:#6e98ff"
                >
                    0%
                </span>

                <div
                    id="progressText"
                    class="subtitle"
                >
                    Waiting
                </div>

                <div class="track">

                    <div
                        id="progressBar"
                        class="bar"
                    ></div>

                </div>

            </div>


            <div class="card">

                <b>Live Agent Activity</b>

                <div class="agent">

                    <div class="agent-icon">
                        AI
                    </div>

                    <div>

                        <strong id="activeAgent">
                            No agent running
                        </strong>

                        <div
                            id="activePhase"
                            class="subtitle"
                        >
                            Start an analysis.
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
                    Waiting for activity...
                </div>

            </div>


            <div
                class="card"
                style="margin-top:15px"
            >

                <b>Final Analysis</b>

                <div id="result">
                    No completed analysis yet.
                </div>

            </div>

        </div>

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

    const response =
        await fetch(url, options);

    if (response.status === 401) {

        showLogin();

        throw new Error(
            "Authentication required"
        );
    }

    if (!response.ok) {

        let text =
            await response.text();

        try {

            const data =
                JSON.parse(text);

            text =
                data.detail || text;

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
            data.model ||
            "Not configured";

    } catch {}

    document
        .getElementById("date")
        .value =
        new Date()
            .toISOString()
            .slice(0,10);
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

        alert(
            "Please enter a ticker."
        );

        return;
    }

    const button =
        document
            .getElementById("startButton");

    button.disabled = true;

    button.textContent =
        "Starting...";

    document
        .getElementById("result")
        .innerHTML =
        "<div>Analysis running...</div>";

    document
        .getElementById("events")
        .innerHTML = "";

    try {

        const data =
            await api(
                "/api/analyze",
                {
                    method:"POST",

                    headers:{
                        "Content-Type":
                            "application/json"
                    },

                    body:JSON.stringify({
                        ticker:ticker,
                        analysis_date:date
                    })
                }
            );

        jobId =
            data.job_id;

        document
            .getElementById("connection")
            .textContent =
            "Analysis running";

        poll();

    } catch(error) {

        alert(error.message);

        button.disabled = false;

        button.textContent =
            "Start Analysis";
    }
}


async function poll() {

    if (!jobId) {
        return;
    }

    try {

        const job =
            await api(
                "/api/jobs/" +
                jobId
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
                .getElementById(
                    "startButton"
                )
                .disabled = false;

            document
                .getElementById(
                    "startButton"
                )
                .textContent =
                "Start Analysis";

            document
                .getElementById(
                    "connection"
                )
                .textContent =
                job.status === "completed"
                    ? "Completed"
                    : "Failed";
        }

    } catch(error) {

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
        job.current_phase ||
        "Working...";

    document
        .getElementById("activeAgent")
        .textContent =
        job.current_agent ||
        "System";

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

    renderEvents(
        job.events || []
    );

    if (job.status === "completed") {

        document
            .getElementById("result")
            .innerHTML =
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
            "Waiting for activity...";

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

                        <div class="time">
                            ${escapeHtml(time)}
                        </div>

                        <div class="agent-name">
                            ${escapeHtml(
                                event.agent ||
                                "System"
                            )}
                        </div>

                        <div>
                            ${escapeHtml(
                                event.message ||
                                ""
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
        .replaceAll(
            "&",
            "&amp;"
        )
        .replaceAll(
            "<",
            "&lt;"
        )
        .replaceAll(
            ">",
            "&gt;"
        )
        .replaceAll(
            '"',
            "&quot;"
        )
        .replaceAll(
            "'",
            "&#039;"
        );
}


boot();

</script>

</body>
</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse,
)
def home():
    return HTML


# ============================================================
# 401 HANDLER
# ============================================================

@app.exception_handler(401)
async def unauthorized(
    request: Request,
    exc: HTTPException,
):

    return JSONResponse(
        status_code=401,
        content={
            "error":
                "Authentication required"
        },
    )
