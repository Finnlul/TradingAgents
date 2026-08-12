import os
import uuid
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# TradingAgents Module
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


# ============================================================
# CONFIGURATION & ENVIRONMENT
# ============================================================

APP_PASSWORD = os.getenv("APP_PASSWORD", "")

MODEL = os.getenv(
    "TRADINGAGENTS_MODEL",
    "openai/gpt-4o",
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
    or os.getenv("OPENAI_API_KEY")
    or ""
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="TradingAgents Enterprise Command Center",
    version="4.2.0",
    description="Multi-Agent Market Analysis Orchestrator with Job History & Suggestions",
)


# ============================================================
# EXTENDED DEBUG LOGGING & JOB STORAGE (THREAD-SAFE)
# ============================================================

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()
MAX_JOBS_RETAINED = 100


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(ticker: str, analysis_date: str) -> str:
    job_id = str(uuid.uuid4())
    with jobs_lock:
        if len(jobs) >= MAX_JOBS_RETAINED:
            oldest_key = next(iter(jobs))
            jobs.pop(oldest_key, None)

        jobs[job_id] = {
            "id": job_id,
            "ticker": ticker.upper(),
            "analysis_date": analysis_date,
            "status": "queued",
            "progress": 0,
            "current_agent": "System",
            "current_phase": "Initializing Workflow",
            "started_at": utc_now(),
            "finished_at": None,
            "events": [],
            "debug_logs": [],
            "reports": {
                "market": None,
                "fundamentals": None,
                "technical": None,
                "sentiment": None,
                "news": None,
                "bull_case": None,
                "bear_case": None,
                "debate_summary": None,
                "risk_assessment": None,
            },
            "result": None,
            "error": None,
        }
    return job_id


def update_job(job_id: str, **changes):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(changes)


def log_debug(
    job_id: str,
    message: str,
    level: str = "INFO",
    source: str = "Engine",
    metadata: Optional[Dict[str, Any]] = None,
):
    entry = {
        "time": utc_now(),
        "level": level.upper(),
        "source": source,
        "message": message,
        "metadata": metadata or {},
    }
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["debug_logs"].append(entry)
        if len(job["debug_logs"]) > 2000:
            job["debug_logs"] = job["debug_logs"][-2000:]


def add_event(
    job_id: str,
    message: str,
    agent: str = "System",
    phase: str = "",
    progress: Optional[int] = None,
    level: str = "INFO",
):
    event = {
        "time": utc_now(),
        "agent": agent,
        "phase": phase,
        "message": message,
        "level": level,
    }
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["events"].append(event)
        if len(job["events"]) > 1000:
            job["events"] = job["events"][-1000:]
        if progress is not None:
            job["progress"] = max(0, min(100, progress))
        job["current_agent"] = agent
        if phase:
            job["current_phase"] = phase

    log_debug(job_id, f"[{agent}] ({phase}) {message}", level=level, source=agent)


# ============================================================
# SAFE DATA PARSER & SERIALIZER
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
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): serialize_result(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_result(v) for v in value]
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


def extract_agent_reports(final_state: Any) -> Dict[str, Any]:
    reports = {
        "market": None,
        "fundamentals": None,
        "technical": None,
        "sentiment": None,
        "news": None,
        "bull_case": None,
        "bear_case": None,
        "debate_summary": None,
        "risk_assessment": None,
    }
    if not isinstance(final_state, dict):
        return reports

    key_mappings = {
        "market": ["market_analysis", "market_report", "market_data"],
        "fundamentals": ["fundamental_analysis", "fundamentals_report", "financial_metrics"],
        "technical": ["technical_analysis", "technical_report", "indicators"],
        "sentiment": ["sentiment_analysis", "sentiment_report", "social_sentiment"],
        "news": ["news_analysis", "news_report", "recent_headlines"],
        "bull_case": ["bull_case", "bullish_thesis", "bull_research"],
        "bear_case": ["bear_case", "bearish_thesis", "bear_research"],
        "debate_summary": ["debate_summary", "research_manager_verdict", "debate_state"],
        "risk_assessment": ["risk_assessment", "risk_report", "risk_limits"],
    }

    for report_name, candidate_keys in key_mappings.items():
        extracted = None
        for key in candidate_keys:
            if key in final_state and final_state[key]:
                extracted = final_state[key]
                break
        reports[report_name] = serialize_result(extracted) if extracted else None

    return reports


# ============================================================
# AUTHENTICATION & VALIDATION
# ============================================================

def check_password(request: Request):
    if not APP_PASSWORD:
        return
    supplied = request.headers.get("X-App-Password", "")
    if supplied != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid authorization token or password.")


def validate_llm_config():
    if not MODEL:
        raise RuntimeError("TRADINGAGENTS_MODEL is not set.")
    if not BACKEND_URL:
        raise RuntimeError("OMNIROUTER_BASE_URL (or backend base URL) is missing.")
    if not API_KEY:
        raise RuntimeError("OMNIROUTER_API_KEY (or OpenAI key) is missing.")


class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=30)
    analysis_date: Optional[str] = None
    debug_mode: Optional[bool] = True


# ============================================================
# WORKER EXECUTION
# ============================================================

def run_analysis(job_id: str, ticker: str, analysis_date: str):
    try:
        update_job(job_id, status="running", progress=5)
        add_event(job_id, f"Initializing multi-agent pipeline for target {ticker}", "System", "Setup", 5)
        
        validate_llm_config()

        os.environ["OPENAI_COMPATIBLE_API_KEY"] = API_KEY
        os.environ["OPENAI_API_KEY"] = API_KEY
        os.environ["OPENROUTER_API_KEY"] = API_KEY
        os.environ["OPENAI_BASE_URL"] = BACKEND_URL
        os.environ["OPENAI_API_BASE"] = BACKEND_URL

        sanitized_model = MODEL
        if sanitized_model.startswith("nvidia/") and not sanitized_model.startswith("openrouter/"):
            sanitized_model = f"openrouter/{sanitized_model}"

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "openai_compatible"
        config["deep_think_llm"] = sanitized_model
        config["quick_think_llm"] = sanitized_model
        config["backend_url"] = BACKEND_URL

        add_event(job_id, "Compiling LangGraph workflow graph...", "System", "Compilation", 12)
        ta = TradingAgentsGraph(debug=True, config=config)
        
        add_event(job_id, "Workflow graph compiled. Executing evaluation steps...", "System", "Execution", 20)
        target_date = analysis_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        
        # Blocking Multi-Agent Propagation
        final_state, decision = ta.propagate(ticker, target_date)

        serialized_decision = serialize_result(decision)
        serialized_state = serialize_result(final_state)
        extracted_reports = extract_agent_reports(serialized_state)

        result = {
            "ticker": ticker,
            "analysis_date": target_date,
            "decision": serialized_decision,
            "reports": extracted_reports,
            "final_state": serialized_state,
        }

        with jobs_lock:
            if job_id in jobs:
                jobs[job_id]["reports"] = extracted_reports

        add_event(job_id, f"Analysis complete for {ticker}.", "Portfolio Manager", "Finalized", 100)
        
        update_job(
            job_id,
            status="completed",
            progress=100,
            current_agent="Portfolio Manager",
            current_phase="Analysis Complete",
            finished_at=utc_now(),
            result=result,
        )

    except Exception as exc:
        error_msg = safe_string(exc)
        tb = traceback.format_exc()
        log_debug(job_id, f"Execution failed: {error_msg}\n{tb}", level="ERROR", source="WorkerException")
        add_event(job_id, f"Analysis aborted: {error_msg}", "System", "Failure", progress=100, level="ERROR")
        update_job(
            job_id,
            status="failed",
            finished_at=utc_now(),
            error={"message": error_msg, "traceback": tb},
        )


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": utc_now()}


@app.get("/api/config")
def api_config(request: Request):
    check_password(request)
    return {
        "model": MODEL or "Not configured",
        "backend_url": BACKEND_URL or "Not configured",
    }


@app.get("/api/jobs")
def list_jobs(request: Request):
    check_password(request)
    with jobs_lock:
        summary = [
            {
                "id": j["id"],
                "ticker": j["ticker"],
                "status": j["status"],
                "progress": j["progress"],
                "started_at": j["started_at"],
                "current_phase": j["current_phase"]
            }
            for j in reversed(list(jobs.values()))
        ]
    return summary


@app.post("/api/analyze")
def start_analyze(data: AnalyzeRequest, request: Request):
    check_password(request)
    ticker = data.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol is required.")

    analysis_date = data.analysis_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    job_id = create_job(ticker, analysis_date)

    worker_thread = threading.Thread(
        target=run_analysis,
        args=(job_id, ticker, analysis_date),
        daemon=True,
    )
    worker_thread.start()

    return {"status": "started", "job_id": job_id, "ticker": ticker}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str, request: Request):
    check_password(request)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        return job


# ============================================================
# COMPLETE ENTERPRISE DASHBOARD UI
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingAgents | Command Center</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #06090e;
            --bg-surface: #0d121d;
            --bg-elevated: #131b2e;
            --border-subtle: #1e293b;
            --border-bright: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --text-dim: #64748b;
            --accent-blue: #38bdf8;
            --accent-indigo: #6366f1;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --accent-amber: #f59e0b;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .container {
            width: min(1600px, calc(100% - 32px));
            margin: 0 auto;
            padding: 20px 0 40px;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 20px;
            flex-wrap: wrap;
            gap: 12px;
        }

        .brand { display: flex; align-items: center; gap: 14px; }
        .logo-badge {
            width: 40px; height: 40px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-indigo));
            border-radius: 10px;
            display: grid; place-items: center;
            font-weight: 800; color: white;
        }
        .title-group h1 { font-size: 18px; font-weight: 700; }
        .title-group p { font-size: 12px; color: var(--text-muted); }

        .layout-grid {
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 20px;
        }

        @media (max-width: 1024px) {
            .layout-grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 13px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 12px;
        }

        label { display: block; font-size: 11px; font-weight: 600; color: var(--text-muted); margin: 10px 0 4px; }
        input, select {
            width: 100%; background: var(--bg-base); border: 1px solid var(--border-subtle);
            color: var(--text-main); padding: 10px 12px; border-radius: 8px; font-size: 13px; outline: none;
        }
        input:focus { border-color: var(--accent-blue); }

        .btn-primary {
            width: 100%; background: linear-gradient(135deg, var(--accent-blue), var(--accent-indigo));
            border: none; color: white; padding: 12px; border-radius: 8px; font-size: 13px;
            font-weight: 700; cursor: pointer; margin-top: 16px; transition: opacity 0.2s ease;
        }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

        .job-history-list {
            max-height: 240px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 6px;
        }
        .job-item {
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            padding: 8px 10px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s ease;
        }
        .job-item:hover, .job-item.active { border-color: var(--accent-blue); background: var(--bg-elevated); }
        .badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; font-family: monospace; }
        .badge.completed { background: rgba(16, 185, 129, 0.2); color: var(--accent-emerald); }
        .badge.running { background: rgba(56, 189, 248, 0.2); color: var(--accent-blue); }
        .badge.failed { background: rgba(244, 63, 94, 0.2); color: var(--accent-rose); }
        .badge.queued { background: rgba(245, 158, 11, 0.2); color: var(--accent-amber); }

        .metrics-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }
        .metric-tile {
            background: var(--bg-surface); border: 1px solid var(--border-subtle);
            border-radius: 10px; padding: 12px;
        }
        .metric-tile .label { font-size: 10px; text-transform: uppercase; color: var(--text-dim); font-weight: 700; }
        .metric-tile .value { font-size: 15px; font-weight: 700; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

        .progress-container {
            background: var(--bg-surface); border: 1px solid var(--border-subtle);
            border-radius: 10px; padding: 14px; margin-bottom: 16px;
        }
        .progress-track { height: 6px; background: var(--bg-base); border-radius: 999px; overflow: hidden; margin-top: 8px; }
        .progress-fill { height: 100%; width: 0%; background: linear-gradient(90deg, var(--accent-blue), var(--accent-emerald)); transition: width 0.3s ease; }

        .tabs-nav { display: flex; gap: 6px; border-bottom: 1px solid var(--border-subtle); margin-bottom: 16px; overflow-x: auto; }
        .tab-btn {
            background: transparent; border: none; color: var(--text-muted); padding: 8px 14px;
            font-size: 12px; font-weight: 600; cursor: pointer; border-bottom: 2px solid transparent; white-space: nowrap;
        }
        .tab-btn.active { color: var(--accent-blue); border-bottom-color: var(--accent-blue); }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .logs-window {
            background: var(--bg-base); border: 1px solid var(--border-subtle);
            border-radius: 10px; height: 420px; overflow-y: auto; padding: 12px;
            font-family: 'JetBrains Mono', monospace; font-size: 11px; line-height: 1.5;
        }
        .log-entry { padding: 4px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.03); }
        .log-entry.error { color: var(--accent-rose); }

        .report-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
        .report-card { background: var(--bg-base); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 14px; }
        .report-card h4 { font-size: 12px; color: var(--accent-blue); margin-bottom: 8px; text-transform: uppercase; }

        pre { background: var(--bg-surface); border: 1px solid var(--border-subtle); padding: 12px; border-radius: 8px; font-family: 'JetBrains Mono', monospace; font-size: 11px; overflow-x: auto; max-height: 400px; color: var(--text-muted); }

        .auth-overlay {
            position: fixed; inset: 0; background: rgba(6, 9, 14, 0.95);
            display: grid; place-items: center; z-index: 100;
        }
        .auth-modal { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 16px; padding: 24px; width: min(380px, 90vw); }
        .hidden { display: none !important; }
    </style>
</head>
<body>

<div id="authScreen" class="auth-overlay hidden">
    <div class="auth-modal">
        <h3 style="margin-bottom: 12px;">Authentication Required</h3>
        <label>Passkey / Secret</label>
        <input id="authPassword" type="password" placeholder="Enter APP_PASSWORD">
        <button class="btn-primary" onclick="handleAuth()">Submit</button>
    </div>
</div>

<div class="container">
    <header>
        <div class="brand">
            <div class="logo-badge">TA</div>
            <div class="title-group">
                <h1>TradingAgents Command Center</h1>
                <p>Multi-Agent Financial Intelligence Orchestration</p>
            </div>
        </div>
        <div id="gatewayStatus" style="font-size: 12px; color: var(--accent-emerald);">● Online</div>
    </header>

    <div class="metrics-bar">
        <div class="metric-tile"><div class="label">Target Symbol</div><div id="metricTicker" class="value">—</div></div>
        <div class="metric-tile"><div class="label">Active Node</div><div id="metricAgent" class="value">Idle</div></div>
        <div class="metric-tile"><div class="label">Phase</div><div id="metricPhase" class="value">Ready</div></div>
        <div class="metric-tile"><div class="label">Job ID</div><div id="metricJobId" class="value">—</div></div>
    </div>

    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 600;">
            <span id="progressPhaseText">Pipeline Status: Ready</span>
            <span id="progressValue" style="color: var(--accent-blue);">0%</span>
        </div>
        <div class="progress-track"><div id="progressBar" class="progress-fill"></div></div>
    </div>

    <div class="layout-grid">
        <aside>
            <div class="card">
                <div class="card-title">Launch Pipeline</div>
                <label>Equity / Ticker (mit Vorschlägen)</label>
                <input id="inputTicker" type="text" list="tickerSuggestions" placeholder="z.B. SAP, NVDA, AAPL">
                <datalist id="tickerSuggestions">
                    <option value="SAP">SAP SE</option>
                    <option value="NVDA">NVIDIA Corp</option>
                    <option value="AAPL">Apple Inc</option>
                    <option value="TSLA">Tesla Inc</option>
                    <option value="MSFT">Microsoft Corp</option>
                    <option value="BTC-USD">Bitcoin USD</option>
                    <option value="ETH-USD">Ethereum USD</option>
                </datalist>
                
                <label>Anchor Date</label>
                <input id="inputDate" type="date">

                <button id="btnRun" class="btn-primary" onclick="launchAnalysis()">Run Pipeline</button>
            </div>

            <div class="card">
                <div class="card-title">Aktive & Vorherige Jobs</div>
                <div id="jobHistoryList" class="job-history-list">
                    <div style="color: var(--text-dim); font-size: 11px;">Keine Jobs vorhanden.</div>
                </div>
            </div>
        </aside>

        <main>
            <div class="tabs-nav">
                <button class="tab-btn active" onclick="switchTab('tabEvents', event)">Workflow Events</button>
                <button class="tab-btn" onclick="switchTab('tabReports', event)">Intelligence Reports</button>
                <button class="tab-btn" onclick="switchTab('tabDecision', event)">Decision & Mandate</button>
            </div>

            <div id="tabEvents" class="tab-content active">
                <div class="card">
                    <div class="card-title">Telemetry Feed</div>
                    <div id="eventsFeed" class="logs-window">Warte auf Start...</div>
                </div>
            </div>

            <div id="tabReports" class="tab-content">
                <div class="report-grid">
                    <div class="report-card"><h4>Market Context</h4><div id="repMarket" class="text-dim">Keine Daten</div></div>
                    <div class="report-card"><h4>Fundamentals</h4><div id="repFundamentals" class="text-dim">Keine Daten</div></div>
                    <div class="report-card"><h4>Technical Overview</h4><div id="repTechnical" class="text-dim">Keine Daten</div></div>
                    <div class="report-card"><h4>Sentiment</h4><div id="repSentiment" class="text-dim">Keine Daten</div></div>
                    <div class="report-card"><h4>News</h4><div id="repNews" class="text-dim">Keine Daten</div></div>
                    <div class="report-card"><h4>Bull Thesis</h4><div id="repBull" class="text-dim">Keine Daten</div></div>
                    <div class="report-card"><h4>Bear Thesis</h4><div id="repBear" class="text-dim">Keine Daten</div></div>
                    <div class="report-card"><h4>Debate Summary</h4><div id="repDebate" class="text-dim">Keine Daten</div></div>
                    <div class="report-card"><h4>Risk Assessment</h4><div id="repRisk" class="text-dim">Keine Daten</div></div>
                </div>
            </div>

            <div id="tabDecision" class="tab-content">
                <div class="card">
                    <div id="decisionView">Noch keine Entscheidung getroffen.</div>
                </div>
            </div>
        </main>
    </div>
</div>

<script>
let activeJobId = null;
let pollHandle = null;

function getAuthHeader() {
    const pwd = localStorage.getItem("ta_passkey") || "";
    return pwd ? { "X-App-Password": pwd } : {};
}

async function request(endpoint, options = {}) {
    options.headers = { ...options.headers, ...getAuthHeader() };
    const res = await fetch(endpoint, options);
    if (res.status === 401) {
        document.getElementById("authScreen").classList.remove("hidden");
        throw new Error("Unauthorized");
    }
    if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || "Request failed");
    }
    return res.json();
}

async function handleAuth() {
    const pwd = document.getElementById("authPassword").value;
    localStorage.setItem("ta_passkey", pwd);
    document.getElementById("authScreen").classList.add("hidden");
    loadHistory();
}

function switchTab(tabId, evt) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    if (evt && evt.target) evt.target.classList.add("active");
    document.getElementById(tabId).classList.add("active");
}

document.getElementById("inputDate").value = new Date().toISOString().slice(0, 10);

async function loadHistory() {
    try {
        const jobs = await request("/api/jobs");
        const listEl = document.getElementById("jobHistoryList");
        if (!jobs.length) {
            listEl.innerHTML = `<div style="color: var(--text-dim); font-size: 11px;">Keine Jobs vorhanden.</div>`;
            return;
        }
        listEl.innerHTML = jobs.map(j => `
            <div class="job-item ${j.id === activeJobId ? 'active' : ''}" onclick="selectJob('${j.id}')">
                <span><b>${escapeHtml(j.ticker)}</b> <small style="color: var(--text-dim);">${new Date(j.started_at).toLocaleTimeString()}</small></span>
                <span class="badge ${j.status}">${j.status}</span>
            </div>
        `).join("");
    } catch (e) {
        console.error("History load error:", e);
    }
}

async function selectJob(jobId) {
    activeJobId = jobId;
    if (pollHandle) clearTimeout(pollHandle);
    loadHistory();
    poll();
}

async function launchAnalysis() {
    const ticker = document.getElementById("inputTicker").value.trim();
    const date = document.getElementById("inputDate").value;
    if (!ticker) return alert("Bitte Ticker eingeben.");

    const btn = document.getElementById("btnRun");
    btn.disabled = true;
    btn.textContent = "Starte...";

    try {
        const res = await request("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker, analysis_date: date })
        });
        activeJobId = res.job_id;
        loadHistory();
        poll();
    } catch (e) {
        alert("Fehler: " + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = "Run Pipeline";
    }
}

async function poll() {
    if (!activeJobId) return;
    try {
        const job = await request(`/api/jobs/${activeJobId}`);
        renderJob(job);
        loadHistory();
        if (job.status === "running" || job.status === "queued") {
            pollHandle = setTimeout(poll, 1500);
        }
    } catch (e) {
        console.error("Poll Error:", e);
        pollHandle = setTimeout(poll, 3000);
    }
}

function renderJob(job) {
    document.getElementById("metricTicker").textContent = job.ticker || "—";
    document.getElementById("metricAgent").textContent = job.current_agent || "System";
    document.getElementById("metricPhase").textContent = job.current_phase || "—";
    document.getElementById("metricJobId").textContent = job.id.slice(0, 8) + "...";
    document.getElementById("progressValue").textContent = `${job.progress}%`;
    document.getElementById("progressBar").style.width = `${job.progress}%`;
    document.getElementById("progressPhaseText").textContent = `Status: ${job.current_phase}`;

    if (job.events) {
        const feed = document.getElementById("eventsFeed");
        const isAtBottom = feed.scrollHeight - feed.scrollTop <= feed.clientHeight + 40;
        
        feed.innerHTML = job.events.map(ev => `
            <div class="log-entry ${ev.level === 'ERROR' ? 'error' : ''}">
                <span style="color: var(--text-dim);">${new Date(ev.time).toLocaleTimeString()}</span>
                <span style="color: var(--accent-blue);">[${escapeHtml(ev.agent)}]</span>
                <span>${escapeHtml(ev.message)}</span>
            </div>
        `).join("");
        
        if (isAtBottom) feed.scrollTop = feed.scrollHeight;
    }

    if (job.reports) {
        const map = { 
            market: "repMarket", 
            fundamentals: "repFundamentals", 
            technical: "repTechnical", 
            sentiment: "repSentiment", 
            news: "repNews", 
            bull_case: "repBull", 
            bear_case: "repBear",
            debate_summary: "repDebate",
            risk_assessment: "repRisk"
        };
        for (const [k, id] of Object.entries(map)) {
            const el = document.getElementById(id);
            if (el && job.reports[k]) {
                const val = job.reports[k];
                el.innerHTML = `<pre>${escapeHtml(typeof val === "object" ? JSON.stringify(val, null, 2) : val)}</pre>`;
            }
        }
    }

    if (job.result && job.result.decision) {
        document.getElementById("decisionView").innerHTML = `
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid var(--accent-emerald); padding: 16px; border-radius: 10px; margin-bottom: 12px;">
                <h3 style="color: var(--accent-emerald);">MANDAT ERTEILT</h3>
            </div>
            <pre>${escapeHtml(JSON.stringify(job.result.decision, null, 2))}</pre>
        `;
    } else if (job.error) {
        document.getElementById("decisionView").innerHTML = `
            <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid var(--accent-rose); padding: 16px; border-radius: 10px; margin-bottom: 12px;">
                <h3 style="color: var(--accent-rose);">PIPELINE FEHLGESCHLAGEN</h3>
            </div>
            <pre>${escapeHtml(JSON.stringify(job.error, null, 2))}</pre>
        `;
    }
}

function escapeHtml(str) {
    return String(str || '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;');
}

loadHistory();
setInterval(loadHistory, 10000);
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
