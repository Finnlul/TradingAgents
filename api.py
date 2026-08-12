import os
import sys
import io
import uuid
import json
import logging
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

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
    version="4.0.0",
    description="Multi-Agent Market Analysis Orchestrator with Extended Debugging",
)


# ============================================================
# EXTENDED DEBUG LOGGING & JOB STORAGE
# ============================================================

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_job(ticker: str, analysis_date: str) -> str:
    job_id = str(uuid.uuid4())
    with jobs_lock:
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
    """Extrahiert Teilergebnisse der TradingAgents-Stufen."""
    reports = {}
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
# AUTHENTICATION
# ============================================================

def check_password(request: Request):
    if not APP_PASSWORD:
        return
    supplied = request.headers.get("X-App-Password", "")
    if supplied != APP_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid authorization token or password.")


# ============================================================
# SCHEMAS
# ============================================================

class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=30)
    analysis_date: Optional[str] = None
    debug_mode: Optional[bool] = True


# ============================================================
# LLM VALIDATION
# ============================================================

def validate_llm_config():
    if not MODEL:
        raise RuntimeError("TRADINGAGENTS_MODEL is not set. Specify model like 'openai/gpt-4o' or 'anthropic/claude-3.5-sonnet'.")
    if not BACKEND_URL:
        raise RuntimeError("OMNIROUTER_BASE_URL (or OPENAI_BASE_URL) is missing.")
    if not API_KEY:
        raise RuntimeError("OMNIROUTER_API_KEY (or OPENAI_API_KEY) is missing.")


# ============================================================
# WORKER EXECUTION
# ============================================================

def run_analysis(job_id: str, ticker: str, analysis_date: str):
    try:
        update_job(job_id, status="running", progress=2)
        add_event(job_id, f"Initializing multi-agent pipeline for target {ticker}", "System", "Setup", 2)
        log_debug(job_id, f"Job initialized with target date: {analysis_date}", level="DEBUG", source="Bootstrap")

        validate_llm_config()

        os.environ["OPENAI_COMPATIBLE_API_KEY"] = API_KEY
        os.environ["OPENAI_API_KEY"] = API_KEY
        os.environ["OPENROUTER_API_KEY"] = API_KEY
        os.environ["OPENAI_BASE_URL"] = BACKEND_URL
        os.environ["OPENAI_API_BASE"] = BACKEND_URL

        sanitized_model = MODEL
        if sanitized_model.startswith("nvidia/") and not sanitized_model.startswith("openrouter/"):
            sanitized_model = f"openrouter/{sanitized_model}"

        log_debug(job_id, f"Model sanitized: '{sanitized_model}' targeting gateway '{BACKEND_URL}'", level="DEBUG", source="Config")

        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = "openai_compatible"
        config["deep_think_llm"] = sanitized_model
        config["quick_think_llm"] = sanitized_model
        config["backend_url"] = BACKEND_URL

        add_event(job_id, f"Loaded agent config with LLM: {sanitized_model}", "System", "Configuration", 8)

        phases = [
            ("Market Analyst", "Market Overview & Price Action", 15),
            ("Fundamentals Analyst", "Financial Statement & Metric Valuation", 25),
            ("Technical Analyst", "Moving Averages & Oscillators", 35),
            ("Sentiment & News Analyst", "Market Sentiment & Catalyst Breakdown", 48),
            ("Bull vs Bear Debate Team", "Bullish & Bearish Hypothesis Evaluation", 62),
            ("Research Manager", "Consensus & Synthesis Synthesis", 75),
            ("Risk Management", "Position Sizing & Drawdown Limits", 88),
            ("Portfolio Manager", "Execution Mandate & Portfolio Decision", 95),
        ]

        for agent, phase_desc, _ in phases:
            log_debug(job_id, f"Registered stage: {agent} -> {phase_desc}", level="DEBUG", source="GraphBuilder")

        add_event(job_id, "Compiling LangGraph workflow graph...", "System", "Compilation", 12)
        
        ta = TradingAgentsGraph(debug=True, config=config)
        add_event(job_id, "Workflow graph compiled successfully. Starting execution.", "System", "Execution", 15)

        # Execution
        target_date = analysis_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        final_state, decision = ta.propagate(ticker, target_date)

        log_debug(job_id, "Execution finished. Parsing decision states and reports.", level="DEBUG", source="Parser")

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

        add_event(job_id, f"Analysis complete for {ticker}. Final decision dispatched.", "Portfolio Manager", "Finalized", 100)
        
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
        log_debug(job_id, f"Execution failed with error: {error_msg}\n{tb}", level="ERROR", source="WorkerException")
        
        add_event(job_id, f"Analysis aborted: {error_msg}", "System", "Failure", level="ERROR")
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
    return {
        "status": "healthy",
        "service": "TradingAgents Orchestration API",
        "model": MODEL,
        "backend_configured": bool(BACKEND_URL),
        "api_key_configured": bool(API_KEY),
        "timestamp": utc_now(),
    }


@app.get("/api/config")
def api_config(request: Request):
    check_password(request)
    return {
        "model": MODEL or "Not configured",
        "provider": "openai_compatible",
        "backend_url": BACKEND_URL or "Not configured",
        "backend_configured": bool(BACKEND_URL),
        "api_key_configured": bool(API_KEY),
        "password_enabled": bool(APP_PASSWORD),
    }


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

    return {"status": "started", "job_id": job_id, "ticker": ticker, "analysis_date": analysis_date}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str, request: Request):
    check_password(request)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        return job


@app.get("/api/jobs/{job_id}/logs")
def get_job_logs(job_id: str, request: Request, level: Optional[str] = None):
    check_password(request)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        logs = job.get("debug_logs", [])
        if level:
            logs = [l for l in logs if l.get("level") == level.upper()]
        return {"job_id": job_id, "count": len(logs), "logs": logs}


# ============================================================
# WEB UI (HTML / CSS / JS)
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingAgents | Command & Debug Center</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
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
            --accent-purple: #a855f7;
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
            background-image: 
                radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(99, 102, 241, 0.08) 0px, transparent 50%);
        }

        .container {
            width: min(1600px, calc(100% - 40px));
            margin: 0 auto;
            padding: 24px 0 60px;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 24px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .logo-badge {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-indigo));
            border-radius: 12px;
            display: grid;
            place-items: center;
            font-weight: 800;
            font-size: 18px;
            color: white;
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
        }

        .title-group h1 { font-size: 20px; font-weight: 700; letter-spacing: -0.02em; }
        .title-group p { font-size: 13px; color: var(--text-muted); }

        .system-pill {
            display: flex;
            align-items: center;
            gap: 10px;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            padding: 8px 16px;
            border-radius: 999px;
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-emerald);
            box-shadow: 0 0 8px var(--accent-emerald);
        }

        .layout-grid {
            display: grid;
            grid-template-columns: 360px 1fr;
            gap: 24px;
        }

        .card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 20px;
            position: relative;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
        }

        label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            margin: 14px 0 6px;
        }

        input, select {
            width: 100%;
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            color: var(--text-main);
            padding: 12px 14px;
            border-radius: 10px;
            font-size: 14px;
            font-family: inherit;
            outline: none;
            transition: border-color 0.2s;
        }

        input:focus { border-color: var(--accent-blue); }

        .btn-primary {
            width: 100%;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-indigo));
            border: none;
            color: white;
            padding: 14px;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            margin-top: 20px;
            transition: opacity 0.2s, transform 0.1s;
        }

        .btn-primary:hover { opacity: 0.95; }
        .btn-primary:active { transform: scale(0.99); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

        .meta-list {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid var(--border-subtle);
            display: flex;
            flex-direction: column;
            gap: 12px;
            font-size: 13px;
        }

        .meta-row {
            display: flex;
            justify-content: space-between;
        }

        .meta-row span { color: var(--text-dim); }
        .meta-row strong { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

        .metrics-bar {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 16px;
            margin-bottom: 24px;
        }

        .metric-tile {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 16px;
        }

        .metric-tile .label { font-size: 11px; text-transform: uppercase; color: var(--text-dim); font-weight: 700; }
        .metric-tile .value { font-size: 18px; font-weight: 700; margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

        .progress-container {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 18px;
            margin-bottom: 24px;
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 10px;
        }

        .progress-track {
            height: 8px;
            background: var(--bg-base);
            border-radius: 999px;
            overflow: hidden;
            position: relative;
        }

        .progress-fill {
            height: 100%;
            width: 0%;
            background: linear-gradient(90deg, var(--accent-blue), var(--accent-indigo), var(--accent-emerald));
            transition: width 0.4s ease;
        }

        .tabs-nav {
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 20px;
        }

        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 10px 18px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            border-bottom: 2px solid transparent;
            font-family: inherit;
        }

        .tab-btn.active {
            color: var(--accent-blue);
            border-bottom-color: var(--accent-blue);
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        .logs-window {
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            height: 480px;
            overflow-y: auto;
            padding: 14px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            line-height: 1.6;
        }

        .log-entry {
            display: flex;
            gap: 12px;
            padding: 4px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }

        .log-time { color: var(--text-dim); min-width: 85px; }
        .log-level { font-weight: 700; min-width: 55px; }
        .log-level.DEBUG { color: var(--text-dim); }
        .log-level.INFO { color: var(--accent-blue); }
        .log-level.WARN { color: var(--accent-amber); }
        .log-level.ERROR { color: var(--accent-rose); }
        .log-source { color: var(--accent-purple); min-width: 120px; }
        .log-msg { color: var(--text-main); word-break: break-all; }

        .report-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .report-card {
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 16px;
        }

        .report-card h4 {
            font-size: 13px;
            color: var(--accent-blue);
            margin-bottom: 10px;
            text-transform: uppercase;
        }

        .decision-banner {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(56, 189, 248, 0.1));
            border: 1px solid var(--accent-emerald);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 20px;
        }

        .decision-title { font-size: 24px; font-weight: 800; color: var(--accent-emerald); }

        pre {
            background: var(--bg-base);
            border: 1px solid var(--border-subtle);
            padding: 16px;
            border-radius: 12px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            overflow-x: auto;
            max-height: 500px;
        }

        .auth-overlay {
            position: fixed;
            inset: 0;
            background: rgba(6, 9, 14, 0.95);
            display: grid;
            place-items: center;
            z-index: 100;
        }

        .auth-modal {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 32px;
            width: min(420px, 90vw);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        }

        .hidden { display: none !important; }

        @media (max-width: 1024px) {
            .layout-grid { grid-template-columns: 1fr; }
            .metrics-bar { grid-template-columns: repeat(2, 1fr); }
            .report-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

<div id="authScreen" class="auth-overlay hidden">
    <div class="auth-modal">
        <div class="brand" style="margin-bottom: 24px;">
            <div class="logo-badge">TA</div>
            <div class="title-group">
                <h1>TradingAgents</h1>
                <p>Authentication Required</p>
            </div>
        </div>
        <label>Passkey / Secret</label>
        <input id="authPassword" type="password" placeholder="Enter APP_PASSWORD">
        <button class="btn-primary" onclick="handleAuth()">Authenticate Session</button>
        <div id="authError" style="color: var(--accent-rose); font-size: 12px; margin-top: 10px; display: none;">Invalid credentials.</div>
    </div>
</div>

<div class="container">
    <header>
        <div class="brand">
            <div class="logo-badge">TA</div>
            <div class="title-group">
                <h1>TradingAgents Command Center</h1>
                <p>Autonomous Multi-Agent Alpha Generation & Risk Intelligence</p>
            </div>
        </div>
        <div class="system-pill">
            <div id="statusDot" class="status-dot"></div>
            <span id="gatewayStatus">OmniRouter Online</span>
        </div>
    </header>

    <div class="metrics-bar">
        <div class="metric-tile">
            <div class="label">Target Symbol</div>
            <div id="metricTicker" class="value">—</div>
        </div>
        <div class="metric-tile">
            <div class="label">Active Node</div>
            <div id="metricAgent" class="value">System Idle</div>
        </div>
        <div class="metric-tile">
            <div class="label">Execution Phase</div>
            <div id="metricPhase" class="value">Awaiting Order</div>
        </div>
        <div class="metric-tile">
            <div class="label">Job ID</div>
            <div id="metricJobId" class="value" style="font-family: 'JetBrains Mono', monospace; font-size: 13px; line-height: 24px;">—</div>
        </div>
    </div>

    <div class="progress-container">
        <div class="progress-header">
            <span id="progressPhaseText">Pipeline Status: Ready</span>
            <span id="progressValue" style="color: var(--accent-blue); font-family: 'JetBrains Mono';">0%</span>
        </div>
        <div class="progress-track">
            <div id="progressBar" class="progress-fill"></div>
        </div>
    </div>

    <div class="layout-grid">
        <aside>
            <div class="card">
                <div class="card-title">Launch Pipeline</div>
                <label>Equity / Asset Ticker</label>
                <input id="inputTicker" type="text" value="NVDA" placeholder="NVDA, TSLA, BTC-USD">
                
                <label>Analysis Anchor Date</label>
                <input id="inputDate" type="date">

                <button id="btnRun" class="btn-primary" onclick="launchAnalysis()">Run Multi-Agent Engine</button>

                <div class="meta-list">
                    <div class="meta-row">
                        <span>LLM Model</span>
                        <strong id="metaModel">Loading...</strong>
                    </div>
                    <div class="meta-row">
                        <span>Provider Protocol</span>
                        <strong>OpenAI-Compatible</strong>
                    </div>
                    <div class="meta-row">
                        <span>Execution Gateway</span>
                        <strong style="color: var(--accent-emerald);">OmniRouter / v1</strong>
                    </div>
                </div>
            </div>
        </aside>

        <main>
            <div class="tabs-nav">
                <button class="tab-btn active" onclick="switchTab('tabEvents', event)">Live Workflow Events</button>
                <button class="tab-btn" onclick="switchTab('tabReports', event)">Agent Intelligence Reports</button>
                <button class="tab-btn" onclick="switchTab('tabDecision', event)">Decision & Mandate</button>
                <button class="tab-btn" onclick="switchTab('tabLogs', event)">Extended Debug Logs</button>
            </div>

            <div id="tabEvents" class="tab-content active">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Agent Telemetry Feed</div>
                    </div>
                    <div id="eventsFeed" class="logs-window">
                        <div style="color: var(--text-dim);">No analysis active. Initialize a run on the left panel.</div>
                    </div>
                </div>
            </div>

            <div id="tabReports" class="tab-content">
                <div class="report-grid" id="reportsContainer">
                    <div class="report-card"><h4>Market Context</h4><div id="repMarket" class="text-dim">Pending execution...</div></div>
                    <div class="report-card"><h4>Fundamentals</h4><div id="repFundamentals" class="text-dim">Pending execution...</div></div>
                    <div class="report-card"><h4>Technical Overview</h4><div id="repTechnical" class="text-dim">Pending execution...</div></div>
                    <div class="report-card"><h4>Sentiment & News</h4><div id="repSentiment" class="text-dim">Pending execution...</div></div>
                    <div class="report-card"><h4>Bull Thesis</h4><div id="repBull" class="text-dim">Pending execution...</div></div>
                    <div class="report-card"><h4>Bear Thesis</h4><div id="repBear" class="text-dim">Pending execution...</div></div>
                </div>
            </div>

            <div id="tabDecision" class="tab-content">
                <div class="card">
                    <div id="decisionView">
                        <div style="color: var(--text-dim);">No final decision rendered yet.</div>
                    </div>
                </div>
            </div>

            <div id="tabLogs" class="tab-content">
                <div class="card">
                    <div class="card-header">
                        <div class="card-title">Low-Level Intercepted Logs</div>
                        <button onclick="downloadLogs()" style="background: var(--bg-elevated); border: 1px solid var(--border-bright); color: var(--text-muted); padding: 4px 10px; border-radius: 6px; cursor: pointer; font-size: 11px;">Export Logs (.json)</button>
                    </div>
                    <div id="debugLogsFeed" class="logs-window"></div>
                </div>
            </div>
        </main>
    </div>
</div>

<script>
let activeJobId = null;
let pollHandle = null;
let currentLogs = [];

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
    try {
        await init();
        document.getElementById("authScreen").classList.add("hidden");
    } catch {
        document.getElementById("authError").style.display = "block";
    }
}

function switchTab(tabId, evt) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    if (evt && evt.target) {
        evt.target.classList.add("active");
    }
    const targetTab = document.getElementById(tabId);
    if (targetTab) {
        targetTab.classList.add("active");
    }
}

async function init() {
    try {
        const cfg = await request("/api/config");
        document.getElementById("metaModel").textContent = cfg.model;
        document.getElementById("inputDate").value = new Date().toISOString().slice(0, 10);
    } catch (e) {
        console.warn("Bootstrap wait:", e);
    }
}

async function launchAnalysis() {
    const ticker = document.getElementById("inputTicker").value.trim();
    const date = document.getElementById("inputDate").value;
    if (!ticker) return alert("Please enter a valid ticker.");

    const btn = document.getElementById("btnRun");
    btn.disabled = true;
    btn.textContent = "Deploying Agents...";

    try {
        const res = await request("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ ticker: ticker, analysis_date: date })
        });
        activeJobId = res.job_id;
        document.getElementById("metricJobId").textContent = activeJobId.slice(0, 8) + "...";
        poll();
    } catch (e) {
        alert("Launch Failed: " + e.message);
        btn.disabled = false;
        btn.textContent = "Run Multi-Agent Engine";
    }
}

async function poll() {
    if (!activeJobId) return;
    try {
        const job = await request(`/api/jobs/${activeJobId}`);
        renderJob(job);
        if (job.status === "running" || job.status === "queued") {
            pollHandle = setTimeout(poll, 1200);
        } else {
            document.getElementById("btnRun").disabled = false;
            document.getElementById("btnRun").textContent = "Run Multi-Agent Engine";
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
    document.getElementById("progressValue").textContent = `${job.progress}%`;
    document.getElementById("progressBar").style.width = `${job.progress}%`;
    document.getElementById("progressPhaseText").textContent = `Pipeline Status: ${job.current_phase}`;

    const dot = document.getElementById("statusDot");
    if (job.status === "running") {
        dot.style.background = "var(--accent-blue)";
        dot.style.boxShadow = "0 0 10px var(--accent-blue)";
    } else if (job.status === "completed") {
        dot.style.background = "var(--accent-emerald)";
        dot.style.boxShadow = "0 0 10px var(--accent-emerald)";
    } else if (job.status === "failed") {
        dot.style.background = "var(--accent-rose)";
        dot.style.boxShadow = "0 0 10px var(--accent-rose)";
    }

    if (job.events) {
        document.getElementById("eventsFeed").innerHTML = job.events.map(ev => `
            <div class="log-entry">
                <span class="log-time">${new Date(ev.time).toLocaleTimeString()}</span>
                <span class="log-source">[${escapeHtml(ev.agent)}]</span>
                <span class="log-msg">${escapeHtml(ev.message)}</span>
            </div>
        `).join("");
        document.getElementById("eventsFeed").scrollTop = document.getElementById("eventsFeed").scrollHeight;
    }

    if (job.debug_logs) {
        currentLogs = job.debug_logs;
        document.getElementById("debugLogsFeed").innerHTML = job.debug_logs.map(l => `
            <div class="log-entry">
                <span class="log-time">${new Date(l.time).toLocaleTimeString()}</span>
                <span class="log-level ${l.level}">${l.level}</span>
                <span class="log-source">${escapeHtml(l.source)}</span>
                <span class="log-msg">${escapeHtml(l.message)}</span>
            </div>
        `).join("");
    }

    if (job.reports) {
        for (const [key, val] of Object.entries(job.reports)) {
            const elId = {
                market: "repMarket",
                fundamentals: "repFundamentals",
                technical: "repTechnical",
                sentiment: "repSentiment",
                bull_case: "repBull",
                bear_case: "repBear",
            }[key];
            if (elId && val) {
                document.getElementById(elId).innerHTML = `<pre>${escapeHtml(typeof val === "object" ? JSON.stringify(val, null, 2) : val)}</pre>`;
            }
        }
    }

    if (job.result && job.result.decision) {
        document.getElementById("decisionView").innerHTML = `
            <div class="decision-banner">
                <div class="decision-title">TRADE MANDATE: ${escapeHtml(JSON.stringify(job.result.decision.action || "COMPLETE"))}</div>
                <p style="margin-top: 8px; color: var(--text-muted);">Executed by Portfolio Manager Node at ${new Date(job.finished_at).toLocaleString()}</p>
            </div>
            <pre>${escapeHtml(JSON.stringify(job.result.decision, null, 2))}</pre>
        `;
    } else if (job.status === "failed") {
        document.getElementById("decisionView").innerHTML = `
            <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid var(--accent-rose); border-radius: 12px; padding: 20px;">
                <h3 style="color: var(--accent-rose);">Execution Terminated</h3>
                <p style="margin-top: 6px; font-size: 13px;">${escapeHtml(job.error?.message || "Unknown Failure")}</p>
                <pre style="margin-top: 12px;">${escapeHtml(job.error?.traceback || "")}</pre>
            </div>
        `;
    }
}

function downloadLogs() {
    if (!currentLogs.length) return alert("No logs recorded.");
    const blob = new Blob([JSON.stringify(currentLogs, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `tradingagents_debug_${activeJobId || "session"}.json`;
    a.click();
}

function escapeHtml(str) {
    return String(str).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}

init();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.exception_handler(401)
async def auth_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=401, content={"error": "Unauthorized access"})
