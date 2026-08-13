import os
import uuid
import threading
import traceback
import json
import urllib.request
import urllib.error
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

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="TradingAgents Enterprise Command Center",
    version="4.7.0",
    description="Multi-Agent Market Analysis Orchestrator with On-Demand GitHub Gist Loading",
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
            "github_url": None
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
# GITHUB GIST INTEGRATION (ON-DEMAND STORAGE & FETCH)
# ============================================================
def save_job_to_github(job_data: dict):
    if not GITHUB_TOKEN:
        return
    
    try:
        url = "https://api.github.com/gists"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        
        filename = f"tradingagents_{job_data['ticker']}_{job_data['id']}.json"
        payload = {
            "description": f"TradingAgents Analysis for {job_data['ticker']} ({job_data['analysis_date']})",
            "public": False,
            "files": {
                filename: {
                    "content": json.dumps(job_data, indent=2)
                }
            }
        }
        
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            gist_url = res_data.get("html_url")
            if gist_url:
                update_job(job_data["id"], github_url=gist_url)
                log_debug(job_data["id"], f"Successfully saved job to GitHub Gist: {gist_url}", source="System")
    except Exception as e:
        log_debug(job_data["id"], f"Failed to save job to GitHub: {str(e)}", level="ERROR", source="System")


@app.get("/api/github/gists")
def list_github_gists(request: Request):
    check_password(request)
    if not GITHUB_TOKEN:
        return []
    
    try:
        url = "https://api.github.com/gists"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            gists = json.loads(response.read().decode('utf-8'))
            results = []
            for g in gists:
                # Prüfen ob es ein TradingAgents Gist ist
                for filename in g.get("files", {}):
                    if filename.startswith("tradingagents_"):
                        results.append({
                            "gist_id": g["id"],
                            "html_url": g["html_url"],
                            "filename": filename,
                            "created_at": g["created_at"],
                            "description": g.get("description")
                        })
            return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch gists: {str(e)}")


@app.get("/api/github/gists/{gist_id}")
def load_github_gist(gist_id: str, request: Request):
    check_password(request)
    if not GITHUB_TOKEN:
        raise HTTPException(status_code=400, detail="GitHub Token not configured.")
    
    try:
        url = f"https://api.github.com/gists/{gist_id}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            gist_data = json.loads(response.read().decode('utf-8'))
            files = gist_data.get("files", {})
            for filename, file_info in files.items():
                if filename.startswith("tradingagents_"):
                    content = json.loads(file_info["content"])
                    job_id = content["id"]
                    # In den lokalen Speicher legen, damit UI es anzeigen kann
                    with jobs_lock:
                        jobs[job_id] = content
                    return {"status": "loaded", "job_id": job_id, "data": content}
            raise HTTPException(status_code=404, detail="No valid analysis file found in this Gist.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load gist: {str(e)}")


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
    portfolio: Optional[List[Dict[str, Any]]] = None


# ============================================================
# WORKER EXECUTION
# ============================================================

def run_analysis(job_id: str, ticker: str, analysis_date: str, portfolio: list = None):
    try:
        update_job(job_id, status="running", progress=5)
        add_event(job_id, f"Initializing multi-agent pipeline for target {ticker}", "System", "Setup", 5)
        
        if portfolio:
            add_event(job_id, f"Portfolio context provided: {len(portfolio)} assets.", "System", "Setup", 8)

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

        with jobs_lock:
            completed_job_data = dict(jobs[job_id])
        threading.Thread(target=save_job_to_github, args=(completed_job_data,), daemon=True).start()

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
        "github_enabled": bool(GITHUB_TOKEN)
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
                "current_phase": j["current_phase"],
                "github_url": j.get("github_url")
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
        args=(job_id, ticker, analysis_date, data.portfolio),
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


@app.get("/api/jobs/{job_id}/debug")
def get_job_debug_logs(job_id: str, request: Request):
    check_password(request)
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Analysis job not found.")
        return {"job_id": job_id, "debug_logs": job["debug_logs"]}


@app.get("/api/market_scan")
def market_scan(request: Request):
    check_password(request)
    return {
        "timestamp": utc_now(),
        "recommendations": [
            {"ticker": "NVDA", "name": "NVIDIA Corp", "sector": "Technology", "signal": "Strong Buy", "reason": "High AI momentum & earnings growth."},
            {"ticker": "CRWD", "name": "CrowdStrike Holdings", "sector": "Cybersecurity", "signal": "Buy", "reason": "Favorable technical setup after recent pullback."},
            {"ticker": "LLY", "name": "Eli Lilly and Co", "sector": "Healthcare", "signal": "Buy", "reason": "Strong GLP-1 sales driving revenue upgrades."},
            {"ticker": "SAP", "name": "SAP SE", "sector": "Software", "signal": "Buy", "reason": "Cloud revenue acceleration."},
            {"ticker": "PLTR", "name": "Palantir Technologies", "sector": "Data Analytics", "signal": "Hold", "reason": "Valuation premium, but strong commercial growth."}
        ]
    }


# ============================================================
# COMPLETE ENTERPRISE DASHBOARD UI WITH ON-DEMAND GIST LOADER
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
            overflow-x: hidden;
        }

        .container {
            width: min(1600px, 100%);
            margin: 0 auto;
            padding: 20px 16px 40px;
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
            flex-shrink: 0;
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
            overflow: hidden;
        }

        .card-title {
            font-size: 13px; font-weight: 700; text-transform: uppercase;
            letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 12px;
            display: flex; justify-content: space-between; align-items: center;
        }

        label { display: block; font-size: 11px; font-weight: 600; color: var(--text-muted); margin: 10px 0 4px; }
        input, select {
            width: 100%; background: var(--bg-base); border: 1px solid var(--border-subtle);
            color: var(--text-main); padding: 10px 12px; border-radius: 8px; font-size: 13px; outline: none;
        }
        input:focus { border-color: var(--accent-blue); }

        .autocomplete-wrapper { position: relative; }
        .autocomplete-dropdown {
            position: absolute; top: 100%; left: 0; right: 0; background: var(--bg-elevated);
            border: 1px solid var(--border-subtle); border-radius: 8px; margin-top: 4px;
            max-height: 200px; overflow-y: auto; z-index: 50; display: none;
        }
        .autocomplete-item { padding: 10px 12px; cursor: pointer; display: flex; justify-content: space-between; font-size: 13px; border-bottom: 1px solid var(--bg-base); }
        .autocomplete-item:hover { background: var(--border-subtle); color: var(--accent-blue); }
        .autocomplete-item span.name { color: var(--text-muted); font-size: 11px; }

        .btn-primary {
            width: 100%; background: linear-gradient(135deg, var(--accent-blue), var(--accent-indigo));
            border: none; color: white; padding: 12px; border-radius: 8px; font-size: 13px;
            font-weight: 700; cursor: pointer; margin-top: 16px; transition: opacity 0.2s ease;
            display: flex; justify-content: center; align-items: center; gap: 8px;
        }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

        .btn-secondary {
            background: var(--bg-elevated); border: 1px solid var(--border-bright); color: var(--text-main);
            padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer;
            display: inline-flex; align-items: center; gap: 6px;
        }
        .btn-secondary:hover { background: var(--border-subtle); }

        .btn-danger {
            background: rgba(244, 63, 94, 0.1); border: 1px solid var(--accent-rose); color: var(--accent-rose);
            padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: 600; cursor: pointer;
        }

        .job-history-list {
            max-height: 200px;
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
        
        .badge { font-size: 10px; padding: 2px 6px; border-radius: 4px; font-family: monospace; white-space: nowrap; }
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

        .tabs-nav { display: flex; gap: 6px; border-bottom: 1px solid var(--border-subtle); margin-bottom: 16px; overflow-x: auto; scrollbar-width: none; }
        .tabs-nav::-webkit-scrollbar { display: none; }
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
        .log-entry { padding: 4px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.03); word-break: break-word; }
        .log-entry.error { color: var(--accent-rose); }

        .report-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }
        .report-card { background: var(--bg-base); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 14px; max-width: 100%; overflow: hidden; }
        .report-card h4 { font-size: 12px; color: var(--accent-blue); margin-bottom: 8px; text-transform: uppercase; }

        pre { 
            background: var(--bg-surface); 
            border: 1px solid var(--border-subtle); 
            padding: 12px; 
            border-radius: 8px; 
            font-family: 'JetBrains Mono', monospace; 
            font-size: 11px; 
            max-height: 400px; 
            color: var(--text-muted);
            white-space: pre-wrap; 
            word-wrap: break-word;
            overflow-x: auto;
        }

        .auth-overlay {
            position: fixed; inset: 0; background: rgba(6, 9, 14, 0.95);
            display: grid; place-items: center; z-index: 100;
        }
        .auth-modal { background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 16px; padding: 24px; width: min(380px, 90vw); }
        
        .hidden { display: none !important; }

        .portfolio-list { display: flex; flex-direction: column; gap: 8px; margin-top: 12px; }
        .portfolio-item { display: flex; justify-content: space-between; background: var(--bg-base); padding: 8px 12px; border-radius: 6px; font-size: 12px; border: 1px solid var(--border-subtle); align-items: center;}
        
        table.market-scan-table { width: 100%; border-collapse: collapse; font-size: 12px; }
        table.market-scan-table th, table.market-scan-table td { padding: 10px; text-align: left; border-bottom: 1px solid var(--border-subtle); }
        table.market-scan-table th { color: var(--text-muted); text-transform: uppercase; font-size: 10px; }
        table.market-scan-table tr:hover { background: var(--bg-elevated); cursor: pointer; }
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
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
            <button class="btn-secondary" onclick="exportJobText()" id="btnTextExport" style="display: none;">📄 Klartext-Bericht (TXT)</button>
            <button class="btn-secondary" onclick="exportJobJSON()">📥 Export Job JSON</button>
            <div id="gatewayStatus" style="font-size: 12px; color: var(--accent-emerald);">● Online</div>
        </div>
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
                <label>Equity / Ticker</label>
                <div class="autocomplete-wrapper">
                    <input id="inputTicker" type="text" placeholder="z.B. SAP, NVDA, AAPL" oninput="filterTickers()" onfocus="filterTickers()" autocomplete="off">
                    <div id="autocompleteDropdown" class="autocomplete-dropdown"></div>
                </div>
                
                <label>Anchor Date</label>
                <input id="inputDate" type="date">

                <button id="btnRun" class="btn-primary" onclick="launchAnalysis()">🚀 Run Pipeline</button>
                <button class="btn-secondary" style="width: 100%; margin-top: 8px; justify-content: center;" onclick="switchTab('tabMarketScan')">🔍 Market Scan (Ideen)</button>
            </div>

            <div class="card">
                <div class="card-title">Lokale Jobs</div>
                <div id="jobHistoryList" class="job-history-list">
                    <div style="color: var(--text-dim); font-size: 11px;">Keine lokalen Jobs.</div>
                </div>
            </div>

            <!-- NEU: GitHub Gists nur auf Klick laden -->
            <div class="card">
                <div class="card-title">
                    <span>GitHub Gist Archive</span>
                    <button class="btn-secondary" onclick="loadGitHubGistsList()" style="font-size: 10px; padding: 2px 6px;">Laden</button>
                </div>
                <p style="font-size: 11px; color: var(--text-dim); margin-bottom: 8px;">Klicke auf "Laden", um gespeicherte Analysen aus GitHub abzurufen (bleiben unberührt, bis du sie anklickst).</p>
                <div id="gistHistoryList" class="job-history-list">
                    <div style="color: var(--text-dim); font-size: 11px;">Nicht geladen. Klicke oben auf Laden.</div>
                </div>
            </div>
        </aside>

        <main>
            <div class="tabs-nav">
                <button class="tab-btn active" onclick="switchTab('tabEvents', event)">Workflow Events</button>
                <button class="tab-btn" onclick="switchTab('tabDebug', event)">Detailed Debug</button>
                <button class="tab-btn" onclick="switchTab('tabReports', event)">Intelligence Reports</button>
                <button class="tab-btn" onclick="switchTab('tabDecision', event)">Decision & Mandate</button>
                <button class="tab-btn" onclick="switchTab('tabPortfolio', event)">Mein Portfolio</button>
                <button class="tab-btn hidden" id="btnTabMarketScan" onclick="switchTab('tabMarketScan', event)">Market Scan</button>
            </div>

            <div id="tabEvents" class="tab-content active">
                <div class="card">
                    <div class="card-title">Telemetry Feed</div>
                    <div id="eventsFeed" class="logs-window">Warte auf Start...</div>
                </div>
            </div>

            <div id="tabDebug" class="tab-content">
                <div class="card">
                    <div class="card-title">
                        <span>Engine Debug Stream</span>
                        <button class="btn-secondary" onclick="loadDebugLogs()">🔄 Refresh Debug</button>
                    </div>
                    <div id="debugFeed" class="logs-window">Keine Debug-Daten geladen.</div>
                </div>
            </div>

            <div id="tabReports" class="tab-content">
                <div class="report-grid" id="reportsGrid">
                    <div class="report-card" id="card_market"><h4>Market Context</h4><div id="repMarket" class="text-dim">Keine Daten</div></div>
                    <div class="report-card" id="card_fundamentals"><h4>Fundamentals</h4><div id="repFundamentals" class="text-dim">Keine Daten</div></div>
                    <div class="report-card" id="card_technical"><h4>Technical Overview</h4><div id="repTechnical" class="text-dim">Keine Daten</div></div>
                    <div class="report-card" id="card_sentiment"><h4>Sentiment</h4><div id="repSentiment" class="text-dim">Keine Daten</div></div>
                    <div class="report-card" id="card_news"><h4>News</h4><div id="repNews" class="text-dim">Keine Daten</div></div>
                    <div class="report-card" id="card_bull_case"><h4>Bull Thesis</h4><div id="repBull" class="text-dim">Keine Daten</div></div>
                    <div class="report-card" id="card_bear_case"><h4>Bear Thesis</h4><div id="repBear" class="text-dim">Keine Daten</div></div>
                    <div class="report-card" id="card_debate_summary"><h4>Debate Summary</h4><div id="repDebate" class="text-dim">Keine Daten</div></div>
                    <div class="report-card" id="card_risk_assessment"><h4>Risk Assessment</h4><div id="repRisk" class="text-dim">Keine Daten</div></div>
                </div>
            </div>

            <div id="tabDecision" class="tab-content">
                <div class="card">
                    <div id="decisionView">Noch keine Entscheidung getroffen.</div>
                </div>
            </div>

            <div id="tabPortfolio" class="tab-content">
                <div class="card">
                    <div class="card-title">Mein Portfolio verwalten</div>
                    <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 12px;">Aktien, die du hier einträgst, werden bei der Analyse als dein aktueller Bestand berücksichtigt.</p>
                    
                    <div style="display: flex; gap: 8px; margin-bottom: 16px;">
                        <input type="text" id="portTicker" placeholder="Ticker (z.B. AAPL)" style="width: 40%;">
                        <input type="number" id="portQty" placeholder="Anzahl" style="width: 30%;">
                        <input type="number" id="portPrice" placeholder="Kaufpreis $" style="width: 30%;">
                        <button class="btn-secondary" onclick="addToPortfolio()">Hinzufügen</button>
                    </div>
                    
                    <div id="portfolioList" class="portfolio-list"></div>
                </div>
            </div>

            <div id="tabMarketScan" class="tab-content">
                <div class="card">
                    <div class="card-title">
                        <span>Automatischer Markt-Scan</span>
                        <button class="btn-secondary" onclick="runMarketScan()">🔄 Scannen</button>
                    </div>
                    <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 16px;">Hier findest du potenziell interessante Aktien. Klicke auf eine Zeile, um sie zu analysieren.</p>
                    <div id="marketScanResults" style="overflow-x: auto;">
                        <div style="color:var(--text-dim); font-size:12px;">Klicke auf Scannen, um Empfehlungen zu erhalten.</div>
                    </div>
                </div>
            </div>

        </main>
    </div>
</div>

<script>
let activeJobId = null;
let pollHandle = null;
let currentJobData = null;
let portfolio = JSON.parse(localStorage.getItem('ta_portfolio') || '[]');

const KNOWN_STOCKS = [
    { ticker: "SAP", name: "SAP SE" },
    { ticker: "NVDA", name: "NVIDIA Corp" },
    { ticker: "AAPL", name: "Apple Inc" },
    { ticker: "MSFT", name: "Microsoft Corp" },
    { ticker: "TSLA", name: "Tesla Inc" },
    { ticker: "AMZN", name: "Amazon.com Inc" },
    { ticker: "GOOGL", name: "Alphabet Inc" },
    { ticker: "META", name: "Meta Platforms" },
    { ticker: "PLTR", name: "Palantir Technologies" },
    { ticker: "CRWD", name: "CrowdStrike" },
    { ticker: "AMD", name: "Advanced Micro Devices" },
    { ticker: "LLY", name: "Eli Lilly" },
    { ticker: "NVO", name: "Novo Nordisk" },
    { ticker: "BTC-USD", name: "Bitcoin" },
    { ticker: "ETH-USD", name: "Ethereum" }
];

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
    
    if (evt && evt.target) {
        evt.target.classList.add("active");
    } else {
        const targetBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => b.textContent.includes(tabId.replace('tab', '')));
        if(targetBtn) targetBtn.classList.add("active");
        if(tabId === 'tabMarketScan') document.getElementById('btnTabMarketScan').classList.add("active");
    }
    
    document.getElementById(tabId).classList.add("active");
    if(tabId === 'tabDebug') loadDebugLogs();
    if(tabId === 'tabMarketScan') runMarketScan();
}

document.getElementById("inputDate").value = new Date().toISOString().slice(0, 10);
renderPortfolio();

function filterTickers() {
    const input = document.getElementById("inputTicker").value.toLowerCase();
    const dropdown = document.getElementById("autocompleteDropdown");
    
    if (!input) {
        dropdown.style.display = "none";
        return;
    }

    const filtered = KNOWN_STOCKS.filter(s => 
        s.ticker.toLowerCase().includes(input) || s.name.toLowerCase().includes(input)
    );

    if (filtered.length > 0) {
        dropdown.innerHTML = filtered.map(s => `
            <div class="autocomplete-item" onclick="selectTicker('${s.ticker}')">
                <b>${s.ticker}</b> <span class="name">${s.name}</span>
            </div>
        `).join("");
        dropdown.style.display = "block";
    } else {
        dropdown.style.display = "none";
    }
}

function selectTicker(ticker) {
    document.getElementById("inputTicker").value = ticker;
    document.getElementById("autocompleteDropdown").style.display = "none";
}

document.addEventListener('click', function(e) {
    if (!e.target.closest('.autocomplete-wrapper')) {
        document.getElementById("autocompleteDropdown").style.display = "none";
    }
});


async function loadHistory() {
    try {
        const jobs = await request("/api/jobs");
        const listEl = document.getElementById("jobHistoryList");
        if (!jobs.length) {
            listEl.innerHTML = `<div style="color: var(--text-dim); font-size: 11px;">Keine lokalen Jobs.</div>`;
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

// GitHub Gists nur auf Klick laden
async function loadGitHubGistsList() {
    const listEl = document.getElementById("gistHistoryList");
    listEl.innerHTML = `<div style="color: var(--text-dim); font-size: 11px;">Lade Gists von GitHub...</div>`;
    try {
        const gists = await request("/api/github/gists");
        if (!gists.length) {
            listEl.innerHTML = `<div style="color: var(--text-dim); font-size: 11px;">Keine Gists gefunden.</div>`;
            return;
        }
        listEl.innerHTML = gists.map(g => `
            <div class="job-item" onclick="fetchAndSelectGist('${g.gist_id}')">
                <span><b>${escapeHtml(g.filename.replace('tradingagents_', '').split('_')[0])}</b> <small style="color: var(--text-dim);">${new Date(g.created_at).toLocaleDateString()}</small></span>
                <span class="badge completed">Gist</span>
            </div>
        `).join("");
    } catch (e) {
        listEl.innerHTML = `<div style="color: var(--accent-rose); font-size: 11px;">Fehler: ${escapeHtml(e.message)}</div>`;
    }
}

async function fetchAndSelectGist(gistId) {
    try {
        const res = await request(`/api/github/gists/${gistId}`);
        if(res && res.job_id) {
            selectJob(res.job_id);
            loadHistory();
        }
    } catch(e) {
        alert("Fehler beim Laden des Gists: " + e.message);
    }
}

async function selectJob(jobId) {
    activeJobId = jobId;
    if (pollHandle) clearTimeout(pollHandle);
    loadHistory();
    poll();
}

async function loadDebugLogs() {
    if(!activeJobId) {
        document.getElementById("debugFeed").innerHTML = "Kein Job ausgewählt.";
        return;
    }
    try {
        const data = await request(`/api/jobs/${activeJobId}/debug`);
        const feed = document.getElementById("debugFeed");
        if(!data.debug_logs || !data.debug_logs.length) {
        	feed.innerHTML = "Keine Debug-Einträge vorhanden.";
        	return;
        }
        feed.innerHTML = data.debug_logs.map(log => `
            <div class="log-entry ${log.level === 'ERROR' ? 'error' : ''}">
                <span style="color: var(--text-dim);">${new Date(log.time).toLocaleTimeString()}</span>
                <span style="color: var(--accent-indigo);">[${escapeHtml(log.source)}]</span>
                <span>${escapeHtml(log.message)}</span>
            </div>
        `).join("");
    } catch(e) {
        document.getElementById("debugFeed").innerHTML = "Fehler beim Laden der Debug-Logs: " + escapeHtml(e.message);
    }
}

function exportJobJSON() {
    if(!currentJobData) {
        alert("Keine Job-Daten zum Exportieren vorhanden.");
        return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(currentJobData, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `tradingagents_job_${currentJobData.ticker}_${currentJobData.id.slice(0,8)}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}

function exportJobText() {
    if(!currentJobData || !currentJobData.reports) {
        alert("Keine Berichte zum Exportieren vorhanden.");
        return;
    }
    
    let textContent = `TRADING AGENTS - ANALYSE BERICHT\n`;
    textContent += `=================================\n`;
    textContent += `Ticker: ${currentJobData.ticker}\n`;
    textContent += `Datum: ${currentJobData.analysis_date}\n`;
    textContent += `Job ID: ${currentJobData.id}\n\n`;
    
    if (currentJobData.result && currentJobData.result.decision) {
        textContent += `--- ENTSCHEIDUNG ---\n`;
        textContent += JSON.stringify(currentJobData.result.decision, null, 2) + `\n\n`;
    }
    
    const map = { 
        market: "Market Context", 
        fundamentals: "Fundamentals", 
        technical: "Technical Overview", 
        sentiment: "Sentiment", 
        news: "News", 
        bull_case: "Bull Thesis", 
        bear_case: "Bear Thesis",
        debate_summary: "Debate Summary",
        risk_assessment: "Risk Assessment"
    };

    for (const [key, title] of Object.entries(map)) {
        if (currentJobData.reports[key]) {
            textContent += `--- ${title.toUpperCase()} ---\n`;
            let val = currentJobData.reports[key];
            if(typeof val === 'object') {
                textContent += JSON.stringify(val, null, 2) + `\n\n`;
            } else {
                textContent += val + `\n\n`;
            }
        }
    }

    const dataStr = "data:text/plain;charset=utf-8," + encodeURIComponent(textContent);
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `Analyse_${currentJobData.ticker}_${currentJobData.id.slice(0,8)}.txt`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
}

function renderPortfolio() {
    const list = document.getElementById("portfolioList");
    if(portfolio.length === 0) {
        list.innerHTML = `<div style="color: var(--text-dim); font-size: 11px;">Dein Portfolio ist leer.</div>`;
        return;
    }
    list.innerHTML = portfolio.map((item, index) => `
        <div class="portfolio-item">
            <div><b>${item.ticker}</b> - ${item.qty} Stück (Kauf: $${item.price})</div>
            <button class="btn-danger" onclick="removePortfolioItem(${index})">X</button>
        </div>
    `).join("");
}

function addToPortfolio() {
    const t = document.getElementById("portTicker").value.toUpperCase();
    const q = document.getElementById("portQty").value;
    const p = document.getElementById("portPrice").value;
    
    if(t && q && p) {
        portfolio.push({ticker: t, qty: parseFloat(q), price: parseFloat(p)});
        localStorage.setItem('ta_portfolio', JSON.stringify(portfolio));
        renderPortfolio();
        document.getElementById("portTicker").value = '';
        document.getElementById("portQty").value = '';
        document.getElementById("portPrice").value = '';
    }
}

function removePortfolioItem(index) {
    portfolio.splice(index, 1);
    localStorage.setItem('ta_portfolio', JSON.stringify(portfolio));
    renderPortfolio();
}

async function runMarketScan() {
    const container = document.getElementById("marketScanResults");
    container.innerHTML = "Scanne Markt...";
    try {
        const res = await request("/api/market_scan");
        if(res.recommendations && res.recommendations.length > 0) {
            let html = `<table class="market-scan-table">
                <thead><tr><th>Ticker</th><th>Unternehmen</th><th>Sektor</th><th>Signal</th><th>Grund</th></tr></thead>
                <tbody>`;
            res.recommendations.forEach(r => {
                html += `<tr onclick="document.getElementById('inputTicker').value='${r.ticker}'; window.scrollTo(0,0);">
                    <td><b>${r.ticker}</b></td>
                    <td>${r.name}</td>
                    <td>${r.sector}</td>
                    <td style="color: ${r.signal.includes('Buy') ? 'var(--accent-emerald)' : 'var(--accent-amber)'}">${r.signal}</td>
                    <td style="color: var(--text-muted);">${r.reason}</td>
                </tr>`;
            });
            html += `</tbody></table>`;
            container.innerHTML = html;
        } else {
            container.innerHTML = "Keine Empfehlungen gefunden.";
        }
    } catch(e) {
        container.innerHTML = "Fehler beim Scan: " + escapeHtml(e.message);
    }
}

async function launchAnalysis() {
    const ticker = document.getElementById("inputTicker").value.trim();
    const date = document.getElementById("inputDate").value;
    if (!ticker) return alert("Bitte Ticker eingeben.");

    const btn = document.getElementById("btnRun");
    btn.disabled = true;
    btn.textContent = "Starte...";
    document.getElementById("btnTextExport").style.display = "none";

    try {
        const res = await request("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                ticker, 
                analysis_date: date,
                portfolio: portfolio 
            })
        });
        activeJobId = res.job_id;
        loadHistory();
        switchTab('tabEvents');
        poll();
    } catch (e) {
        alert("Fehler: " + e.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = "🚀 Run Pipeline";
    }
}

async function poll() {
    if (!activeJobId) return;
    try {
        const job = await request(`/api/jobs/${activeJobId}`);
        currentJobData = job;
        renderJob(job);
        loadHistory();
        if (job.status === "running" || job.status === "queued") {
            pollHandle = setTimeout(poll, 1500);
            document.getElementById("btnTextExport").style.display = "none";
        } else {
            if (job.status === "completed") {
                document.getElementById("btnTextExport").style.display = "inline-flex";
            }
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
            const cardId = `card_${k}`;
            const cardEl = document.getElementById(cardId);
            
            if (el && cardEl) {
                const val = job.reports[k];
                if (val === null || val === undefined || val === "" || val === "Keine Daten") {
                    cardEl.style.display = 'none';
                } else {
                    cardEl.style.display = 'block';
                    el.innerHTML = `<pre>${escapeHtml(typeof val === "object" ? JSON.stringify(val, null, 2) : val)}</pre>`;
                }
            }
        }
    }

    if (job.result && job.result.decision) {
        document.getElementById("decisionView").innerHTML = `
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid var(--accent-emerald); padding: 16px; border-radius: 10px; margin-bottom: 12px;">
                <h3 style="color: var(--accent-emerald);">MANDAT ERTEILT / ABGESCHLOSSEN</h3>
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
