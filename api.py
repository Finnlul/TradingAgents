import os
from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


app = FastAPI(
    title="TradingAgents API",
    version="1.0.0",
)


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=30)
    analysis_date: str | None = None


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "TradingAgents API",
        "version": "1.0.0",
        "endpoints": ["/health", "/analyze"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "tradingagents": "ready",
        "llm_provider": os.getenv(
            "TRADINGAGENTS_LLM_PROVIDER",
            "openai_compatible",
        ),
        "model": os.getenv(
            "TRADINGAGENTS_QUICK_THINK_LLM",
            "configured",
        ),
    }


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    ticker = request.ticker.strip().upper()

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Ticker darf nicht leer sein.",
        )

    analysis_date = request.analysis_date or date.today().isoformat()

    try:
        # YYYY-MM-DD validieren
        date.fromisoformat(analysis_date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="analysis_date muss YYYY-MM-DD sein.",
        )

    try:
        config = DEFAULT_CONFIG.copy()

        # Explizit setzen, damit Render/ENV sicher verwendet wird.
        config["llm_provider"] = os.getenv(
            "TRADINGAGENTS_LLM_PROVIDER",
            "openai_compatible",
        )

        config["deep_think_llm"] = os.getenv(
            "TRADINGAGENTS_DEEP_THINK_LLM",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        )

        config["quick_think_llm"] = os.getenv(
            "TRADINGAGENTS_QUICK_THINK_LLM",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        )

        backend_url = os.getenv("TRADINGAGENTS_LLM_BACKEND_URL")

        if backend_url:
            config["backend_url"] = backend_url

        # Free Render: nicht unnötig viele parallele Prozesse.
        config["max_debate_rounds"] = int(
            os.getenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1")
        )

        config["max_risk_discuss_rounds"] = int(
            os.getenv("TRADINGAGENTS_MAX_RISK_DISCUSS_ROUNDS", "1")
        )

        ta = TradingAgentsGraph(
            debug=False,
            config=config,
        )

        _, decision = ta.propagate(
            ticker,
            analysis_date,
        )

        return {
            "success": True,
            "ticker": ticker,
            "analysis_date": analysis_date,
            "decision": decision,
        }

    except Exception as exc:
        # Keine Secrets/API-Keys in die Antwort schreiben.
        error_message = str(exc)

        for secret_name in (
            "OPENAI_COMPATIBLE_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            secret = os.getenv(secret_name)
            if secret:
                error_message = error_message.replace(secret, "***")

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": error_message,
            },
        )
