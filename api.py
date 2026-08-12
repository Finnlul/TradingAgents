from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


app = FastAPI(title="TradingAgents API")


class AnalysisRequest(BaseModel):
    ticker: str
    analysis_date: str | None = None


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "TradingAgents API"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    ticker = request.ticker.strip().upper()

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Ticker is required"
        )

    analysis_date = request.analysis_date or date.today().isoformat()

    try:
        config = DEFAULT_CONFIG.copy()

        ta = TradingAgentsGraph(
            debug=True,
            config=config
        )

        _, decision = ta.propagate(
            ticker,
            analysis_date
        )

        return {
            "ticker": ticker,
            "analysis_date": analysis_date,
            "decision": decision
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
