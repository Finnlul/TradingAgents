import os
import threading
import time
import uuid
from datetime import date
from html import escape

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


app = FastAPI(
    title="TradingAgents Web",
    version="2.0.0",
)

# In-memory jobs.
# Ideal für einen einzelnen kostenlosen Render-Web-Service.
JOBS = {}
JOBS_LOCK = threading.Lock()


MODEL = os.getenv(
    "TRADINGAGENTS_QUICK_THINK_LLM",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
)

DEEP_MODEL = os.getenv(
    "TRADINGAGENTS_DEEP_THINK_LLM",
    MODEL,
)

PROVIDER = os.getenv(
    "TRADINGAGENTS_LLM_PROVIDER",
    "openai_compatible",
)


class AnalysisRequest(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=30)
    analysis_date: str | None = None


def create_job(ticker: str, analysis_date: str):
    job_id = uuid.uuid4().hex

    job = {
        "id": job_id,
        "ticker": ticker,
        "analysis_date": analysis_date,
        "status": "queued",
        "started_at": None,
        "finished_at": None,
        "elapsed": 0,
        "decision": None,
        "error": None,
    }

    with JOBS_LOCK:
        JOBS[job_id] = job

    return job


def update_job(job_id, **values):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)


def get_job(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def build_config():
    config = DEFAULT_CONFIG.copy()

    config["llm_provider"] = PROVIDER
    config["deep_think_llm"] = DEEP_MODEL
    config["quick_think_llm"] = MODEL

    backend_url = os.getenv("TRADINGAGENTS_LLM_BACKEND_URL")

    if backend_url:
        config["backend_url"] = backend_url

    # Free Render: bewusst konservativ.
    config["max_debate_rounds"] = int(
        os.getenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "1")
    )

    config["max_risk_discuss_rounds"] = int(
        os.getenv("TRADINGAGENTS_MAX_RISK_DISCUSS_ROUNDS", "1")
    )

    return config


def run_analysis(job_id, ticker, analysis_date):
    started = time.time()

    update_job(
        job_id,
        status="running",
        started_at=time.time(),
    )

    try:
        config = build_config()

        ta = TradingAgentsGraph(
            debug=False,
            config=config,
        )

        final_state, decision = ta.propagate(
            ticker,
            analysis_date,
        )

        elapsed = round(time.time() - started, 1)

        # final_state kann je nach TradingAgents-Version
        # unterschiedlich strukturiert sein.
        state_data = {}

        if isinstance(final_state, dict):
            for key, value in final_state.items():
                try:
                    # JSON-freundlich machen.
                    if isinstance(value, (str, int, float, bool, type(None))):
                        state_data[key] = value
                    else:
                        state_data[key] = str(value)
                except Exception:
                    pass

        update_job(
            job_id,
            status="completed",
            finished_at=time.time(),
            elapsed=elapsed,
            decision=str(decision),
            state=state_data,
        )

    except Exception as exc:
        elapsed = round(time.time() - started, 1)

        message = str(exc)

        # Niemals API-Keys in die Weboberfläche schreiben.
        for name in (
            "OPENAI_COMPATIBLE_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            secret = os.getenv(name)

            if secret:
                message = message.replace(secret, "***")

        update_job(
            job_id,
            status="error",
            finished_at=time.time(),
            elapsed=elapsed,
            error=message,
        )


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        """
<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TradingAgents</title>

<style>
:root{
    --bg:#070b14;
    --panel:#0d1422;
    --panel2:#111a2b;
    --border:#1e2b42;
    --text:#f3f7ff;
    --muted:#8998ad;
    --blue:#5b8cff;
    --cyan:#36d9ff;
    --green:#30d69b;
    --red:#ff6577;
    --yellow:#ffc857;
    --shadow:0 20px 70px rgba(0,0,0,.35);
}

*{box-sizing:border-box}

body{
    margin:0;
    min-height:100vh;
    color:var(--text);
    background:
      radial-gradient(circle at 15% 0%,#13264d 0,transparent 32%),
      radial-gradient(circle at 90% 10%,#102e35 0,transparent 30%),
      var(--bg);
    font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}

.container{
    width:min(1180px,calc(100% - 32px));
    margin:auto;
}

header{
    padding:28px 0 18px;
    display:flex;
    justify-content:space-between;
    align-items:center;
}

.brand{
    display:flex;
    align-items:center;
    gap:13px;
}

.logo{
    width:42px;
    height:42px;
    border-radius:13px;
    display:grid;
    place-items:center;
    background:linear-gradient(135deg,var(--blue),var(--cyan));
    color:#06101b;
    font-weight:900;
    box-shadow:0 8px 30px rgba(91,140,255,.3);
}

.brand h1{
    margin:0;
    font-size:20px;
}

.brand p{
    margin:3px 0 0;
    color:var(--muted);
    font-size:12px;
}

.badge{
    border:1px solid var(--border);
    background:rgba(13,20,34,.7);
    border-radius:999px;
    padding:8px 12px;
    color:#aebbd0;
    font-size:12px;
}

.hero{
    padding:44px 0 30px;
    max-width:850px;
}

.hero h2{
    font-size:clamp(38px,6vw,66px);
    line-height:.98;
    letter-spacing:-3px;
    margin:0 0 18px;
}

.gradient{
    background:linear-gradient(90deg,#fff,#7fa6ff,#47e3ff);
    -webkit-background-clip:text;
    color:transparent;
}

.hero p{
    color:var(--muted);
    font-size:17px;
    line-height:1.7;
    max-width:720px;
}

.panel{
    background:rgba(13,20,34,.82);
    border:1px solid var(--border);
    border-radius:24px;
    box-shadow:var(--shadow);
    backdrop-filter:blur(18px);
}

.controls{
    padding:20px;
    display:grid;
    grid-template-columns:1fr 190px 180px;
    gap:12px;
}

input,button{
    font:inherit;
}

input{
    width:100%;
    color:white;
    background:#080f1d;
    border:1px solid #253550;
    border-radius:14px;
    padding:15px 16px;
    outline:none;
}

input:focus{
    border-color:var(--blue);
    box-shadow:0 0 0 3px rgba(91,140,255,.12);
}

button{
    border:0;
    border-radius:14px;
    padding:15px 18px;
    font-weight:800;
    cursor:pointer;
    color:#06101b;
    background:linear-gradient(135deg,#6b96ff,#3be0ff);
    box-shadow:0 10px 30px rgba(70,150,255,.2);
}

button:disabled{
    opacity:.5;
    cursor:not-allowed;
}

.secondary{
    color:#c9d4e7;
    background:#111c2e;
    border:1px solid #243650;
    box-shadow:none;
}

.status{
    margin-top:14px;
    padding:18px;
    display:none;
}

.status.show{
    display:block;
}

.status-head{
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:10px;
}

.status-title{
    font-weight:800;
}

.status-sub{
    color:var(--muted);
    font-size:13px;
    margin-top:5px;
}

.dot{
    width:10px;
    height:10px;
    border-radius:50%;
    background:var(--yellow);
    box-shadow:0 0 18px var(--yellow);
    display:inline-block;
    margin-right:8px;
}

.dot.done{
    background:var(--green);
    box-shadow:0 0 18px var(--green);
}

.dot.error{
    background:var(--red);
    box-shadow:0 0 18px var(--red);
}

.progress{
    height:6px;
    margin-top:16px;
    background:#07101e;
    border-radius:999px;
    overflow:hidden;
}

.progress div{
    width:30%;
    height:100%;
    border-radius:999px;
    background:linear-gradient(90deg,var(--blue),var(--cyan));
    animation:move 1.3s infinite ease-in-out;
}

@keyframes move{
    0%{transform:translateX(-120%)}
    100%{transform:translateX(430%)}
}

.grid{
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:12px;
    margin:18px 0;
}

.agent{
    padding:17px;
    min-height:125px;
}

.agent-icon{
    font-size:20px;
}

.agent strong{
    display:block;
    margin-top:10px;
}

.agent span{
    display:block;
    color:var(--muted);
    font-size:12px;
    margin-top:5px;
    line-height:1.5;
}

.result{
    display:none;
    margin:22px 0 70px;
}

.result.show{
    display:block;
}

.result-top{
    padding:24px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    gap:20px;
}

.result-title{
    color:var(--muted);
    font-size:12px;
    text-transform:uppercase;
    letter-spacing:1.5px;
}

.decision{
    margin-top:5px;
    font-size:40px;
    font-weight:900;
}

.decision.buy{color:var(--green)}
.decision.sell{color:var(--red)}
.decision.hold{color:var(--yellow)}

.meta{
    color:var(--muted);
    font-size:13px;
}

pre{
    white-space:pre-wrap;
    word-break:break-word;
    margin:0;
    padding:22px;
    color:#d9e4f5;
    line-height:1.65;
    font-size:13px;
    background:#080f1b;
    border-top:1px solid var(--border);
    border-radius:0 0 24px 24px;
}

.footer{
    color:#66758b;
    text-align:center;
    font-size:12px;
    padding:25px 0 50px;
}

@media(max-width:850px){
    .controls{grid-template-columns:1fr}
    .grid{grid-template-columns:repeat(2,1fr)}
    .result-top{flex-direction:column;align-items:flex-start}
}

@media(max-width:500px){
    .container{width:min(100% - 20px,1180px)}
    .grid{grid-template-columns:1fr}
    .hero{padding-top:25px}
    .hero h2{letter-spacing:-2px}
}
</style>
</head>

<body>

<div class="container">

<header>
    <div class="brand">
        <div class="logo">TA</div>
        <div>
            <h1>TradingAgents</h1>
            <p>Multi-Agent Financial Research</p>
        </div>
    </div>

    <div class="badge">● Online</div>
</header>

<section class="hero">
    <h2>
        Research a stock with
        <span class="gradient">multiple AI agents.</span>
    </h2>

    <p>
        Fundamental, sentiment, news and technical analysis are combined
        with bull/bear research, trading analysis and risk management.
    </p>
</section>

<section class="panel">

    <div class="controls">

        <input
            id="ticker"
            placeholder="Ticker — z.B. NVDA, AAPL, BTC-USD"
            value="NVDA"
            autocomplete="off"
        >

        <input
            id="analysisDate"
            type="date"
        >

        <button id="start" onclick="startAnalysis()">
            Analyse starten
        </button>

    </div>

</section>

<section id="status" class="panel status">

    <div class="status-head">
        <div>
            <div class="status-title">
                <span id="dot" class="dot"></span>
                <span id="statusTitle">Analyse wird vorbereitet…</span>
            </div>

            <div id="statusSub" class="status-sub">
                TradingAgents wird gestartet.
            </div>
        </div>

        <div id="timer" class="meta">0.0s</div>
    </div>

    <div class="progress">
        <div></div>
    </div>

</section>

<section class="grid">

    <div class="panel agent">
        <div class="agent-icon">◈</div>
        <strong>Fundamentals</strong>
        <span>Unternehmensdaten, Bewertung und finanzielle Qualität.</span>
    </div>

    <div class="panel agent">
        <div class="agent-icon">◉</div>
        <strong>Sentiment</strong>
        <span>Marktstimmung, Nachrichten und Social Signals.</span>
    </div>

    <div class="panel agent">
        <div class="agent-icon">⌁</div>
        <strong>News</strong>
        <span>Aktuelle Nachrichten und Makroeinflüsse.</span>
    </div>

    <div class="panel agent">
        <div class="agent-icon">⌁</div>
        <strong>Technical</strong>
        <span>Technische Indikatoren und Marktstruktur.</span>
    </div>

    <div class="panel agent">
        <div class="agent-icon">▲</div>
        <strong>Bull Research</strong>
        <span>Argumente für die positive Marktthese.</span>
    </div>

    <div class="panel agent">
        <div class="agent-icon">▼</div>
        <strong>Bear Research</strong>
        <span>Gegenargumente und Risiko-Szenarien.</span>
    </div>

    <div class="panel agent">
        <div class="agent-icon">◆</div>
        <strong>Trader</strong>
        <span>Führt die Research-Ergebnisse zusammen.</span>
    </div>

    <div class="panel agent">
        <div class="agent-icon">盾</div>
        <strong>Risk & Portfolio</strong>
        <span>Bewertet Risiko und finale Entscheidung.</span>
    </div>

</section>

<section id="result" class="panel result">

    <div class="result-top">

        <div>
            <div class="result-title">Final decision</div>
            <div id="decision" class="decision">—</div>
            <div id="resultMeta" class="meta"></div>
        </div>

        <button class="secondary" onclick="copyResult()">
            Ergebnis kopieren
        </button>

    </div>

    <pre id="output"></pre>

</section>

<div class="footer">
    TradingAgents • Research tool • AI-generated results are not financial advice
</div>

</div>

<script>

let currentJob = null;
let timerInterval = null;
let startedAt = null;

function today(){
    const d = new Date();
    const month = String(d.getMonth()+1).padStart(2,"0");
    const day = String(d.getDate()).padStart(2,"0");
    return `${d.getFullYear()}-${month}-${day}`;
}

document.getElementById("analysisDate").value = today();

async function startAnalysis(){

    const ticker =
        document.getElementById("ticker").value.trim().toUpperCase();

    const analysisDate =
        document.getElementById("analysisDate").value;

    if(!ticker){
        alert("Bitte einen Ticker eingeben.");
        return;
    }

    const button = document.getElementById("start");
    button.disabled = true;
    button.textContent = "Analyse läuft…";

    document.getElementById("result").classList.remove("show");

    const status = document.getElementById("status");
    status.classList.add("show");

    setStatus(
        "running",
        "Analyse läuft…",
        `${ticker} wird von TradingAgents analysiert.`
    );

    startedAt = performance.now();

    timerInterval = setInterval(()=>{
        const seconds =
            (performance.now() - startedAt) / 1000;

        document.getElementById("timer").textContent =
            `${seconds.toFixed(1)}s`;
    },100);

    try{

        const response = await fetch("/analyze/start",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                ticker:ticker,
                analysis_date:analysisDate
            })
        });

        const data = await response.json();

        if(!response.ok){
            throw new Error(data.detail || "Analyse konnte nicht gestartet werden.");
        }

        currentJob = data.job_id;

        pollJob();

    }catch(error){

        finishTimer();

        setStatus(
            "error",
            "Fehler",
            error.message
        );

        button.disabled = false;
        button.textContent = "Analyse starten";
    }
}


async function pollJob(){

    if(!currentJob) return;

    try{

        const response =
            await fetch(`/analyze/status/${currentJob}`);

        const job = await response.json();

        if(job.status === "queued"){

            setStatus(
                "running",
                "Warteschlange",
                "Analyse wird gestartet…"
            );

        }else if(job.status === "running"){

            setStatus(
                "running",
                "TradingAgents arbeitet…",
                "Mehrere spezialisierte Agenten analysieren das Wertpapier."
            );

        }else if(job.status === "completed"){

            finishTimer();

            setStatus(
                "done",
                "Analyse abgeschlossen",
                `${job.ticker} wurde erfolgreich analysiert.`
            );

            showResult(job);

            const button = document.getElementById("start");
            button.disabled = false;
            button.textContent = "Neue Analyse";

            return;

        }else if(job.status === "error"){

            finishTimer();

            setStatus(
                "error",
                "Analyse fehlgeschlagen",
                job.error || "Unbekannter Fehler."
            );

            const button = document.getElementById("start");
            button.disabled = false;
            button.textContent = "Erneut versuchen";

            return;
        }

        setTimeout(pollJob,1500);

    }catch(error){

        finishTimer();

        setStatus(
            "error",
            "Verbindungsfehler",
            error.message
        );

        const button = document.getElementById("start");
        button.disabled = false;
        button.textContent = "Erneut versuchen";
    }
}


function setStatus(type,title,sub){

    const dot = document.getElementById("dot");

    dot.className = "dot";

    if(type === "done")
        dot.classList.add("done");

    if(type === "error")
        dot.classList.add("error");

    document.getElementById("statusTitle").textContent = title;
    document.getElementById("statusSub").textContent = sub;
}


function finishTimer(){

    if(timerInterval){
        clearInterval(timerInterval);
        timerInterval = null;
    }
}


function showResult(job){

    const result =
        document.getElementById("result");

    const decision =
        document.getElementById("decision");

    let text =
        String(job.decision || "Keine Entscheidung");

    const upper = text.toUpperCase();

    decision.className = "decision";

    if(upper.includes("BUY"))
        decision.classList.add("buy");
    else if(upper.includes("SELL"))
        decision.classList.add("sell");
    else
        decision.classList.add("hold");

    if(upper.includes("BUY"))
        decision.textContent = "BUY";
    else if(upper.includes("SELL"))
        decision.textContent = "SELL";
    else if(upper.includes("HOLD"))
        decision.textContent = "HOLD";
    else
        decision.textContent = "COMPLETED";

    document.getElementById("resultMeta").textContent =
        `${job.ticker} • ${job.analysis_date} • ${job.elapsed}s • ${PROVIDER_LABEL}`;

    document.getElementById("output").textContent =
        text;

    result.classList.add("show");

    result.scrollIntoView({
        behavior:"smooth",
        block:"start"
    });
}


function copyResult(){

    const text =
        document.getElementById("output").textContent;

    navigator.clipboard.writeText(text)
        .then(()=>{
            alert("Ergebnis kopiert.");
        });
}


const PROVIDER_LABEL =
    "OpenAI-compatible / Nemotron";

</script>

</body>
</html>
"""
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "tradingagents": "ready",
        "llm_provider": PROVIDER,
        "model": MODEL,
    }


@app.post("/analyze/start")
def start_analysis(request: AnalysisRequest):

    ticker = request.ticker.strip().upper()

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Ticker darf nicht leer sein.",
        )

    analysis_date = request.analysis_date or date.today().isoformat()

    try:
        date.fromisoformat(analysis_date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="analysis_date muss YYYY-MM-DD sein.",
        )

    job = create_job(
        ticker,
        analysis_date,
    )

    thread = threading.Thread(
        target=run_analysis,
        args=(
            job["id"],
            ticker,
            analysis_date,
        ),
        daemon=True,
    )

    thread.start()

    return {
        "success": True,
        "job_id": job["id"],
        "ticker": ticker,
        "analysis_date": analysis_date,
    }


@app.get("/analyze/status/{job_id}")
def analysis_status(job_id: str):

    job = get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail="Analyse nicht gefunden.",
        )

    return job


@app.post("/analyze")
def analyze_sync(request: AnalysisRequest):

    ticker = request.ticker.strip().upper()

    if not ticker:
        raise HTTPException(
            status_code=400,
            detail="Ticker darf nicht leer sein.",
        )

    analysis_date = request.analysis_date or date.today().isoformat()

    try:
        date.fromisoformat(analysis_date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="analysis_date muss YYYY-MM-DD sein.",
        )

    try:

        config = build_config()

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
            "decision": str(decision),
        }

    except Exception as exc:

        message = str(exc)

        for name in (
            "OPENAI_COMPATIBLE_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            secret = os.getenv(name)

            if secret:
                message = message.replace(secret, "***")

        raise HTTPException(
            status_code=500,
            detail=message,
        )
