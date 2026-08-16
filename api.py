import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from agent.agent import build_graph

app = FastAPI(
    title="Agentic Equity Research API",
    description="An AI-powered equity research pipeline using LangGraph",
    version="1.0.0"
)

class ResearchRequest(BaseModel):
    ticker: str
    query: str

class ResearchResponse(BaseModel):
    ticker: str
    memo: str
    pdf_path: Optional[str] = None
    chart_path: Optional[str] = None


@app.post("/research", response_model=ResearchResponse)
async def generate_research(request: ResearchRequest):
    try:
        graph = build_graph()
        ticker = request.ticker.upper().strip()

        initial_state = {
            "ticker": ticker,
            "user_query": request.query.strip(),
            "revision_count": 0
        }

        result = graph.invoke(initial_state)

        raw_memo = result.get("memo", "")
        if isinstance(raw_memo, list) and len(raw_memo) > 0:
            latest = raw_memo[-1]
            if isinstance(latest, dict):
                memo_text = latest.get("text", str(latest))
            else:
                memo_text = str(latest)
        elif isinstance(raw_memo, dict):
            memo_text = raw_memo.get("text", str(raw_memo))
        else:
            memo_text = str(raw_memo)

        pdf_path = result.get("pdf_path")

        chart_path = result.get("chart_path")
        if not chart_path:
            for potential_path in [
                f"outputs/{ticker}_price_chart.png",
                f"outputs/{ticker}_chart.png",
                "outputs/price_chart.png"
            ]:
                if os.path.exists(potential_path):
                    chart_path = potential_path
                    break

        return ResearchResponse(
            ticker=initial_state["ticker"],
            memo=memo_text,
            pdf_path=pdf_path,
            chart_path=chart_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
