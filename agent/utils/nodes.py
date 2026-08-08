import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.utils.state import AgentState
from agent.utils.tools import get_stock_info, search_news

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.2)

def planner_node(state: AgentState) -> dict:
    """
    Determines what data is required to answer the user's question based on the target ticker and user query
    """
    ticker = state["ticker"]
    user_query = state["user_query"]

    prompt = f"""
    You are the Lead Investment Strategist.
    Target Ticker: {ticker}
    User Question: {user_query}
    
    Determine what data is required to answer this question. 
    Return a JSON object with a 'required_data' list containing any of: ["stock_info", "recent_news"].
    Example: {{"required_data": ["stock_info", "recent_news"]}}
    """

    response = llm.invoke([SystemMessage(content="Output pure JSON only."), HumanMessage(content=prompt)])

    try:
        clean_json = response.content.strip().replace("```json", "").replace("```", "")
        plan = json.loads(clean_json)
    except Exception:
        plan = {"required_data": ["stock_info", "recent_news"]}

    return {"plan": plan}


def data_fetcher_node(state: AgentState) -> dict:
    """
    Executes the tools specified by the Planner node and stores raw results in state['raw_data']
    """
    ticker = state["ticker"]
    plan = state.get("plan", {}).get("required_data", ["stock_info", "recent_news"])

    raw_data = {}

    if "stock_info" in plan:
        raw_data["stock_info"] = get_stock_info.invoke({"ticker": ticker})

    if "recent_news" in plan:
        raw_data["recent_news"] = search_news.invoke({"ticker": ticker})

    return {"raw_data": raw_data}


def analysis_node(state: AgentState) -> dict:
    """
    Analyzes the raw data fetched by the Data Fetcher node and produces a structured summary
    to identify key financial highlights, bullish factors and bearish factors for the target ticker
    """
    ticker = state["ticker"]
    raw_data = state["raw_data"]

    prompt = f"""
    You are a Financial Analyst specializing in quantitative and qualitative analysis.
    Analyze the raw data below for {ticker}:
    
    {json.dumps(raw_data, indent=2)}
    
    Provide a structured summary containing:
    1. Key Financial Highlights (Price, P/E ratio, Market Cap)
    2. Bullish Factors (Positives / Growth drivers)
    3. Bearish Factors (Risks / Red flags)
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"analysis": response.content}


def draft_report_node(state: AgentState) -> dict:
    """
    Drafts a professional Investment Memo based on the analysis and user query.
    The memo should be in Markdown format and include an executive summary, financial overview, key drivers and risk factors.   
    """
    ticker = state["ticker"]
    user_query = state["user_query"]
    analysis = state["analysis"]

    prompt = f"""
    You are an Equity Research Writer crafting an Investment Memo for a client.
    
    Client Query: {user_query}
    Target Ticker: {ticker}
    Analysis: 
    {analysis}
    
    Write a clear, professional Investment Memo in Markdown format.
    Include:
    - Executive Summary & Verdict (Buy / Hold / Avoid)
    - Financial Overview
    - Key Drivers & Catalysts
    - Risk Factors
    
    Ensure all claims directly reference the provided data.
    """

    response = llm.invoke([HumanMessage(content=prompt)])

    return {"memo": response.content}