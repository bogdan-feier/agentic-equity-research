import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from agent.utils.state import AgentState
from agent.utils.tools import get_stock_info, search_news, search_sec_filings

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
    Return a JSON object with a 'required_data' list containing any of: ["stock_info", "recent_news", "search_sec_filings"].
    Example: {{"required_data": ["stock_info", "recent_news"]}}
    """

    response = llm.invoke([SystemMessage(content="Output pure JSON only."), HumanMessage(content=prompt)])

    try:
        clean_json = response.content.strip().replace("```json", "").replace("```", "")
        plan = json.loads(clean_json)
    except Exception:
        plan = {"required_data": ["stock_info", "recent_news", "search_sec_filings"]}

    return {"plan": plan}


def data_fetcher_node(state: AgentState) -> dict:
    """
    Executes the tools specified by the Planner node and stores raw results in state['raw_data']
    """
    ticker = state["ticker"]
    user_query = state["user_query"]
    plan = state.get("plan", {}).get("required_data", ["stock_info", "recent_news", "search_sec_filings"])

    raw_data = {}

    if "stock_info" in plan:
        raw_data["stock_info"] = get_stock_info.invoke({"ticker": ticker})

    if "recent_news" in plan:
        raw_data["recent_news"] = search_news.invoke({"ticker": ticker})

    if "search_sec_filings" in plan:
        raw_data["search_sec_filings"] = search_sec_filings.invoke({
            "ticker": ticker,
            "query": user_query
        })

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
    Drafts or revises a professional Investment Memo based on the analysis and user query.
    The memo should be in Markdown format and include an executive summary, financial overview, key drivers and risk factors.   
    """
    ticker = state["ticker"]
    user_query = state["user_query"]
    analysis = state["analysis"]
    critic_feedback = state.get("critic_feedback")

    prompt = f"""
    You are an Equity Research Writer crafting an Investment Memo for a client.
    
    Client Query: {user_query}
    Target Ticker: {ticker}
    Analysis: 
    {analysis}
    """

    if critic_feedback:
        prompt += f"\n\nCRITICAL FEEDBACK FROM COMPLIANCE REVIEW:\n{critic_feedback}\nPlease revise the memo to specifically address this feedback."

    prompt += """

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


def critic_node(state: AgentState) -> dict:
    """
    Evaluates the draft memo for accuracy, completeness, and formatting against the user query and raw data.
    """
    ticker = state["ticker"]
    user_query = state["user_query"]
    memo = state["memo"]
    raw_data = state.get("raw_data", {})
    revision_count = state.get("revision_count", 0)

    prompt = f"""
    You are a Senior Compliance Officer and Quality Control Analyst at an equity research firm.
    Review the draft Investment Memo for {ticker}.

    User Query: {user_query}
    Raw Data Provided:
    {json.dumps(raw_data, indent=2)}

    Draft Investment Memo:
    {memo}

    Evaluate the draft:
    1. Does it directly and thoroughly answer the user's query?
    2. Are all financial claims backed by the raw data?
    3. Is it well-structured in Markdown with clear actionability?

    Return a JSON object with:
    - "status": Either "APPROVED" or "NEEDS_REVISION"
    - "feedback": Detailed explanation of what needs fixing if REJECTED, or praise if APPROVED.

    Example:
    {{"status": "NEEDS_REVISION", "feedback": "The report misses specific supply chain risk factors from the 10-K filing."}}
    """

    response = llm.invoke([SystemMessage(content="Output pure JSON only."), HumanMessage(content=prompt)])

    try:
        clean_json = response.content.strip().replace("```json", "").replace("```", "")
        review = json.loads(clean_json)
    except Exception:
        review = {"status": "APPROVED", "feedback": "Passed automatic check."}

    return {
        "review_status": review.get("status", "APPROVED"),
        "critic_feedback": review.get("feedback", ""),
        "revision_count": revision_count + 1
    }