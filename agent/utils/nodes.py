import os
import json
import markdown
import matplotlib
matplotlib.use('Agg')
import yfinance as yf
from weasyprint import HTML
import matplotlib.pyplot as plt
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
    CRITICAL RULE: Be as restrictive as possible. ONLY select a tool if it is absolutely necessary to answer the user's specific query. Do not select tools "just in case".
    
    Available tools:
    - "stock_info": Use ONLY if the user asks about current price, market cap or PE ratio.
    - "recent_news": Use ONLY if the user asks about recent events, today's movement or news.
    - "search_sec_filings": Use ONLY if the user asks about risks, 10-K, SEC or deep fundamental business models.

    Return a pure JSON object with a 'required_data' list.
    Example: {{"required_data": ["search_sec_filings"]}}
    """

    response = llm.invoke([SystemMessage(content="Output pure JSON only. No markdown formatting. No conversational text."), HumanMessage(content=prompt)])

    raw_content = response.content

    if isinstance(raw_content, list) and len(raw_content) > 0:
        if isinstance(raw_content[0], dict):
            text_content = raw_content[0].get("text", str(raw_content[0]))
        else:
            text_content = str(raw_content[0])
    elif isinstance(raw_content, dict):
        text_content = raw_content.get("text", str(raw_content))
    else:
        text_content = str(raw_content)

    try:
        clean_json = text_content.strip().replace("```json", "").replace("```", "")
        plan = json.loads(clean_json)
    except Exception as e:
        print(f"\nPlanner JSON parsing failed ({e}). Defaulting to all tools.")
        plan = {"required_data": ["stock_info", "recent_news", "search_sec_filings"]}

    return {"plan": plan}


def fetch_stock_node(state: AgentState) -> dict:
    """
    Fetches stock data independently.
    """
    print(f"Fetching stock data for {state['ticker']}...")
    return {"stock_data": get_stock_info.invoke({"ticker": state["ticker"]})}


def fetch_news_node(state: AgentState) -> dict:
    """
    Fetches news data independently.
    """
    print(f"Fetching news for {state['ticker']}...")
    return {"news_data": search_news.invoke({"ticker": state["ticker"]})}


def fetch_sec_node(state: AgentState) -> dict:
    """
    Fetches SEC filings independently.
    """
    ticker = state['ticker']
    print(f"Fetching SEC filings for {ticker}...")

    try:
        sec_data = search_sec_filings.invoke({
            "ticker": ticker,
            "query": state["user_query"]
        })

        if not sec_data:
            sec_data = f"No SEC 10-K filing found for {ticker}. Please note this in your report and base your analysis strictly on news and historical price data."

    except Exception as e:
        print(f"SEC fetch failed for {ticker}: {e}")
        sec_data = f"No SEC 10-K filing found for {ticker}. Please note this in your report and base your analysis strictly on news and historical price data."


    return {"sec_data": sec_data}
    

def analysis_node(state: AgentState) -> dict:
    """
    Analyzes the raw data fetched by the Data Fetcher node and produces a structured summary
    to identify key financial highlights, bullish factors and bearish factors for the target ticker
    """
    ticker = state["ticker"]
    raw_data = {
        "stock_info": state.get("stock_data"),
        "recent_news": state.get("news_data"),
        "sec_filings": state.get("sec_data")
    }

    raw_data = {k: v for k, v in raw_data.items() if v is not None}

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
    revision_count = state.get("revision_count", 0)
    raw_data = {
        "stock_info": state.get("stock_data"),
        "recent_news": state.get("news_data"),
        "sec_filings": state.get("sec_data")
    }

    raw_data = {k: v for k, v in raw_data.items() if v is not None}

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

    response = llm.invoke([SystemMessage(content="Output pure JSON only. No markdown formatting. No conversational text."), HumanMessage(content=prompt)])

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


def export_node(state: AgentState) -> dict:
    """
    Generates a financial chart, appends it to the memo and exports everything to a PDF.
    """
    ticker = state["ticker"]
    raw_memo = state.get("memo", "")

    if isinstance(raw_memo, list) and len(raw_memo) > 0:
        latest = raw_memo[-1]
        if isinstance(latest, dict):
            memo = latest.get("text", str(latest))
        else:
            memo = str(latest)
    elif isinstance(raw_memo, dict):
        memo = raw_memo.get("text", str(raw_memo))
    else:
        memo = str(raw_memo)

    print(f"Generating 6-month price chart for {ticker}...")
    os.makedirs("outputs", exist_ok=True)

    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    chart_filename = f"{ticker}_chart.png"
    chart_path = os.path.join("outputs", chart_filename)

    plt.figure(figsize=(10, 5))
    plt.plot(hist.index, hist['Close'], color='#007AFF', linewidth=2)
    plt.title(f"{ticker} - 6 Month Price History", fontsize=14, fontweight='bold')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()

    memo_with_chart = memo + f"\n\n## 6-Month Price Performance\n\n<img src='{chart_filename}' width='100%'>"

    html_content = markdown.markdown(memo_with_chart, extensions=['tables'])

    styled_html = f"""
    <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #333; padding: 40px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #111; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 30px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                
                img {{ 
                    max-width: 100%; 
                    height: auto; 
                    display: block;
                    border-radius: 8px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
                    margin-top: 20px; 
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
    </html>
    """

    html_path = os.path.join("outputs", f"{ticker}_report.html")
    pdf_path = os.path.join("outputs", f"{ticker}_Investment_Memo.pdf")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(styled_html)

    try:
        HTML(string=styled_html, base_url=os.path.abspath("outputs")).write_pdf(pdf_path)
        print(f"PDF successfully generated at: {pdf_path}")
    except Exception as e:
        print(f"PDF Generation failed: {e}")

    return {"pdf_path": pdf_path}
