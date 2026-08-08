import os
import getpass
import yfinance as yf
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

@tool
def get_stock_info(ticker: str) -> dict:
    """
    Fetches the current stock price, market cap and key financial ratios for a given ticker.
    Use this tool FIRST whenever you are asked to analyze a company or need current market numbers.
    Input should be a standard stock ticker symbol (e.g, 'AAPL', 'MSFT', 'NVDA'). Do not pass company names.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        essential_data = {
            "current_price": info.get("currentPrice", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "business_summary": info.get("longBusinessSummary", "N/A")[:500] + "..."
        }
        return essential_data

    except Exception as e:
        return {"error": f"Failed to fetch data for {ticker}. Error: {str(e)}"}


@tool
def search_news(ticker: str) -> str:
    """
    Searches the web for the most recent news, articles and sentiment regarding a specific stock ticker.
    Use this tool to find qualitative data, recent product launches or leadership changes.
    Input MUST be a standard stock ticker symbol (e.g, 'AAPL', 'MSFT', 'NVDA').
    """
    try:
        search = TavilySearch(max_results=5, topic="general")
        query = f"Latest financial and business news for {ticker} stock"
        results = search.invoke({"query": query})
        return str(results)

    except Exception as e:
        return f"Failed to fetch news for {ticker}. Error: {str(e)}"