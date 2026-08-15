import os
from langgraph.graph import StateGraph, START, END
from agent.utils.state import AgentState
from agent.utils.nodes import planner_node, fetch_stock_node, fetch_news_node, fetch_sec_node, analysis_node, draft_report_node, critic_node, export_node
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt


def route_fetching(state: AgentState) -> list:
    """
    Routes the graph to execute multiple fetch nodes simultaneously.
    """
    plan = state.get("plan", {}).get("required_data", [])
    destinations = []

    if "stock_info" in plan:
        destinations.append("fetch_stock_node")
    if "recent_news" in plan:
        destinations.append("fetch_news_node")
    if "search_sec_filings" in plan:
        destinations.append("fetch_sec_node")

    if not destinations:
        return ["fetch_stock_node", "fetch_news_node", "fetch_sec_node"]

    return destinations


def should_continue(state: AgentState) -> str:
    """
    Determines whether the workflow should continue based on the review status in the state.
    """
    review_status = state.get("review_status", "APPROVED")
    revision_count = state.get("revision_count", 0)

    MAX_REVISIONS = 2

    if review_status == "NEEDS_REVISION" and revision_count < MAX_REVISIONS:
        print(f"\n Critic node requested revision (Attempt {revision_count}/{MAX_REVISIONS}). Re-drafting...")
        return "draft_report_node"

    print("\nCritic approved the memo or max revisions reached! Generating Financial Charts & PDF...")
    return "export_node"

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner_node", planner_node)
    workflow.add_node("fetch_stock_node", fetch_stock_node)
    workflow.add_node("fetch_news_node", fetch_news_node)
    workflow.add_node("fetch_sec_node", fetch_sec_node)
    workflow.add_node("analysis_node", analysis_node)
    workflow.add_node("draft_report_node", draft_report_node)
    workflow.add_node("critic_node", critic_node)
    workflow.add_node("export_node", export_node)

    workflow.add_edge(START, "planner_node")

    workflow.add_conditional_edges(
        "planner_node",
        route_fetching,
        {
            "fetch_stock_node": "fetch_stock_node",
            "fetch_news_node": "fetch_news_node",
            "fetch_sec_node": "fetch_sec_node"
        }
    )

    workflow.add_edge("fetch_stock_node", "analysis_node")
    workflow.add_edge("fetch_news_node", "analysis_node")
    workflow.add_edge("fetch_sec_node", "analysis_node")

    workflow.add_edge("analysis_node", "draft_report_node")
    workflow.add_edge("draft_report_node", "critic_node")

    workflow.add_conditional_edges(
        "critic_node",
        should_continue,
        {
            "draft_report_node": "draft_report_node",
            "export_node": "export_node"
        }
    )

    workflow.add_edge("export_node", END)

    return workflow.compile()

# For testing
if __name__ == "__main__":
    console = Console()
    console.print("\n[bold green]Starting Agentic Equity Research...[/bold green]\n")

    app = build_graph()

    target_ticker = Prompt.ask("[bold cyan]Enter the stock ticker (e.g., AAPL, TSLA)[/bold cyan]").upper().strip()
    custom_query = Prompt.ask(f"[bold cyan]What specific question do you have about {target_ticker}?[/bold cyan]").strip()

    initial_state = {
        "ticker": target_ticker,
        "user_query": custom_query,
        "revision_count": 0
    }

    result = app.invoke(initial_state)

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

    os.makedirs("outputs", exist_ok=True)

    output_filename = f"outputs/{initial_state['ticker']}_Investment_Memo.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(memo_text)

    console.print(Panel(f"[bold cyan]Report Saved to:[/bold cyan] {output_filename}", title="Success"))
    console.print(Markdown(memo_text))