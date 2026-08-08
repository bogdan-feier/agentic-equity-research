import os
from langgraph.graph import StateGraph, START, END
from agent.utils.state import AgentState
from agent.utils.nodes import planner_node, data_fetcher_node, analysis_node, draft_report_node
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

workflow = StateGraph(AgentState)

workflow.add_node("planner_node", planner_node)
workflow.add_node("data_fetcher_node", data_fetcher_node)
workflow.add_node("analysis_node", analysis_node)
workflow.add_node("draft_report_node", draft_report_node)

workflow.add_edge(START, "planner_node")
workflow.add_edge("planner_node", "data_fetcher_node")
workflow.add_edge("data_fetcher_node", "analysis_node")
workflow.add_edge("analysis_node", "draft_report_node")
workflow.add_edge("draft_report_node", END)

app = workflow.compile()

# For testing
if __name__ == "__main__":
    console = Console()
    console.print("\n[bold green] Starting Agentic Equity Research...[/bold green]\n")

    initial_state = {
        "ticker": "AAPL",
        "user_query": "Is Apple a good stock to buy right now given their recent news and current P/E ratio?"
    }

    result = app.invoke(initial_state)

    raw_memo = result["memo"]
    if isinstance(raw_memo, list) and len(raw_memo) > 0:
        memo_text = raw_memo[0].get("text", str(raw_memo))
    elif isinstance(raw_memo, dict):
        memo_text = raw_memo.get("text", str(raw_memo))
    else:
        memo_text = str(raw_memo)

    output_filename = f"outputs/{initial_state['ticker']}_Investment_Memo.md"

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(memo_text)

    console.print(Panel(f"[bold cyan]Report Saved to:[/bold cyan] {output_filename}", title="Success"))
    console.print(Markdown(memo_text))