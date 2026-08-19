import os
import json
import requests
import streamlit as st

st.set_page_config(page_title="Agentic Equity Research", page_icon="📈", layout="wide")

st.title("📈 Agentic Equity Research")
st.markdown("Enter a stock ticker and a specific research question to generate an AI-driven investment memo.")

if "research_data" not in st.session_state:
    st.session_state.research_data = None

with st.sidebar:
    st.header("Research Parameters")
    ticker = st.text_input("Stock Ticker", placeholder="e.g., NVDA, TSLA").strip().upper()
    query = st.text_area("Research Question", placeholder="What are the specific risks mentioned in the 10-K?")
    submit_btn = st.button("Generate Report", type="primary")

if submit_btn:
    if not ticker or not query:
        st.warning("Please enter both a ticker and a research question.")
    else:
        with st.status(f"Agents are researching {ticker}...", expanded=True) as status_box:
            try:
                API_URL = os.environ.get("API_URL", "http://backend:8000/research")

                response = requests.post(
                    API_URL,
                    json={"ticker": ticker, "query": query},
                    stream=True,
                    timeout=300
                )

                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8'))

                        if data.get("type") == "update":
                            step_name = data['step'].replace('_', ' ').title()
                            st.write(f"🔄 Agent finished task: **{step_name}**")
                        elif data.get("type") == "final":
                            st.session_state.research_data = data
                            status_box.update(label="Research Complete!", state="complete", expanded=False)
                        elif data.get("type") == "error":
                            st.error(f"API Error: {data.get('detail')}")
                            st.session_state.research_data = None
                            status_box.update(label="Error occurred", state="error")

            except requests.exceptions.ConnectionError:
                st.error("Failed to connect! Make sure your FastAPI server is running.")
                status_box.update(label="Connection Error", state="error")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                status_box.update(label="Error", state="error")

if st.session_state.research_data:
    data = st.session_state.research_data
    current_ticker = data.get("ticker", "Report")

    chart_path = data.get("chart_path")
    if chart_path and os.path.exists(chart_path):
        st.subheader(f"📊 {current_ticker} Historical Performance")
        st.image(chart_path, caption=f"{current_ticker} 6-Month Price & Trend Chart", width="stretch")
    
    st.markdown("---")
    st.markdown(data.get("memo"))
    
    pdf_path = data.get("pdf_path")
    if pdf_path and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Full PDF Report",
                data=pdf_file,
                file_name=f"{current_ticker}_Investment_Memo.pdf",
                mime="application/pdf"
            )