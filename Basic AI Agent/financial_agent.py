import os
import time
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo


dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("❌ No Groq API key found. Please check your .env file.")
print(f"✅ GROQ_API_KEY loaded: {GROQ_API_KEY[:5]}...") 


groq_model = Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)


def create_agents():
    """Reinitialize agents before each run to avoid cached state issues."""
    web_search_agent = Agent(
        name="Web Search Agent",
        role="Fetch recent news and updates about companies",
        model=groq_model,
        tools=[DuckDuckGo()],
        instructions=[
            "Find recent news and company background information.",
            "Focus on recent developments and include credible sources.",
            "Provide clear and structured results.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

    finance_agent = Agent(
        name="Finance AI Agent",
        model=groq_model,
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_news=True,
            ),
        ],
        instructions=[
            "Provide financial analysis using structured data.",
            "Include key stock market metrics and analyst recommendations.",
            "Present data in a well-formatted table.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

    multi_ai_agent = Agent(
        team=[finance_agent, web_search_agent],
        model=groq_model,
        instructions=[
            "Merge financial data with recent company news.",
            "Ensure information is well-structured, clear, and up to date.",
            "Use markdown for better readability.",
        ],
        show_tool_calls=True,
        markdown=True,
    )

    return finance_agent, web_search_agent, multi_ai_agent


finance_agent, web_search_agent, multi_ai_agent = create_agents()

print("\n🔍 Finance AI Agent:")
finance_agent.print_response("Get the current stock price and key financials for NVIDIA (NVDA).")

print("\n🌐 Web Search Agent:")
web_search_agent.print_response("Find recent news about NVIDIA (NVDA).")

# Reinitialize before multi-agent query
finance_agent, web_search_agent, multi_ai_agent = create_agents()

print("\n🚀 Multi-Agent Analysis for NVIDIA (NVDA):")
multi_ai_agent.print_response(
    "Get NVIDIA (NVDA) stock analysis:\n"
    "- Current stock price and recent performance\n"
    "- Latest analyst recommendations\n"
    "- Recent company news and updates\n"
    "- Key financial metrics (Revenue, EPS, Market Cap)"
)

print("✅ Multi-Agent Response Successfully Generated.")
