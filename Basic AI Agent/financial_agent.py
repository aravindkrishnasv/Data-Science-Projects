import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

# Load environment variables
load_dotenv()

# Get Groq API Key
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("No Groq API key found. Please set GROQ_API_KEY in your .env file.")

# Configure Groq model with API key
groq_model = Groq(
    id="llama-3.3-70b-versatile", 
    api_key=GROQ_API_KEY
)

# Web Search Agent
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search for additional context about the company",
    model=groq_model, 
    tools=[DuckDuckGo()],
    instructions=[
        "Provide recent news and background information",
        "Focus on recent developments",
        "Include credible sources"
    ],
    show_tool_calls=True,
    markdown=True,   
)

# Financial Agent
finance_agent = Agent(
    name="Finance AI Agent",
    model=groq_model,
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True
        ),
    ],
    instructions=[
        "Provide comprehensive financial analysis",
        "Use clear and structured reporting",
        "Focus on key financial metrics",
        "Present data in a tabular format"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Multi-Agent System (Combining Web Search & Finance)
multi_ai_agent = Agent(
    team=[finance_agent, web_search_agent],
    model=groq_model,
    instructions=[
        "Combine financial data with recent news",
        "Provide a comprehensive and nuanced analysis",
        "Ensure information is current and relevant",
        "Use markdown for clear formatting"
    ],
    show_tool_calls=True,
    markdown=True,
)

# Detailed query for NVDA
try:
    multi_ai_agent.print_response(
        "Provide a comprehensive analysis of NVIDIA (NVDA), including:"
        " 1) Current stock price and recent performance, "
        " 2) Latest analyst recommendations, "
        " 3) Recent company news and developments, "
        " 4) Key financial fundamentals",
        stream=True
    )
except Exception as e:
    print(f"An error occurred: {e}")