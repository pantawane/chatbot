
# backend.py

import os
import sqlite3
import requests

from dotenv import load_dotenv
from ddgs import DDGS

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


# =========================================
# 1. ENVIRONMENT VARIABLES
# =========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file")


# =========================================
# 2. LLM
# =========================================

model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=GROQ_API_KEY,
)


# =========================================
# 3. TOOLS
# =========================================

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information, news, prices,
    facts, and information that may have changed recently.
    Use this tool when the answer requires current information.
    """
    try:
        with DDGS(timeout=10) as ddgs:
            results = list(
                ddgs.text(
                    query,
                    max_results=5,
                    region="us-en",
                )
            )

        if not results:
            return (
                "No search results were returned. "
                "Try a different query or another search provider."
            )

        formatted_results = []

        for index, result in enumerate(results, start=1):
            formatted_results.append(
                f"Result {index}:\n"
                f"Title: {result.get('title', 'N/A')}\n"
                f"URL: {result.get('href', 'N/A')}\n"
                f"Snippet: {result.get('body', 'N/A')}"
            )

        return "\n\n".join(formatted_results)

    except Exception as exc:
        # A failed search should not crash the entire application.
        return (
            "Web search is temporarily unavailable. "
            f"Error: {type(exc).__name__}: {exc}. "
            "Do not invent current information; tell the user "
            "that live search could not be completed."
        )


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Perform arithmetic on two numbers.
    Supported operations: add, sub, mul, div.
    """
    operation = operation.lower().strip()

    if operation == "add":
        result = first_num + second_num

    elif operation == "sub":
        result = first_num - second_num

    elif operation == "mul":
        result = first_num * second_num

    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero is not allowed"}

        result = first_num / second_num

    else:
        return {
            "error": (
                f"Unsupported operation: {operation}. "
                "Use add, sub, mul, or div."
            )
        }

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest available stock quote using Alpha Vantage.
    Examples: IBM, AAPL, TSLA.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return {
            "error": "ALPHA_VANTAGE_API_KEY is missing from .env"
        }

    symbol = symbol.strip().upper()

    if not symbol:
        return {"error": "Please provide a valid stock symbol"}

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_API_KEY,
            },
            timeout=10,
        )

        response.raise_for_status()
        data = response.json()

        # Alpha Vantage may return API limits as JSON,
        # rather than an HTTP error.
        if "Error Message" in data:
            return {"error": data["Error Message"]}

        if "Note" in data or "Information" in data:
            return {
                "error": "API limit or access restriction",
                "details": data.get("Note")
                or data.get("Information"),
            }

        quote = data.get("Global Quote", {})

        if not quote:
            return {
                "error": (
                    f"No quote returned for {symbol}. "
                    "Check the ticker and API access."
                )
            }

        return {
            "symbol": quote.get("01. symbol", symbol),
            "price": quote.get("05. price"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
            "latest_trading_day": quote.get("07. latest trading day"),
            "source": "Alpha Vantage",
        }

    except requests.RequestException as exc:
        return {"error": f"Stock API request failed: {exc}"}

    except ValueError:
        return {"error": "Stock API returned invalid JSON"}


# =========================================
# 4. BIND TOOLS TO MODEL
# =========================================

tools = [
    web_search,
    get_stock_price,
    calculator,
]

llm_with_tools = model.bind_tools(tools)


# =========================================
# 5. GRAPH STATE
# =========================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =========================================
# 6. GRAPH NODES
# =========================================

def chat_node(state: ChatState):
    """Generate a response or request a tool call."""

    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


# ToolNode executes the requested tools.
tool_node = ToolNode(tools)


# =========================================
# 7. SQLITE CHECKPOINTER
# =========================================

conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn=conn)

# Initialize SQLite checkpoint tables if needed.
checkpointer.setup()


# =========================================
# 8. BUILD LANGGRAPH
# =========================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(
    checkpointer=checkpointer,
)


# =========================================
# 9. THREAD HELPERS
# =========================================

def retrieve_all_threads() -> list[str]:
    """Return the unique conversation thread IDs."""

    all_threads = set()

    for checkpoint in checkpointer.list(None):
        thread_id = (
            checkpoint.config
            .get("configurable", {})
            .get("thread_id")
        )

        if thread_id is not None:
            all_threads.add(thread_id)

    return list(all_threads)


def chat(message: str, thread_id: str = "default") -> str:
    """
    Send a message and return the assistant's final response.
    Reuse thread_id to continue an existing conversation.
    """

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    result = chatbot.invoke(
        {
            "messages": [
                {"role": "user", "content": message}
            ]
        },
        config=config,
    )

    return result["messages"][-1].content
