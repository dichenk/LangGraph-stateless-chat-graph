import os
import operator
import logging
from datetime import datetime, timezone
from typing import Annotated, List, TypedDict, Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Логгирование
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# === State Schema ===
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# === Tool ===
@tool
def get_current_time() -> Dict[str, str]:
    """Return the current UTC time in ISO‑8601 format."""
    utc_time = datetime.now(timezone.utc).isoformat()
    logger.info(f"[Tool] get_current_time called → {utc_time}")
    return {"utc": utc_time}

tools = [get_current_time]

# === Model Setup ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not found in .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-latest",
    google_api_key=GOOGLE_API_KEY,
).bind_tools(tools)

# === Graph Nodes ===
def call_model(state: AgentState) -> Dict[str, Any]:
    """Invoke the model with current messages."""
    messages = state["messages"]
    logger.info(f"[Model] Invoking with messages: {[m.pretty_repr() for m in messages]}")
    response = llm.invoke(messages)
    logger.info(f"[Model] Response: {response.pretty_repr()}")
    return {"messages": [response]}

def call_tools(state: AgentState) -> Dict[str, Any]:
    """Invoke tools requested by the model."""
    last_msg = state["messages"][-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {"messages": []}

    tool_msgs = []
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call["name"]
        logger.info(f"[Tool] Executing: {tool_name}")
        tool_obj = next((t for t in tools if t.name == tool_name), None)
        if not tool_obj:
            content = f"Tool '{tool_name}' not found."
            tool_msgs.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
            continue
        try:
            result = tool_obj.invoke(tool_call["args"])
            tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        except Exception as e:
            content = f"Error in tool '{tool_name}': {e}"
            tool_msgs.append(ToolMessage(content=content, tool_call_id=tool_call["id"]))
    return {"messages": tool_msgs}

def should_continue(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "call_tools"
    return END

# === Graph Assembly ===
builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("call_tools", call_tools)
builder.set_entry_point("call_model")
builder.add_conditional_edges(
    "call_model",
    should_continue,
    {"call_tools": "call_tools", END: END},
)
builder.add_edge("call_tools", "call_model")
graph = builder.compile()

# === CLI ===
if __name__ == "__main__":
    print("🧠 LangGraph Agent (Gemini-1.5-Flash) started. Type message or 'exit'.")
    messages = [
        SystemMessage(
            content="You are a helpful assistant. Use `get_current_time` tool when asked about the time."
        )
    ]
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        messages.append(HumanMessage(content=user_input))
        try:
            state = graph.invoke({"messages": messages})
        except Exception as exc:
            logger.error(f"Error during agent execution: {exc}")
            print("Sorry, an error occurred.")
            continue
        bot_message = state["messages"][-1]
        messages = state["messages"]
        print(f"Bot: {bot_message.content}")