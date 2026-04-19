"""
LangGraph agent. Built fresh per session (tools are df-bound),
but state is persisted across turns via the session checkpointer.
"""

import os

import pandas as pd
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from prompts import build_system_prompt
from tools import make_tools


def build_agent(df: pd.DataFrame, data_dictionary: dict, checkpointer: MemorySaver):
    llm = ChatOpenAI(
        model="gpt-4o",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0,
    )

    tools = make_tools(df)
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = build_system_prompt(data_dictionary)

    def assistant(state: MessagesState):
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=checkpointer)


def run_turn(agent, session_id: str, user_message: str) -> str:
    """Send one message and return the assistant's reply."""
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config=config,
    )
    return result["messages"][-1].content