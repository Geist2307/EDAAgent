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
from langchain_community.callbacks import get_openai_callback


from prompts import build_system_prompt
from tools import make_tools


def build_agent(df: pd.DataFrame, data_dictionary: dict, checkpointer: MemorySaver):
    llm = ChatOpenAI(
        model="gpt-5.2",
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=0, # this is for deterministic output
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



def run_turn(agent, session_id: str, user_message: str) -> dict:
    """Send one message and return the assistant's reply plus token/cost metadata."""
    from langchain_core.messages import HumanMessage

    config = {"configurable": {"thread_id": session_id}}

    with get_openai_callback() as cb:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )

    return {
        "reply": result["messages"][-1].content,
        "tokens_used": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "cost_usd": round(cb.total_cost, 5),
    }