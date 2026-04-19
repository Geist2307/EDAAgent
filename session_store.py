"""
In-memory session store.
Holds the dataframe and LangGraph checkpointer per session.
For production: replace with Redis + Postgres checkpointer.
"""

import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver


@dataclass
class Session:
    session_id: str
    df: pd.DataFrame
    data_dictionary: Dict[str, str]
    checkpointer: MemorySaver = field(default_factory=MemorySaver)


# Global in-memory store  { session_id -> Session }
_store: Dict[str, Session] = {}


def create_session(df: pd.DataFrame, data_dictionary: Dict[str, str] | None = None) -> str:
    session_id = str(uuid.uuid4())
    _store[session_id] = Session(
        session_id=session_id,
        df=df,
        data_dictionary=data_dictionary or {},
    )
    return session_id


def get_session(session_id: str) -> Optional[Session]:
    return _store.get(session_id)


def delete_session(session_id: str) -> bool:
    if session_id in _store:
        del _store[session_id]
        return True
    return False