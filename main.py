"""
EDA Agent — FastAPI service

Endpoints
---------
POST   /session                   Upload CSV + optional data dictionary → session_id
GET    /session/{id}/visuals      Returns base64 distribution + correlation images
POST   /session/{id}/message      Send a message, get agent reply
DELETE /session/{id}              Clean up session
"""

import io
import json
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent import build_agent, run_turn
from session_store import create_session, delete_session, get_session
from viz import get_correlation_matrix_image, get_distribution_images

app = FastAPI(
    title="EDA Agent",
    description="Upload a CSV, ask questions, get a structured EDA report.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache compiled agents per session so we don't rebuild on every turn
_agent_cache: dict = {}


# ── Models ────────────────────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    message: str


class SessionResponse(BaseModel):
    session_id: str
    columns: list[str]
    shape: list[int]


class MessageResponse(BaseModel):
    reply: str
    tokens_used: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class VisualsResponse(BaseModel):
    numerical: Optional[str] = None    # base64 PNG
    categorical: Optional[str] = None  # base64 PNG
    correlation: Optional[str] = None  # base64 PNG


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_agent(session_id: str):
    if session_id not in _agent_cache:
        session = get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found.")
        _agent_cache[session_id] = build_agent(
            df=session.df,
            data_dictionary=session.data_dictionary,
            checkpointer=session.checkpointer,
        )
    return _agent_cache[session_id]


@app.post("/session", response_model=SessionResponse)
async def create_new_session(
    file: UploadFile = File(..., description="CSV file to analyse"),
    data_dictionary: Optional[str] = Form(
        None,
        description='Optional JSON string mapping column names to descriptions. E.g. {"col_a": "description"}'
    ),
):
    """Upload a CSV and start a new EDA session."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    parsed_dict = {}
    if data_dictionary:
        try:
            parsed_dict = json.loads(data_dictionary)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="data_dictionary must be valid JSON.")

    session_id = create_session(df, parsed_dict)

    return SessionResponse(
        session_id=session_id,
        columns=list(df.columns),
        shape=list(df.shape),
    )
@app.post("/session/{session_id}/message", response_model=MessageResponse)
def send_message(session_id: str, body: MessageRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    agent = _get_agent(session_id)
    result = run_turn(agent, session_id, body.message)
    return MessageResponse(**result)

@app.get("/session/{session_id}/visuals", response_model=VisualsResponse)
def get_visuals(session_id: str):
    """
    Returns base64-encoded PNG images for distribution plots and correlation matrix.
    Call this once after creating a session to render visuals on the client side.
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    dist_images = get_distribution_images(session.df)
    corr_image = get_correlation_matrix_image(session.df)

    return VisualsResponse(
        numerical=dist_images.get("numerical"),
        categorical=dist_images.get("categorical"),
        correlation=corr_image,
    )



@app.delete("/session/{session_id}")
def end_session(session_id: str):
    """Remove a session and free its memory."""
    _agent_cache.pop(session_id, None)
    removed = delete_session(session_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"detail": "Session deleted."}


@app.get("/health")
def health():
    return {"status": "ok"}