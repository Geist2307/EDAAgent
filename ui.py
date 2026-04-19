"""
EDA Agent — Streamlit UI
Run with: streamlit run ui.py
"""

import base64
import json
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_BASE = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EDA Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    code, pre, .stCode {
        font-family: 'DM Mono', monospace !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0f1117;
        border-right: 1px solid #1e2130;
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Main background */
    .stApp {
        background-color: #f7f8fc;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e8eaf0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Input box */
    [data-testid="stChatInput"] {
        border-radius: 12px;
        border: 1.5px solid #d0d5e8;
        background: #fff;
    }

    /* Header strip */
    .header-strip {
        background: #0f1117;
        color: white;
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .header-strip h1 {
        margin: 0;
        font-size: 1.4rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: white !important;
    }
    .header-strip p {
        margin: 0;
        font-size: 0.85rem;
        color: #8892a4 !important;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    .badge-green  { background: #e6f4ea; color: #1e7e34; }
    .badge-yellow { background: #fff8e1; color: #856404; }

    /* Section label */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8892a4;
        margin: 1.2rem 0 0.4rem 0;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        flex: 1;
        background: #fff;
        border: 1px solid #e8eaf0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-card .val {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f1117;
        font-family: 'DM Mono', monospace;
    }
    .metric-card .lbl {
        font-size: 0.7rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #e8eaf0 !important;
        border-radius: 10px !important;
        background: #fff !important;
    }
            
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {
    color: #0f1117 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────

for key, default in {
    "session_id": None,
    "messages": [],
    "columns": [],
    "shape": [],
    "visuals": {},
    "filename": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────

def b64_to_image(b64_str: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64_str)))


def api_create_session(file_bytes, filename, data_dict: dict) -> dict:
    resp = requests.post(
        f"{API_BASE}/session",
        files={"file": (filename, file_bytes, "text/csv")},
        data={"data_dictionary": json.dumps(data_dict) if data_dict else "{}"},
    )
    resp.raise_for_status()
    return resp.json()


def api_get_visuals(session_id: str) -> dict:
    resp = requests.get(f"{API_BASE}/session/{session_id}/visuals")
    resp.raise_for_status()
    return resp.json()

def api_send_message(session_id: str, message: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/session/{session_id}/message",
        json={"message": message},
    )
    resp.raise_for_status()
    return resp.json()


def reset_session():
    for key in ["session_id", "messages", "columns", "shape", "visuals", "filename"]:
        st.session_state[key] = [] if key in ["messages", "columns", "shape"] else (
            {} if key == "visuals" else None
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        label_visibility="collapsed",
    )

    st.markdown('<p class="section-label">Data Dictionary (optional)</p>', unsafe_allow_html=True)
    dict_input = st.text_area(
        "data_dict",
        placeholder='{"column_name": "what it means", ...}',
        height=130,
        label_visibility="collapsed",
    )

    launch = st.button("🚀  Start Session", use_container_width=True, type="primary")

    if launch:
        if not uploaded_file:
            st.error("Please upload a CSV file first.")
        else:
            data_dict = {}
            if dict_input.strip():
                try:
                    data_dict = json.loads(dict_input)
                except json.JSONDecodeError:
                    st.error("Data dictionary must be valid JSON.")
                    st.stop()

            with st.spinner("Uploading dataset…"):
                try:
                    result = api_create_session(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        data_dict,
                    )
                    st.session_state.session_id = result["session_id"]
                    st.session_state.columns = result["columns"]
                    st.session_state.shape = result["shape"]
                    st.session_state.filename = uploaded_file.name
                    st.session_state.messages = []
                    st.session_state.visuals = {}
                except Exception as e:
                    st.error(f"Failed to create session: {e}")
                    st.stop()

            with st.spinner("Generating visuals…"):
                try:
                    st.session_state.visuals = api_get_visuals(st.session_state.session_id)
                except Exception as e:
                    st.warning(f"Could not load visuals: {e}")

            st.rerun()

    # ── Session info ──
    if st.session_state.session_id:
        st.divider()
        st.markdown('<p class="section-label">Active Session</p>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge badge-green">● Connected</span>', unsafe_allow_html=True)
        st.caption(f"**File:** {st.session_state.filename}")
        st.caption(f"**Rows:** {st.session_state.shape[0]:,}  |  **Cols:** {st.session_state.shape[1]}")

        with st.expander("Columns"):
            for col in st.session_state.columns:
                st.markdown(f"• `{col}`")

        if st.button("🗑  End Session", use_container_width=True):
            try:
                requests.delete(f"{API_BASE}/session/{st.session_state.session_id}")
            except Exception:
                pass
            reset_session()
            st.rerun()
    else:
        st.divider()
        st.markdown('<span class="badge badge-yellow">○ No active session</span>', unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="header-strip">
    <div>
        <h1>📊 EDA Agent</h1>
        <p>Upload a dataset, explore it visually, then ask questions in plain English.</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.session_id:
    st.info("👈  Upload a CSV file in the sidebar to get started.", icon="📂")
    st.stop()

# ── Visuals tabs ──────────────────────────────────────────────────────────────

visuals = st.session_state.visuals
has_visuals = any(visuals.get(k) for k in ["numerical", "categorical", "correlation"])

if has_visuals:
    tab_labels = []
    if visuals.get("numerical"):
        tab_labels.append("📈 Distributions — Numerical")
    if visuals.get("categorical"):
        tab_labels.append("📊 Distributions — Categorical")
    if visuals.get("correlation"):
        tab_labels.append("🔗 Correlation Matrix")

    tabs = st.tabs(tab_labels)
    tab_idx = 0

    if visuals.get("numerical"):
        with tabs[tab_idx]:
            st.image(b64_to_image(visuals["numerical"]), use_container_width=True)
        tab_idx += 1

    if visuals.get("categorical"):
        with tabs[tab_idx]:
            st.image(b64_to_image(visuals["categorical"]), use_container_width=True)
        tab_idx += 1

    if visuals.get("correlation"):
        with tabs[tab_idx]:
            st.image(b64_to_image(visuals["correlation"]), use_container_width=True)

    st.divider()

# ── Chat ──────────────────────────────────────────────────────────────────────

# ── Chat ──────────────────────────────────────────────────────────────────────

st.markdown("#### 💬 Ask the Agent")

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "📊"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("tokens_used"):
            st.caption(
                f"🔢 **{msg['tokens_used']:,} tokens** "
                f"({msg['prompt_tokens']:,} prompt · {msg['completion_tokens']:,} completion) "
                f"· 💰 **${msg['cost_usd']:.5f}**"
            )

# Suggested prompts — shown only before first message
if not st.session_state.messages:
    st.markdown('<p class="section-label">Suggested prompts</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    suggestions = [
        "Produce a full EDA report for this dataset.",
        "What ML problems could I solve with this data?",
        "Which features have the most missing values?",
    ]
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, use_container_width=True):
            st.session_state._quick_prompt = suggestion
            st.rerun()

# Resolve prompt — either from suggestion buttons or chat input
prompt = st.session_state.pop("_quick_prompt", None)
if not prompt:
    prompt = st.chat_input("Ask anything about your dataset…")

# Prevent re-processing the same prompt on rerun
if prompt and prompt == st.session_state.get("_last_prompt"):
    prompt = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📊"):
        with st.spinner("Thinking…"):
            try:
                result = api_send_message(st.session_state.session_id, prompt)
                reply = result["reply"]
            except Exception as e:
                reply = f"⚠️ Error contacting the agent: {e}"
                result = {}
        st.markdown(reply)

        if result.get("tokens_used"):
            st.caption(
                f"🔢 **{result['tokens_used']:,} tokens** "
                f"({result['prompt_tokens']:,} prompt · {result['completion_tokens']:,} completion) "
                f"· 💰 **${result['cost_usd']:.5f}**"
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "tokens_used": result.get("tokens_used"),
        "prompt_tokens": result.get("prompt_tokens"),
        "completion_tokens": result.get("completion_tokens"),
        "cost_usd": result.get("cost_usd"),
    })
    st.session_state._last_prompt = prompt
    st.rerun()