import streamlit as st
import os
import glob
import io
import sqlite3
import hashlib
import base64
from datetime import date, datetime
from PIL import Image
import fitz  # PyMuPDF
import pandas as pd
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit.components.v1 as components

# ==========================================================
# 1. PAGE CONFIGURATION & SETUP
# ==========================================================
st.set_page_config(
    page_title="EduHub - Academic AI Workspace",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Favicon & Head Metas
st.markdown(
    """
    <head>
        <link rel="apple-touch-icon" href="https://i.ibb.co.com/8DstCsX1/attachment-158389628.png">
        <link rel="icon" href="https://i.ibb.co.com/8DstCsX1/attachment-158389628.png">
    </head>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# THEME STATE (Light / Dark) — persisted for the session
# ==========================================================
if "ui_theme" not in st.session_state:
    st.session_state["ui_theme"] = "light"

# ==========================================================
# Global Network Connection Status Detector (upgraded)
# ==========================================================
components.html("""
<div id="net-status-banner" style="
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100%;
    background: linear-gradient(135deg, #F97316 0%, #DC2626 100%);
    color: white; text-align: center;
    padding: 12px; font-weight: 700; font-family: 'Plus Jakarta Sans', sans-serif;
    z-index: 999999; box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    font-size: 0.9rem;
    transform: translateY(-100%);
    transition: transform 0.35s cubic-bezier(.4,0,.2,1);
    display: flex; align-items: center; justify-content: center; gap: 8px;
">
    <span style="display:inline-flex;width:8px;height:8px;border-radius:50%;background:#fff;animation:pulseDot 1.2s infinite;"></span>
    অফলাইন মোড: ইন্টারনেট কানেকশন বিচ্ছিন্ন! আপনি সেভ করা অফলাইন PDF পড়তে পারবেন।
</div>
<style>
@keyframes pulseDot { 0%,100%{opacity:1;transform:scale(1);} 50%{opacity:.4;transform:scale(1.4);} }
</style>
<script>
function updateOnlineStatus() {
    var banner = document.getElementById("net-status-banner");
    if (!navigator.onLine) {
        banner.style.display = "flex";
        requestAnimationFrame(()=>{ banner.style.transform = "translateY(0)"; });
    } else {
        banner.style.transform = "translateY(-100%)";
        setTimeout(()=>{ banner.style.display = "none"; }, 350);
    }
}
window.addEventListener('online', updateOnlineStatus);
window.addEventListener('offline', updateOnlineStatus);
updateOnlineStatus();
</script>
""", height=0)

# ==========================================================
# 2. PROGRESS TRACKING (SQLite Database)
# ==========================================================
DB_PATH = os.path.join(BASE_DIR, "eduhub_progress.db")

def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT,
                course_code TEXT NOT NULL,
                action TEXT NOT NULL,
                activity_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        return conn
    except Exception as e:
        st.error(f"Database Error: {e}")
        return None

def log_activity(student_id, student_name, course_code, action):
    if not student_id:
        return
    conn = get_db_connection()
    if conn:
        try:
            conn.execute(
                "INSERT INTO activity_log (student_id, student_name, course_code, action, activity_date, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (student_id, student_name, course_code, action, date.today().isoformat(), datetime.now().isoformat())
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()

def get_study_streak(student_id):
    if not student_id: return 0
    conn = get_db_connection()
    if not conn: return 0
    rows = conn.execute("SELECT DISTINCT activity_date FROM activity_log WHERE student_id = ?", (student_id,)).fetchall()
    conn.close()
    if not rows: return 0
    activity_dates = {date.fromisoformat(r[0]) for r in rows}
    streak, cursor_date = 0, date.today()
    while cursor_date in activity_dates:
        streak += 1
        cursor_date = date.fromordinal(cursor_date.toordinal() - 1)
    return streak

def get_total_activities(student_id):
    if not student_id: return 0
    conn = get_db_connection()
    if not conn: return 0
    row = conn.execute("SELECT COUNT(*) FROM activity_log WHERE student_id = ?", (student_id,)).fetchone()
    conn.close()
    return row[0] if row else 0

def get_course_progress(student_id):
    if not student_id: return {}
    conn = get_db_connection()
    if not conn: return {}
    rows = conn.execute("SELECT course_code, COUNT(*) FROM activity_log WHERE student_id = ? GROUP BY course_code", (student_id,)).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}

def get_leaderboard():
    conn = get_db_connection()
    if not conn: return []
    rows = conn.execute(
        "SELECT student_name, student_id, COUNT(*) as total_act, MAX(activity_date) as last_active "
        "FROM activity_log GROUP BY student_id ORDER BY total_act DESC LIMIT 10"
    ).fetchall()
    conn.close()
    return rows

def track(action, course_code):
    sid = st.session_state.get("student_id")
    sname = st.session_state.get("student_name", "")
    if sid:
        log_activity(sid, sname, course_code, action)

# ==========================================================
# 3. HELPER FUNCTIONS & AI RAG
# ==========================================================
def ask_gemini(llm, docs, question):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"Role: Expert Academic Assistant.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer in Bengali clearly:"
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"

def display_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        st.info(f"📖 **Total Pages:** {len(doc)}")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=130)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))
            st.image(image, caption=f"Page {page_num + 1}", use_container_width=True)
    except Exception as e:
        st.error(f"Error viewing PDF: {e}")

def skeleton_html(label="Processing"):
    return f"""
        <div class="skeleton-wrap">
            <div class="skeleton-badge"><span class="dot"></span>{label}...</div>
            <div class="skeleton-line" style="width:90%"></div>
            <div class="skeleton-line" style="width:75%"></div>
            <div class="skeleton-line" style="width:60%"></div>
        </div>
    """

def circular_progress_svg(value, max_value=30, label="Day Streak", emoji="🔥"):
    """Render an animated circular progress ring as inline SVG."""
    pct = min(value / max_value, 1.0) if max_value else 0
    radius = 54
    circumference = 2 * 3.14159265 * radius
    offset = circumference * (1 - pct)
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;padding:18px;">
        <svg width="140" height="140" viewBox="0 0 140 140" style="transform:rotate(-90deg);">
            <circle cx="70" cy="70" r="{radius}" fill="none" stroke="rgba(99,102,241,0.12)" stroke-width="12"/>
            <circle cx="70" cy="70" r="{radius}" fill="none" stroke="url(#streakGrad)" stroke-width="12"
                stroke-linecap="round" stroke-dasharray="{circumference}" stroke-dashoffset="{offset}"
                style="transition: stroke-dashoffset 1s cubic-bezier(.4,0,.2,1);"/>
            <defs>
                <linearGradient id="streakGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#6366F1"/>
                    <stop offset="100%" stop-color="#EC4899"/>
                </linearGradient>
            </defs>
        </svg>
        <div style="margin-top:-96px;text-align:center;">
            <div style="font-size:1.9rem;">{emoji}</div>
            <div style="font-family:'Outfit',sans-serif;font-size:1.7rem;font-weight:800;color:#4F46E5;">{value}</div>
        </div>
        <div style="margin-top:14px;font-size:0.8rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
    </div>
    """

@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm(_api_key):
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=_api_key, temperature=0.3)

@st.cache_data(show_spinner=False)
def extract_text_from_local_pdfs(pdf_paths, cache_key):
    raw, pages = "", 0
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            pages += len(reader.pages)
            for page in reader.pages:
                text = page.extract_text()
                if text: raw += text + "\n"
        except Exception:
            pass
    return raw, pages

@st.cache_resource(show_spinner=False)
def build_vector_store(course_code, text_hash, raw_text):
    if not raw_text.strip(): return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    embeddings = get_embeddings_model()
    return FAISS.from_texts(chunks, embedding=embeddings)

# ==========================================================
# 4. PREMIUM UI SYSTEM — Light + Dark, Glassmorphism, Motion
# ==========================================================
_dark = st.session_state["ui_theme"] == "dark"

_root_vars_light = """
    --primary: #6366F1;
    --primary-2: #8B5CF6;
    --primary-gradient: linear-gradient(135deg, #6366F1 0%, #8B5CF6 55%, #EC4899 100%);
    --accent-glow: rgba(99, 102, 241, 0.25);
    --bg-app-1: #EEF2FF;
    --bg-app-2: #F8FAFC;
    --bg-app-3: #E0E7FF;
    --surface: rgba(255, 255, 255, 0.78);
    --surface-solid: #FFFFFF;
    --border-color: rgba(226, 232, 240, 0.9);
    --text-dark: #0F172A;
    --text-muted: #64748B;
    --sidebar-bg: rgba(255, 255, 255, 0.5);
    --shadow-soft: 0 10px 25px -5px rgba(30, 41, 59, 0.08), 0 8px 10px -6px rgba(30, 41, 59, 0.03);
"""

_root_vars_dark = """
    --primary: #818CF8;
    --primary-2: #A78BFA;
    --primary-gradient: linear-gradient(135deg, #6366F1 0%, #8B5CF6 55%, #EC4899 100%);
    --accent-glow: rgba(129, 140, 248, 0.28);
    --bg-app-1: #0B1220;
    --bg-app-2: #0F172A;
    --bg-app-3: #131C31;
    --surface: rgba(30, 41, 59, 0.55);
    --surface-solid: #161F32;
    --border-color: rgba(71, 85, 105, 0.5);
    --text-dark: #F1F5F9;
    --text-muted: #94A3B8;
    --sidebar-bg: rgba(15, 23, 42, 0.55);
    --shadow-soft: 0 10px 25px -5px rgba(0, 0, 0, 0.45), 0 8px 10px -6px rgba(0, 0, 0, 0.25);
"""

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

    :root {{
        {_root_vars_dark if _dark else _root_vars_light}
        --radius-lg: 22px;
        --radius-md: 16px;
        --radius-sm: 10px;
    }}

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', sans-serif;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Outfit', sans-serif !important;
        letter-spacing: -0.02em;
    }}
    p, span, label, li, div {{ color: var(--text-dark); }}

    .stApp {{
        background:
            radial-gradient(circle at 15% 10%, var(--accent-glow) 0%, transparent 45%),
            radial-gradient(circle at 85% 90%, rgba(236,72,153,0.15) 0%, transparent 45%),
            linear-gradient(135deg, var(--bg-app-1) 0%, var(--bg-app-2) 50%, var(--bg-app-3) 100%);
        background-attachment: fixed;
    }}
    .block-container {{ padding-top: 1.5rem !important; max-width: 1260px; }}

    #MainMenu, footer, [data-testid="stDeployButton"] {{ visibility: hidden; height: 0; }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}

    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: var(--primary-2); border-radius: 10px; opacity: 0.5; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}

    /* ---------- Animated Header ---------- */
    .header-box {{
        background: linear-gradient(135deg, #020617 0%, #111827 55%, #1E1B4B 100%) !important;
        padding: 40px 24px;
        border-radius: var(--radius-lg);
        text-align: center;
        color: #FFFFFF !important;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 25px 45px -12px rgba(2, 6, 23, 0.55);
    }}
    .header-box::before {{
        content: "";
        position: absolute; inset: -40%;
        background:
            radial-gradient(circle at 20% 30%, rgba(99,102,241,0.35), transparent 55%),
            radial-gradient(circle at 80% 70%, rgba(236,72,153,0.3), transparent 55%),
            radial-gradient(circle at 50% 100%, rgba(139,92,246,0.25), transparent 55%);
        animation: meshDrift 14s ease-in-out infinite alternate;
        pointer-events: none;
    }}
    @keyframes meshDrift {{
        0% {{ transform: translate(0,0) scale(1); }}
        100% {{ transform: translate(3%, -3%) scale(1.08); }}
    }}
    .header-inner {{ position: relative; z-index: 2; }}

    .uni-logo-wrapper {{
        display: inline-block;
        background: #FFFFFF;
        padding: 9px 15px;
        border-radius: 20px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.4);
        margin-bottom: 16px;
    }}
    .uni-logo-corner {{
        height: 66px;
        width: auto;
        display: block;
        margin: 0 auto;
        object-fit: contain;
        border-radius: 4px;
    }}
    .header-box h2 {{
        color: #FFFFFF !important;
        font-size: 1.9rem !important;
        font-weight: 800 !important;
        margin: 0;
        letter-spacing: 0.2px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
    }}
    .header-tagline {{
        margin-top: 8px;
        color: rgba(255,255,255,0.65) !important;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        backdrop-filter: blur(22px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(22px) saturate(180%) !important;
        border-right: 1px solid var(--border-color) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.05) !important;
    }}
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background: var(--surface-solid) !important;
        color: var(--text-dark) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03) !important;
        transition: all 0.25s ease !important;
    }}
    section[data-testid="stSidebar"] .stTextInput input:focus,
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div:focus-within {{
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }}

    .theme-toggle-row {{ display:flex; gap:8px; margin-bottom: 6px; }}

    /* ---------- Course Card Banner ---------- */
    .course-card {{
        background: var(--primary-gradient);
        padding: 22px 30px;
        border-radius: var(--radius-md);
        color: white;
        margin-bottom: 26px;
        box-shadow: 0 15px 30px -8px rgba(99, 102, 241, 0.4);
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: relative;
        overflow: hidden;
    }}
    .course-card::after {{
        content: "";
        position: absolute; top: -50%; right: -10%;
        width: 220px; height: 220px; border-radius: 50%;
        background: rgba(255,255,255,0.12);
        filter: blur(10px);
    }}
    .course-card h1 {{
        color: #FFFFFF !important;
        font-size: 1.5rem;
        margin: 0;
        font-weight: 700;
        position: relative; z-index: 2;
    }}
    .course-card .badge {{
        position: relative; z-index: 2;
        font-size: 0.78rem; text-transform: uppercase; opacity: 0.9;
        font-weight: 800; letter-spacing: 0.08em;
        background: rgba(255,255,255,0.18);
        padding: 4px 12px; border-radius: 20px;
        display: inline-block; margin-bottom: 6px;
    }}

    /* ---------- Metric / Stat Cards ---------- */
    .metric-card {{
        background: var(--surface);
        backdrop-filter: blur(14px);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 22px;
        text-align: center;
        box-shadow: var(--shadow-soft);
        transition: all 0.28s cubic-bezier(.4,0,.2,1);
        position: relative;
    }}
    .metric-card:hover {{
        transform: translateY(-5px);
        border-color: var(--primary-2);
        box-shadow: 0 18px 34px -10px var(--accent-glow);
    }}
    .metric-card-icon {{ font-size: 1.4rem; margin-bottom: 6px; }}
    .metric-card-val {{
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2px;
        line-height: 1;
    }}
    .metric-card-lbl {{
        font-size: 0.78rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    /* ---------- Nav Tabs ---------- */
    div[data-testid="stRadio"] {{
        background: var(--surface);
        backdrop-filter: blur(12px);
        padding: 7px;
        border-radius: 20px;
        border: 1px solid var(--border-color);
        margin-bottom: 22px;
        box-shadow: var(--shadow-soft);
    }}
    div[data-testid="stRadio"] > div {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        gap: 6px !important;
        padding: 2px !important;
        scrollbar-width: none;
    }}
    div[data-testid="stRadio"] > div::-webkit-scrollbar {{ display: none; }}
    div[data-testid="stRadio"] input[type="radio"] {{ display: none !important; }}
    div[data-testid="stRadio"] label {{
        background-color: transparent !important;
        border: none !important;
        border-radius: 13px !important;
        padding: 11px 20px !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        cursor: pointer !important;
        white-space: nowrap !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 0 !important;
    }}
    div[data-testid="stRadio"] label:hover {{
        color: var(--text-dark) !important;
        background: rgba(99,102,241,0.08) !important;
    }}
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) {{
        background: var(--primary-gradient) !important;
        box-shadow: 0 6px 16px var(--accent-glow) !important;
    }}
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) p {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        background: var(--primary-gradient) !important;
        color: white !important;
        border-radius: 13px !important;
        padding: 10px 24px !important;
        font-weight: 700 !important;
        border: none !important;
        box-shadow: 0 6px 16px var(--accent-glow) !important;
        transition: all 0.2s ease !important;
    }}
    .stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 22px var(--accent-glow) !important;
    }}
    .stDownloadButton > button {{
        border-radius: 13px !important;
        font-weight: 700 !important;
    }}

    /* ---------- Chat ---------- */
    [data-testid="stChatMessage"] {{
        background: var(--surface) !important;
        backdrop-filter: blur(10px);
        border: 1px solid var(--border-color) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-soft);
    }}

    /* ---------- Skeleton Loader (shimmer) ---------- */
    .skeleton-wrap {{
        padding: 22px;
        background: var(--surface-solid);
        border-radius: 16px;
        border: 1px solid var(--border-color);
        margin: 12px 0;
        box-shadow: var(--shadow-soft);
    }}
    .skeleton-badge {{
        font-weight: 700; color: var(--primary);
        margin-bottom: 14px; font-size: 0.9rem;
        display: flex; align-items: center; gap: 8px;
    }}
    .skeleton-badge .dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: var(--primary-gradient);
        animation: dotPulse 1s infinite ease-in-out;
    }}
    @keyframes dotPulse {{ 0%,100%{{transform:scale(1);opacity:1;}} 50%{{transform:scale(1.5);opacity:.5;}} }}
    .skeleton-line {{
        height: 13px; border-radius: 7px; margin-bottom: 10px;
        background: linear-gradient(90deg, rgba(148,163,184,0.15) 25%, rgba(148,163,184,0.35) 37%, rgba(148,163,184,0.15) 63%);
        background-size: 400% 100%;
        animation: shimmer 1.4s ease-in-out infinite;
    }}
    @keyframes shimmer {{ 0%{{background-position:100% 50%;}} 100%{{background-position:0% 50%;}} }}

    /* ---------- Leaderboard rank pills ---------- */
    .rank-pill {{
        display:inline-flex; align-items:center; justify-content:center;
        width: 30px; height: 30px; border-radius: 50%;
        font-weight: 800; font-size: 0.85rem; color: white;
    }}

    /* ---------- Misc ---------- */
    .section-heading {{
        font-family:'Outfit',sans-serif; font-weight:800; font-size:1.15rem;
        margin: 18px 0 10px 0; color: var(--text-dark);
        display:flex; align-items:center; gap:8px;
    }}
    .empty-state {{
        text-align:center; padding: 40px 20px; color: var(--text-muted);
        background: var(--surface); border: 1px dashed var(--border-color);
        border-radius: var(--radius-md);
    }}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 5. HEADER & NAVIGATION (WITH LOGO LOADER)
# ==========================================================
def _load_logo_b64():
    logo_path = os.path.join(ASSETS_DIR, "university_logo.png")
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return ""
    return ""

_logo_b64 = _load_logo_b64()
if _logo_b64:
    _logo_img = f'<img src="data:image/png;base64,{_logo_b64}" class="uni-logo-corner" alt="Logo">'
else:
    _logo_img = '<img src="https://i.ibb.co.com/8DstCsX1/attachment-158389628.png" class="uni-logo-corner" alt="Logo">'

st.markdown(f"""
    <div class="header-box">
        <div class="header-inner">
            <div class="uni-logo-wrapper">
                {_logo_img}
            </div>
            <h2>Department of Environmental Science and Engineering</h2>
            <div class="header-tagline">🎓 EduHub · AI-Powered Academic Workspace</div>
        </div>
    </div>
""", unsafe_allow_html=True)

COURSES = {
    "SYL": "Syllabus",
    "ROU": "Routine",
    "ESE 2101": "Hydrology and Hydrogeology",
    "ESE 2103": "Oceanography and Limnology",
    "ESE 2105": "Ecology",
    "ESE 2102": "Ecology - Lab",
    "ESE 2107": "Environmental Microbiology",
    "ESE 2104": "Environmental Microbiology - Lab",
    "ESE 2109": "Survey and Settlement",
    "ESE 2106": "Survey and Settlement - Lab",
    "ESE 2111": "Soil Mechanics",
    "ESE 2108": "Engineering Drawing Lab",
    "ESE 2113": "Statistics for Environment",
    "PYQ": "Previous Year Questions",
    "MEQ": "Mid Exam Questions"
}

course_options = [f"{code} - {title}" for code, title in COURSES.items()]

with st.sidebar:
    st.markdown("<h3 style='font-size: 1.15rem; font-weight: 800; margin-bottom: 12px;'>⚙️ Workspace Navigation</h3>", unsafe_allow_html=True)

    theme_label = "🌙 Switch to Dark" if not _dark else "☀️ Switch to Light"
    if st.button(theme_label, use_container_width=True, key="theme_toggle_btn"):
        st.session_state["ui_theme"] = "dark" if not _dark else "light"
        st.rerun()

    st.divider()

    search_query = st.text_input("🔍 Search Courses", placeholder="e.g. hydrology", key="global_search_input")
    if search_query.strip():
        q = search_query.strip().lower()
        matched = [(code, title) for code, title in COURSES.items() if q in code.lower() or q in title.lower()]
        for code, title in matched[:5]:
            if st.button(f"📘 {code} — {title}", key=f"search_{code}", use_container_width=True):
                st.session_state["course_selectbox"] = f"{code} - {title}"
                st.rerun()
        st.divider()

    selected_option = st.selectbox("📌 Select Course Material", course_options, key="course_selectbox")
    selected_code = selected_option.split(" - ")[0]
    selected_title = COURSES[selected_code]

    st.divider()
    st.markdown("<p style='font-size: 0.8rem; font-weight: 800; color: var(--text-muted); letter-spacing: 0.05em;'>👤 YOUR PROFILE</p>", unsafe_allow_html=True)
    student_name_input = st.text_input("Your Name", placeholder="e.g. Amir Hamja Ratul", key="student_name_field")
    student_roll_input = st.text_input("Roll Number", placeholder="e.g. 25103402", key="student_roll_field")

    if student_roll_input.strip():
        st.session_state["student_id"] = student_roll_input.strip()
        st.session_state["student_name"] = student_name_input.strip() or student_roll_input.strip()
        st.caption(f"✅ Active: **{st.session_state['student_name']}**")
    else:
        st.session_state["student_id"] = None

    st.divider()
    query_params = st.query_params
    admin_pass = st.text_input("🔒 Admin Key", type="password") if query_params.get("admin") == "true" else ""

st.markdown(f"""
    <div class="course-card">
        <div>
            <span class="badge">Active Course Material</span>
            <h1>🎓 {selected_code}: {selected_title}</h1>
        </div>
    </div>
""", unsafe_allow_html=True)

folder_code = selected_code.replace(" ", "_")
course_folder = os.path.join(DATA_DIR, folder_code)
os.makedirs(course_folder, exist_ok=True)

if admin_pass == "285277":
    st.success("⚡ Admin Mode Active")
    uploaded_files = st.file_uploader("📥 Upload PDFs", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        for u_file in uploaded_files:
            with open(os.path.join(course_folder, u_file.name), "wb") as f:
                f.write(u_file.getbuffer())
        st.toast("✅ Files saved successfully!", icon="🎉")
        st.rerun()

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", None))
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY পাওয়া যায়নি! Streamlit Secrets বা Environment Variable-এ যুক্ত করুন।")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

local_pdfs = glob.glob(os.path.join(course_folder, "*.pdf"))
raw_text, total_pages = "", 0
files_count = len(local_pdfs)

if local_pdfs:
    mtimes = tuple(os.path.getmtime(p) for p in local_pdfs)
    raw_text, total_pages = extract_text_from_local_pdfs(tuple(local_pdfs), mtimes)

if raw_text.strip():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-card-icon">📁</div><div class="metric-card-val">{files_count}</div><div class="metric-card-lbl">Documents Loaded</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-card-icon">📄</div><div class="metric-card-val">{total_pages}</div><div class="metric-card-lbl">Pages Indexed</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-card-icon">🤖</div><div class="metric-card-val">AI</div><div class="metric-card-lbl">Assistant Ready</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

tab_selection = st.radio(
    "Navigation Tabs",
    ["📖 View & Download", "📲 Offline Saved PDFs", "💬 AI Q&A", "📝 Smart Summary", "🎯 Exam Quiz", "📈 My Progress", "📊 Leaderboard"],
    horizontal=True,
    label_visibility="collapsed"
)

llm = get_llm(api_key)
vector_store = None

if raw_text.strip():
    text_hash = hashlib.md5(raw_text.encode("utf-8")).hexdigest()
    vector_store = build_vector_store(selected_code, text_hash, raw_text)

# ==========================================================
# TAB 1: 📖 VIEW & DOWNLOAD
# ==========================================================
if tab_selection == "📖 View & Download":
    st.markdown(f'<div class="section-heading">📖 View &amp; Download — {selected_code}</div>', unsafe_allow_html=True)
    track("View Document", selected_code)

    if local_pdfs:
        selected_pdf = st.selectbox("📄 Select PDF File", local_pdfs, format_func=os.path.basename)
        pdf_name = os.path.basename(selected_pdf)

        with open(selected_pdf, "rb") as f:
            pdf_bytes = f.read()
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

        col_dl, col_save = st.columns([1, 1])
        with col_dl:
            st.download_button(
                label=f"⬇️ Download {pdf_name}",
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True
            )

        with col_save:
            save_offline_html = f"""
            <button onclick="saveToIndexedDB()" style="
                background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                color: white; padding: 11px 20px; border: none; border-radius: 13px;
                font-weight: 700; font-size: 0.9rem; cursor: pointer; width: 100%;
                box-shadow: 0 6px 16px rgba(16, 185, 129, 0.3); font-family: 'Plus Jakarta Sans', sans-serif;
                transition: transform 0.2s ease;
            " onmouseover="this.style.transform='translateY(-2px)'" onmouseout="this.style.transform='translateY(0)'">
                💾 Save for Offline Reading
            </button>
            <p id="save-status" style="margin-top: 8px; font-weight: 600; color: #10B981; text-align: center; font-size: 0.85rem;"></p>

            <script>
            function saveToIndexedDB() {{
                let request = indexedDB.open("EduHubOfflineDB", 2);
                request.onupgradeneeded = function(e) {{
                    let db = e.target.result;
                    if (!db.objectStoreNames.contains("pdf_store")) {{
                        db.createObjectStore("pdf_store", {{ keyPath: "id" }});
                    }}
                }};
                request.onsuccess = function(e) {{
                    let db = e.target.result;
                    let tx = db.transaction("pdf_store", "readwrite");
                    let store = tx.objectStore("pdf_store");
                    let pdfData = {{
                        id: "{selected_code}_" + "{pdf_name}",
                        course_code: "{selected_code}",
                        course_title: "{selected_title}",
                        file_name: "{pdf_name}",
                        base64: "{base64_pdf}",
                        saved_at: new Date().toLocaleDateString()
                    }};
                    store.put(pdfData);
                    tx.oncomplete = function() {{
                        document.getElementById("save-status").innerText = "✅ Saved to Browser Memory!";
                    }};
                }};
            }}
            </script>
            """
            components.html(save_offline_html, height=80)

        st.divider()
        display_pdf(selected_pdf)
    else:
        st.markdown('<div class="empty-state">⚠️ এই কোর্সের জন্য কোনো স্থানীয় PDF ফাইল খুঁজে পাওয়া যায়নি।</div>', unsafe_allow_html=True)

# ==========================================================
# TAB 2: 📲 OFFLINE SAVED PDFS
# ==========================================================
elif tab_selection == "📲 Offline Saved PDFs":
    st.markdown('<div class="section-heading">📲 Course-Wise Offline PDF Manager</div>', unsafe_allow_html=True)
    st.caption("🌐 নেট কানেকশন না থাকলেও পূর্বে সেভ করা PDF কোর্স অনুযায়ী বেছে পড়তে পারবেন।")

    courses_js_array = str(list(COURSES.keys()))
    _panel_bg = "rgba(22,31,50,0.85)" if _dark else "rgba(255,255,255,0.85)"
    _panel_border = "#334155" if _dark else "#E2E8F0"
    _text_col = "#F1F5F9" if _dark else "#0F172A"
    _muted_col = "#94A3B8" if _dark else "#64748B"

    offline_manager_html = f"""
    <div style="background: {_panel_bg}; backdrop-filter: blur(14px); padding: 24px; border-radius: 18px; border: 1px solid {_panel_border}; box-shadow: 0 8px 20px rgba(0,0,0,0.06);">
        <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 20px; flex-wrap: wrap;">
            <label style="font-weight: 700; color: {_text_col}; font-family: sans-serif;">📂 Select Course:</label>
            <select id="courseFilter" onchange="loadOfflinePDFs()" style="
                padding: 10px 16px; border-radius: 10px; border: 1px solid #6366F1;
                font-weight: 600; background: {_panel_bg}; color:{_text_col}; outline: none; cursor: pointer; font-family: sans-serif;
            ">
                <option value="ALL">-- ALL SAVED COURSES --</option>
            </select>
            <button onclick="loadOfflinePDFs()" style="
                background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
                color: white; padding: 10px 18px; border: none; border-radius: 10px;
                font-weight: 700; cursor: pointer; font-family: sans-serif;
            ">
                🔄 Refresh List
            </button>
            <button onclick="clearAllOfflineData()" style="
                background: #EF4444; color: white; padding: 10px 18px; border: none;
                border-radius: 10px; font-weight: 700; cursor: pointer; margin-left: auto; font-family: sans-serif;
            ">
                🗑️ Clear All Saved PDFs
            </button>
        </div>

        <div id="status-msg" style="font-weight: 600; margin-bottom: 15px; color: #818CF8; font-family: sans-serif;"></div>
        <div id="pdf-display-area"></div>
    </div>

    <script>
    const courseList = {courses_js_array};
    const textCol = "{_text_col}";
    const mutedCol = "{_muted_col}";
    const panelBg = "{_panel_bg}";
    const panelBorder = "{_panel_border}";

    function populateDropdown() {{
        let select = document.getElementById("courseFilter");
        courseList.forEach(code => {{
            let opt = document.createElement("option");
            opt.value = code;
            opt.innerText = code;
            select.appendChild(opt);
        }});
        let currentCode = "{selected_code}";
        if(courseList.includes(currentCode)) {{
            select.value = currentCode;
        }}
    }}

    function loadOfflinePDFs() {{
        let selectedCourse = document.getElementById("courseFilter").value;
        let container = document.getElementById("pdf-display-area");
        let statusDiv = document.getElementById("status-msg");
        container.innerHTML = "";
        statusDiv.innerText = "⏳ Reading offline database...";

        let request = indexedDB.open("EduHubOfflineDB", 2);
        request.onsuccess = function(e) {{
            let db = e.target.result;
            if (!db.objectStoreNames.contains("pdf_store")) {{
                statusDiv.innerHTML = "❌ কোনো সেভ করা PDF পাওয়া যায়নি। 'View & Download' ট্যাব থেকে আগে সেভ করুন।";
                return;
            }}
            let tx = db.transaction("pdf_store", "readonly");
            let store = tx.objectStore("pdf_store");
            let req = store.getAll();

            req.onsuccess = function() {{
                let allFiles = req.result;
                let filtered = (selectedCourse === "ALL")
                    ? allFiles
                    : allFiles.filter(item => item.course_code === selectedCourse);

                if (filtered.length === 0) {{
                    statusDiv.innerHTML = "⚠️ <b>" + selectedCourse + "</b> কোর্সের কোনো সেভ করা অফলাইন ফাইল পাওয়া যায়নি।";
                }} else {{
                    statusDiv.innerHTML = "✅ মোট <b>" + filtered.length + "</b> টি অফলাইন PDF পাওয়া গেছে:";
                    filtered.forEach(item => {{
                        let card = document.createElement("div");
                        card.style.cssText = "background: " + panelBg + "; border: 1px solid " + panelBorder + "; border-radius: 14px; padding: 16px; margin-bottom: 20px;";

                        let header = document.createElement("div");
                        header.style.cssText = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;";
                        header.innerHTML = "<div><h4 style='margin:0; color:" + textCol + "; font-family:sans-serif;'>📄 " + item.file_name + "</h4><small style='color:" + mutedCol + "; font-family:sans-serif;'>Course: " + item.course_code + " | Saved on: " + (item.saved_at || 'N/A') + "</small></div>";

                        let delBtn = document.createElement("button");
                        delBtn.innerText = "🗑️ Delete";
                        delBtn.style.cssText = "background:#EF4444; color:white; border:none; padding:6px 12px; border-radius:8px; font-weight:600; cursor:pointer; font-family:sans-serif;";
                        delBtn.onclick = function() {{ deleteOfflinePDF(item.id); }};

                        header.appendChild(delBtn);
                        card.appendChild(header);

                        let iframe = document.createElement("iframe");
                        iframe.src = "data:application/pdf;base64," + item.base64;
                        iframe.style.cssText = "width: 100%; height: 600px; border: 1px solid " + panelBorder + "; border-radius: 10px;";

                        card.appendChild(iframe);
                        container.appendChild(card);
                    }});
                }}
            }};
        }};
    }}

    function deleteOfflinePDF(id) {{
        let request = indexedDB.open("EduHubOfflineDB", 2);
        request.onsuccess = function(e) {{
            let db = e.target.result;
            let tx = db.transaction("pdf_store", "readwrite");
            let store = tx.objectStore("pdf_store");
            store.delete(id);
            tx.oncomplete = function() {{
                loadOfflinePDFs();
            }};
        }};
    }}

    function clearAllOfflineData() {{
        if(confirm("আপনি কি অফলাইনে সেভ করা সকল PDF মুছে ফেলতে চান?")) {{
            let request = indexedDB.open("EduHubOfflineDB", 2);
            request.onsuccess = function(e) {{
                let db = e.target.result;
                let tx = db.transaction("pdf_store", "readwrite");
                let store = tx.objectStore("pdf_store");
                store.clear();
                tx.oncomplete = function() {{
                    loadOfflinePDFs();
                }};
            }};
        }}
    }}

    populateDropdown();
    setTimeout(loadOfflinePDFs, 300);
    </script>
    """
    components.html(offline_manager_html, height=760, scrolling=True)

# ==========================================================
# TAB 3: 💬 AI Q&A
# ==========================================================
elif tab_selection == "💬 AI Q&A":
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.markdown(f'<div class="section-heading">💬 AI Study Assistant — {selected_code}</div>', unsafe_allow_html=True)
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask any question from course materials..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            if vector_store:
                placeholder = st.empty()
                placeholder.markdown(skeleton_html("Analyzing Documents"), unsafe_allow_html=True)

                docs = vector_store.similarity_search(user_query, k=4)
                answer = ask_gemini(llm, docs, user_query)

                placeholder.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                track("Ask Question", selected_code)
            else:
                st.error("⚠️ পর্যাপ্ত ফাইল টেক্সট নেই। অনুগ্রহ করে ফাইল আপলোড বা সিলেক্ট করুন।")

# ==========================================================
# TAB 4: 📝 SMART SUMMARY
# ==========================================================
elif tab_selection == "📝 Smart Summary":
    st.markdown(f'<div class="section-heading">📝 Auto Notes &amp; Summary Generator — {selected_code}</div>', unsafe_allow_html=True)

    if st.button("✨ Generate Smart Academic Notes", use_container_width=True):
        if raw_text.strip():
            placeholder = st.empty()
            placeholder.markdown(skeleton_html("Summarizing Course Topics"), unsafe_allow_html=True)

            prompt = (
                f"Create concise, well-structured academic study notes from the text below.\n"
                f"Include key definitions, main topics, and bullet points in Bengali:\n\n{raw_text[:12000]}"
            )
            try:
                summary_res = llm.invoke(prompt)
                content = summary_res.content if hasattr(summary_res, 'content') else str(summary_res)
                placeholder.markdown(content)
                track("Generated Summary", selected_code)
            except Exception as e:
                placeholder.error(f"Failed to generate summary: {e}")
        else:
            st.warning("⚠️ নোট তৈরি করতে ফাইল টেক্সট প্রয়োজন।")

# ==========================================================
# TAB 5: 🎯 EXAM QUIZ
# ==========================================================
elif tab_selection == "🎯 Exam Quiz":
    st.markdown(f'<div class="section-heading">🎯 Interactive Exam Quiz — {selected_code}</div>', unsafe_allow_html=True)

    if st.button("🎲 Generate Practice Quiz", use_container_width=True):
        if raw_text.strip():
            placeholder = st.empty()
            placeholder.markdown(skeleton_html("Creating Quiz Questions"), unsafe_allow_html=True)

            prompt = (
                f"Generate 5 Multiple Choice Questions (MCQs) with options and 3 Short Answer Questions "
                f"based on the text below. Language: Bengali.\n\n{raw_text[:10000]}"
            )
            try:
                quiz_res = llm.invoke(prompt)
                content = quiz_res.content if hasattr(quiz_res, 'content') else str(quiz_res)
                placeholder.empty()

                with st.expander("📝 View Practice Questions & Solutions", expanded=True):
                    st.markdown(content)
                track("Generated Quiz", selected_code)
            except Exception as e:
                placeholder.error(f"Quiz generation error: {e}")
        else:
            st.warning("⚠️ কুইজ তৈরি করতে ডকুমেন্ট টেক্সট পাওয়া যায়নি।")

# ==========================================================
# TAB 6: 📈 MY PROGRESS
# ==========================================================
elif tab_selection == "📈 My Progress":
    st.markdown('<div class="section-heading">📈 Personal Activity &amp; Progress Tracker</div>', unsafe_allow_html=True)
    sid = st.session_state.get("student_id")

    if sid:
        streak = get_study_streak(sid)
        total_act = get_total_activities(sid)
        progress = get_course_progress(sid)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown(
                f'<div class="metric-card">{circular_progress_svg(streak, max_value=30, label="Study Streak", emoji="🔥")}</div>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown(
                f'<div class="metric-card">{circular_progress_svg(total_act, max_value=max(total_act,10), label="Total Actions", emoji="⚡")}</div>',
                unsafe_allow_html=True
            )

        st.markdown('<div class="section-heading">📊 Course Activity Distribution</div>', unsafe_allow_html=True)
        if progress:
            df = pd.DataFrame(list(progress.items()), columns=["Course Code", "Total Activities"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="empty-state">এখনো কোন এক্টিভিটি রেকর্ড হয়নি। পড়ালেখা ও চ্যাট শুরু করলে ডাটা আপডেট হবে।</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">👉 আপনার প্রতিদিনের অগ্রগতি সেভ করতে সাইডবারে রোল নম্বর যোগ করুন।</div>', unsafe_allow_html=True)

# ==========================================================
# TAB 7: 📊 LEADERBOARD
# ==========================================================
elif tab_selection == "📊 Leaderboard":
    st.markdown('<div class="section-heading">📊 Top Active Student Leaderboard</div>', unsafe_allow_html=True)
    board_data = get_leaderboard()

    if board_data:
        medal_colors = {0: "#F59E0B", 1: "#94A3B8", 2: "#B45309"}
        medal_emojis = {0: "🥇", 1: "🥈", 2: "🥉"}

        for idx, (name, roll, acts, last_active) in enumerate(board_data):
            rank_display = medal_emojis.get(idx, f"#{idx+1}")
            rank_bg = medal_colors.get(idx, "#6366F1")
            initial = (name or "?")[0].upper()
            st.markdown(f"""
                <div class="metric-card" style="display:flex; align-items:center; justify-content:space-between; text-align:left; padding: 14px 20px; margin-bottom: 10px;">
                    <div style="display:flex; align-items:center; gap:14px;">
                        <div class="rank-pill" style="background:{rank_bg};">{rank_display if idx < 3 else idx+1}</div>
                        <div style="width:40px;height:40px;border-radius:50%;background:var(--primary-gradient);display:flex;align-items:center;justify-content:center;color:white;font-weight:800;">{initial}</div>
                        <div>
                            <div style="font-weight:700; font-size:0.95rem;">{name}</div>
                            <div style="font-size:0.78rem; color:var(--text-muted);">Roll: {roll} · Last active: {last_active}</div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-family:'Outfit',sans-serif; font-weight:800; font-size:1.3rem; color:var(--primary);">{acts}</div>
                        <div style="font-size:0.72rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.05em;">Activities</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">লিডারবোর্ডে এখনো ডাটা যুক্ত হয়নি। সাইডবারে নাম ও রোল দিয়ে কুইজ বা চ্যাট শুরু করুন!</div>', unsafe_allow_html=True)
