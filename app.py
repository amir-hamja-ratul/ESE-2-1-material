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
    page_title="EduHub - Academic AI Assistant",
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

# Global Network Connection Status Detector
components.html("""
<div id="net-status-banner" style="
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100%;
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    color: white; text-align: center;
    padding: 10px; font-weight: 700; font-family: sans-serif;
    z-index: 999999; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    font-size: 0.9rem;
">
    📡 Offline Mode: ইন্টারনেট কানেকশন বিচ্ছিন্ন! আপনি সেভ করা অফলাইন PDF পড়তে পারবেন।
</div>

<script>
function updateOnlineStatus() {
    var banner = document.getElementById("net-status-banner");
    if (!navigator.onLine) {
        banner.style.display = "block";
    } else {
        banner.style.display = "none";
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
# 4. CUSTOM STYLING
# ==========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --violet: #6D5DFC; --violet-deep: #4C3FD7;
        --surface: #FFFFFF; --border: #E7E9F3; --text-muted: #6B7186;
        --radius-xl: 22px; --radius-lg: 16px; --radius-md: 12px;
        --shadow-glow: 0 10px 30px -10px rgba(109, 93, 252, 0.35);
    }

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }

    .stApp { background: #FAFBFF; }
    .block-container { padding-top: 1.8rem !important; max-width: 1180px; }

    #MainMenu, footer, [data-testid="stDeployButton"] { visibility: hidden; height: 0; }
    header[data-testid="stHeader"] { background: transparent !important; }

    .header-box {
        background: linear-gradient(155deg, #05070F 0%, #0B0F1E 50%, #131A2E 100%);
        padding: 30px 24px; border-radius: var(--radius-xl); text-align: center; color: white;
        margin-bottom: 20px; box-shadow: var(--shadow-glow); border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .header-box h2 { color: #FFFFFF !important; font-size: 1.75rem; margin: 0 0 10px 0; }
    
    .uni-logo-corner {
        display: block; margin: 0 auto 12px auto; width: 72px; height: 72px; border-radius: 16px;
        background: rgba(255,255,255,0.95); padding: 5px; box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        object-fit: contain;
    }

    .badge {
        background: rgba(255,255,255,0.08); backdrop-filter: blur(10px); color: #E4E7FF !important;
        font-weight: 600; font-size: 0.8rem; padding: 6px 18px; border-radius: 30px;
        border: 1px solid rgba(255,255,255,0.15); display: inline-block;
    }

    .course-card {
        background: linear-gradient(135deg, #6D5DFC 0%, #4C3FD7 100%);
        padding: 18px 24px; border-radius: var(--radius-lg); color: white; margin-bottom: 20px;
        box-shadow: var(--shadow-glow);
    }
    .course-card h1 { color: #FFFFFF !important; font-size: 1.45rem; margin: 0; }

    .metric-card {
        background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
        padding: 16px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }
    .metric-card-val {
        font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700;
        color: #6D5DFC; margin-bottom: 2px;
    }
    .metric-card-lbl { font-size: 0.75rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; }

    div[data-testid="stRadio"] > div {
        display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
        overflow-x: auto !important; gap: 8px !important; padding: 4px 2px 14px 2px !important;
    }
    div[data-testid="stRadio"] input[type="radio"] { display: none !important; }
    div[data-testid="stRadio"] label {
        background-color: var(--surface) !important; border: 1px solid var(--border) !important;
        border-radius: 25px !important; padding: 8px 18px !important; color: #454A5E !important;
        font-weight: 600 !important; font-size: 0.85rem !important; cursor: pointer !important;
        white-space: nowrap !important; transition: all 0.2s ease;
    }
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) {
        background: linear-gradient(135deg, #6D5DFC 0%, #4C3FD7 100%) !important; color: #FFFFFF !important;
    }
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) p { color: #FFFFFF !important; font-weight: 700 !important; }

    .stButton > button {
        background: linear-gradient(135deg, #6D5DFC 0%, #4C3FD7 100%) !important; color: white !important;
        border-radius: 12px !important; padding: 10px 20px !important; font-weight: 700 !important; border: none !important;
    }

    .skeleton-wrap { padding: 16px; background: #FFF; border-radius: 12px; border: 1px solid #E7E9F3; margin: 10px 0; }
    .skeleton-badge { font-weight: 600; color: #6D5DFC; margin-bottom: 12px; font-size: 0.85rem; }
    .skeleton-line { height: 12px; background: #E7E9F3; margin-bottom: 8px; border-radius: 6px; }
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
    _logo_html = f'<img src="data:image/png;base64,{_logo_b64}" class="uni-logo-corner">'
else:
    _logo_html = '<img src="https://i.ibb.co.com/8DstCsX1/attachment-158389628.png" class="uni-logo-corner">'

st.markdown(f"""
    <div class="header-box">
        {_logo_html}
        <h2>🌱 Department of Environmental Science and Engineering</h2>
        <span class="badge">📚 Academic Resource & Smart AI Workspace</span>
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
    st.markdown("<h3 style='text-align: center;'>Workspace Navigation</h3>", unsafe_allow_html=True)
    
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
    st.markdown("👤 **YOUR PROFILE**", unsafe_allow_html=True)
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
        <h1>🎓 {selected_code}: {selected_title}</h1>
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
    st.error("⚠️ GOOGLE_API_KEY পাওয়া যায়নি! Streamlit Secrets বা Environment Variable-এ যুক্ত করুন।")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

local_pdfs = glob.glob(os.path.join(course_folder, "*.pdf"))
raw_text, total_pages = "", 0
files_count = len(local_pdfs)

if local_pdfs:
    mtimes = tuple(os.path.getmtime(p) for p in local_pdfs)
    raw_text, total_pages = extract_text_from_local_pdfs(tuple(local_pdfs), mtimes)

if raw_text.strip():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-card-val">{files_count}</div><div class="metric-card-lbl">📂 Documents</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-card-val">{total_pages}</div><div class="metric-card-lbl">📄 Total Pages</div></div>', unsafe_allow_html=True)
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
    st.subheader(f"📖 View & Download - {selected_code}")
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
                color: white; padding: 11px 20px; border: none; border-radius: 12px;
                font-weight: 700; font-size: 0.9rem; cursor: pointer; width: 100%;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
            ">
                💾 Save for Offline Reading
            </button>
            <p id="save-status" style="margin-top: 6px; font-weight: 600; color: #10B981; text-align: center; font-size: 0.85rem;"></p>

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
            components.html(save_offline_html, height=75)

        st.divider()
        display_pdf(selected_pdf)
    else:
        st.warning("⚠️ এই কোর্সের জন্য কোনো স্থানীয় PDF ফাইল খুঁজে পাওয়া যায়নি।")

# ==========================================================
# TAB 2: 📲 OFFLINE SAVED PDFS (BROWSER INDEXEDDB VIEWER)
# ==========================================================
elif tab_selection == "📲 Offline Saved PDFs":
    st.subheader("📲 Course-Wise Offline PDF Manager")
    st.caption("🌐 নেট কানেকশন না থাকলেও পূর্বে সেভ করা PDF কোর্স অনুযায়ী বেছে পড়তে পারবেন।")

    courses_js_array = str(list(COURSES.keys()))

    offline_manager_html = f"""
    <div style="background: #ffffff; padding: 20px; border-radius: 16px; border: 1px solid #E7E9F3;">
        <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 20px; flex-wrap: wrap;">
            <label style="font-weight: 700; color: #10121C;">📂 Select Course:</label>
            <select id="courseFilter" onchange="loadOfflinePDFs()" style="
                padding: 10px 16px; border-radius: 10px; border: 1px solid #6D5DFC;
                font-weight: 600; background: #F6F7FB; outline: none; cursor: pointer;
            ">
                <option value="ALL">-- ALL SAVED COURSES --</option>
            </select>
            <button onclick="loadOfflinePDFs()" style="
                background: linear-gradient(135deg, #6D5DFC 0%, #4C3FD7 100%);
                color: white; padding: 10px 18px; border: none; border-radius: 10px;
                font-weight: 700; cursor: pointer;
            ">
                🔄 Refresh List
            </button>
            <button onclick="clearAllOfflineData()" style="
                background: #EF4444; color: white; padding: 10px 18px; border: none;
                border-radius: 10px; font-weight: 700; cursor: pointer; margin-left: auto;
            ">
                🗑️ Clear All Saved PDFs
            </button>
        </div>

        <div id="status-msg" style="font-weight: 600; margin-bottom: 15px; color: #6D5DFC;"></div>
        <div id="pdf-display-area"></div>
    </div>

    <script>
    const courseList = {courses_js_array};
    
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
                statusDiv.innerHTML = "❌ কোনো সেভ করা PDF পাওয়া যায়নি। 'View & Download' ট্যাব থেকে আগে সেভ করুন।";
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
                    statusDiv.innerHTML = "⚠️ <b>" + selectedCourse + "</b> কোর্সের কোনো সেভ করা অফলাইন ফাইল পাওয়া যায়নি।";
                }} else {{
                    statusDiv.innerHTML = "✅ মোট <b>" + filtered.length + "</b> টি অফলাইন PDF পাওয়া গেছে:";
                    filtered.forEach(item => {{
                        let card = document.createElement("div");
                        card.style.cssText = "background: #F9FAFB; border: 1px solid #E7E9F3; border-radius: 12px; padding: 16px; margin-bottom: 20px;";
                        
                        let header = document.createElement("div");
                        header.style.cssText = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;";
                        header.innerHTML = "<div><h4 style='margin:0; color:#10121C;'>📄 " + item.file_name + "</h4><small style='color:#6B7186;'>Course: " + item.course_code + " | Saved on: " + (item.saved_at || 'N/A') + "</small></div>";
                        
                        let delBtn = document.createElement("button");
                        delBtn.innerText = "🗑️ Delete";
                        delBtn.style.cssText = "background:#EF4444; color:white; border:none; padding:6px 12px; border-radius:8px; font-weight:600; cursor:pointer;";
                        delBtn.onclick = function() {{ deleteOfflinePDF(item.id); }};
                        
                        header.appendChild(delBtn);
                        card.appendChild(header);

                        let iframe = document.createElement("iframe");
                        iframe.src = "data:application/pdf;base64," + item.base64;
                        iframe.style.cssText = "width: 100%; height: 600px; border: 1px solid #CBD5E1; border-radius: 8px;";
                        
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
    components.html(offline_manager_html, height=750, scrolling=True)

# ==========================================================
# TAB 3: 💬 AI Q&A
# ==========================================================
elif tab_selection == "💬 AI Q&A":
    col_title, col_clear = st.columns([4, 1])
    with col_title:
        st.subheader(f"💬 AI Study Assistant - {selected_code}")
    with col_clear:
        if st.button("🧹 Clear Chat"):
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
    st.subheader(f"📝 Auto Notes & Summary Generator - {selected_code}")
    
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
            st.warning("⚠️ নোট তৈরি করতে ফাইল টেক্সট প্রয়োজন।")

# ==========================================================
# TAB 5: 🎯 EXAM QUIZ
# ==========================================================
elif tab_selection == "🎯 Exam Quiz":
    st.subheader(f"🎯 Interactive Exam Quiz - {selected_code}")
    
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
            st.warning("⚠️ কুইজ তৈরি করতে ডকুমেন্ট টেক্সট পাওয়া যায়নি।")

# ==========================================================
# TAB 6: 📈 MY PROGRESS
# ==========================================================
elif tab_selection == "📈 My Progress":
    st.subheader("📈 Personal Activity & Progress Tracker")
    sid = st.session_state.get("student_id")
    
    if sid:
        streak = get_study_streak(sid)
        total_act = get_total_activities(sid)
        progress = get_course_progress(sid)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="metric-card-val">🔥 {streak} Days</div><div class="metric-card-lbl">Study Streak</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="metric-card-val">⚡ {total_act}</div><div class="metric-card-lbl">Total Actions</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>### 📊 Course Activity Distribution", unsafe_allow_html=True)
        if progress:
            df = pd.DataFrame(list(progress.items()), columns=["Course Code", "Total Activities"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("এখনো কোন এক্টিভিটি রেকর্ড হয়নি। পড়ালেখা ও চ্যাট শুরু করলে ডাটা আপডেট হবে।")
    else:
        st.info("👉 আপনার প্রতিদিনের অগ্রগতি সেভ করতে সাইডবারে রোল নম্বর যোগ করুন।")

# ==========================================================
# TAB 7: 📊 LEADERBOARD
# ==========================================================
elif tab_selection == "📊 Leaderboard":
    st.subheader("📊 Top Active Student Leaderboard")
    board_data = get_leaderboard()
    
    if board_data:
        df_lb = pd.DataFrame(board_data, columns=["Student Name", "Roll Number", "Activities", "Last Active"])
        st.dataframe(df_lb, use_container_width=True)
    else:
        st.info("লিডারবোর্ডে এখনো ডাটা যুক্ত হয়নি। সাইডবারে নাম ও রোল দিয়ে কুইজ বা চ্যাট শুরু করুন!")
