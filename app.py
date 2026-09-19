import streamlit as st
import os
import glob
import io
import sqlite3
import hashlib
import base64
from datetime import date, datetime
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

# Global Network Connection Status Detector
components.html("""
<div id="net-status-banner" style="
    display: none;
    position: fixed;
    top: 0; left: 0; width: 100%;
    background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
    color: white; text-align: center;
    padding: 10px; font-weight: 700; font-family: 'Plus Jakarta Sans', sans-serif;
    z-index: 999999; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    font-size: 0.9rem;
">
    📡 Offline Mode: ইন্টারনেট কানেকশন বিচ্ছিন্ন! আপনি সেভ করা অফলাইন PDF পড়তে পারবেন।
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
# 2. ACTIVITY LOGGING & DATABASE (SQLite Database)
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
# 3. HELPER FUNCTIONS & LAZY-LOADED AI RAG
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
        import fitz  # Lazy import PyMuPDF
        from PIL import Image  # Lazy import PIL
        
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

@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm(_api_key):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=_api_key, temperature=0.3)

@st.cache_data(show_spinner=False)
def extract_text_from_local_pdfs(pdf_paths, cache_key):
    from PyPDF2 import PdfReader
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
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    embeddings = get_embeddings_model()
    return FAISS.from_texts(chunks, embedding=embeddings)

# ==========================================================
# 4. ADVANCED MODERN SAAS STYLING
# ==========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

    :root {
        --primary: #6366F1;
        --primary-gradient: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        --accent-glow: rgba(99, 102, 241, 0.25);
        --bg-main: #F1F5F9;
        --surface: #FFFFFF;
        --border-color: #E2E8F0;
        --text-dark: #0F172A;
        --text-muted: #64748B;
        --radius-lg: 20px;
        --radius-md: 14px;
        --shadow-soft: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
    }

    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }
    h1, h2, h3, h4 { 
        font-family: 'Outfit', sans-serif !important; 
        letter-spacing: -0.02em;
    }

    .stApp { 
        background: linear-gradient(135deg, #EEF2FF 0%, #F8FAFC 50%, #E0E7FF 100%);
        background-attachment: fixed;
    }
    .block-container { padding-top: 1.5rem !important; max-width: 1240px; }

    #MainMenu, footer, [data-testid="stDeployButton"] { visibility: hidden; height: 0; }
    header[data-testid="stHeader"] { background: transparent !important; }

    .header-box {
        background: linear-gradient(135deg, #020617 0%, #0F172A 60%, #1E293B 100%) !important;
        padding: 32px 24px;
        border-radius: var(--radius-lg);
        text-align: center;
        color: #FFFFFF !important;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 20px 35px -10px rgba(2, 6, 23, 0.5);
    }
    
    .uni-logo-wrapper {
        display: inline-block;
        background: #FFFFFF;
        padding: 8px 14px;
        border-radius: 18px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.35);
        margin-bottom: 14px;
    }
    .uni-logo-corner {
        height: 68px; 
        width: auto;
        display: block;
        margin: 0 auto;
        object-fit: contain;
        border-radius: 4px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }

    .header-box h2 { 
        color: #FFFFFF !important; 
        font-size: 1.85rem !important; 
        font-weight: 800 !important;
        margin: 0; 
        letter-spacing: 0.3px;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
    }

    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(20px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(20px) saturate(180%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.6) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.03) !important;
    }

    section[data-testid="stSidebar"] .stTextInput input, 
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background: rgba(255, 255, 255, 0.65) !important;
        border: 1px solid rgba(203, 213, 225, 0.7) !important;
        backdrop-filter: blur(8px) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.02) !important;
        transition: all 0.25s ease !important;
    }

    .course-card {
        background: var(--primary-gradient);
        padding: 20px 28px; 
        border-radius: var(--radius-md); 
        color: white; 
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.35);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .course-card h1 { 
        color: #FFFFFF !important; 
        font-size: 1.45rem; 
        margin: 0; 
        font-weight: 700;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.75); 
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.8); 
        border-radius: var(--radius-md);
        padding: 20px; 
        text-align: center; 
        box-shadow: var(--shadow-soft);
        transition: all 0.25s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #C7D2FE;
        box-shadow: 0 15px 30px -10px rgba(99, 102, 241, 0.15);
    }
    .metric-card-val {
        font-family: 'Outfit', sans-serif; 
        font-size: 2.2rem; 
        font-weight: 800;
        color: #4F46E5; 
        margin-bottom: 2px;
        line-height: 1;
    }
    .metric-card-lbl { 
        font-size: 0.78rem; 
        font-weight: 700; 
        color: var(--text-muted); 
        text-transform: uppercase; 
        letter-spacing: 0.05em;
    }

    div[data-testid="stRadio"] {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        padding: 6px;
        border-radius: 18px;
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 20px;
    }
    div[data-testid="stRadio"] > div {
        display: flex !important; 
        flex-direction: row !important; 
        flex-wrap: nowrap !important;
        overflow-x: auto !important; 
        gap: 6px !important; 
        padding: 2px !important;
        scrollbar-width: none;
    }
    div[data-testid="stRadio"] > div::-webkit-scrollbar { display: none; }
    div[data-testid="stRadio"] input[type="radio"] { display: none !important; }
    div[data-testid="stRadio"] label {
        background-color: transparent !important; 
        border: none !important;
        border-radius: 12px !important; 
        padding: 10px 20px !important; 
        color: #475569 !important;
        font-weight: 600 !important; 
        font-size: 0.88rem !important; 
        cursor: pointer !important;
        white-space: nowrap !important; 
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        margin: 0 !important;
    }
    div[data-testid="stRadio"] label:hover {
        color: #0F172A !important;
        background: rgba(255,255,255,0.7) !important;
    }
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) {
        background: #FFFFFF !important; 
        color: #4F46E5 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }
    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) p { 
        color: #4F46E5 !important; 
        font-weight: 700 !important; 
    }

    .stButton > button {
        background: var(--primary-gradient) !important; 
        color: white !important;
        border-radius: 12px !important; 
        padding: 10px 24px !important; 
        font-weight: 700 !important; 
        border: none !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.35) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# 5. HEADER & NAVIGATION
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
        <div class="uni-logo-wrapper">
            {_logo_img}
        </div>
        <h2>Department of Environmental Science and Engineering</h2>
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
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] img {
            width: 120px !important;
            height: 120px !important;
            border-radius: 50% !important;
            border: 4px solid #2e7d32 !important;
            object-fit: cover !important;
            display: block;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    sidebar_logo_png = os.path.join(ASSETS_DIR, "ese10_logo.png")
    sidebar_logo_jpg = os.path.join(ASSETS_DIR, "ese10_logo.jpg")
    
    logo_file = sidebar_logo_png if os.path.exists(sidebar_logo_png) else (sidebar_logo_jpg if os.path.exists(sidebar_logo_jpg) else None)
    
    if logo_file:
        try:
            with open(logo_file, "rb") as f:
                b64_logo = base64.b64encode(f.read()).decode()
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 18px; padding: 10px; background: rgba(255, 255, 255, 0.7); border-radius: 16px; border: 1px solid rgba(226, 232, 240, 0.8); box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
                    <img src="data:image/png;base64,{b64_logo}" style="max-width: 90%; height: auto; border-radius: 8px;">
                </div>
            """, unsafe_allow_html=True)
        except Exception:
            pass

    st.markdown("<h3 style='font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 12px;'>Workspace Navigation</h3>", unsafe_allow_html=True)
    
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
    st.markdown("<p style='font-size: 0.8rem; font-weight: 800; color: #64748B; letter-spacing: 0.05em;'>YOUR PROFILE</p>", unsafe_allow_html=True)
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
            <span style="font-size: 0.8rem; text-transform: uppercase; opacity: 0.85; font-weight: 700; letter-spacing: 0.05em;">Active Course Material</span>
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
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-card-val">{files_count}</div><div class="metric-card-lbl">📁 Documents Loaded</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-card-val">{total_pages}</div><div class="metric-card-lbl">📄 Total Pages Indexed</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Navigation without Exam Quiz and My Progress
tab_selection = st.radio(
    "Navigation Tabs",
    ["📖 View & Download", "📲 Offline Saved PDFs", "💬 AI Q&A", "📝 Smart Summary", "📊 Leaderboard"],
    horizontal=True,
    label_visibility="collapsed"
)

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
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25); font-family: 'Plus Jakarta Sans', sans-serif;
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
        st.warning("⚠️ এই কোর্সের জন্য কোনো স্থানীয় PDF ফাইল খুঁজে পাওয়া যায়নি।")

# ==========================================================
# TAB 2: 📲 OFFLINE SAVED PDFS
# ==========================================================
elif tab_selection == "📲 Offline Saved PDFs":
    st.subheader("📲 Course-Wise Offline PDF Manager")
    st.caption("🌐 নেট কানেকশন না থাকলেও পূর্বে সেভ করা PDF কোর্স অনুযায়ী বেছে পড়তে পারবেন।")

    courses_js_array = str(list(COURSES.keys()))

    offline_manager_html = f"""
    <!-- PDF.js Library CDN -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <script>
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    </script>

    <div style="background: rgba(255, 255, 255, 0.8); backdrop-filter: blur(12px); padding: 22px; border-radius: 16px; border: 1px solid #E2E8F0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
        <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 20px; flex-wrap: wrap;">
            <label style="font-weight: 700; color: #0F172A; font-family: sans-serif;">📂 Select Course:</label>
            <select id="courseFilter" onchange="loadOfflinePDFs()" style="
                padding: 10px 16px; border-radius: 10px; border: 1px solid #6366F1;
                font-weight: 600; background: #F8FAFC; outline: none; cursor: pointer; font-family: sans-serif;
            ">
                <option value="ALL">-- ALL SAVED COURSES --</option>
            </select>
            <button onclick="loadOfflinePDFs()" style="
                background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
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

        <div id="status-msg" style="font-weight: 600; margin-bottom: 15px; color: #4F46E5; font-family: sans-serif;"></div>
        <div id="pdf-display-area"></div>
    </div>

    <script>
    const courseList = {courses_js_array};

    function renderPdfPages(base64Data, container) {{
        try {{
            let binaryStr = atob(base64Data);
            let len = binaryStr.length;
            let bytes = new Uint8Array(len);
            for (let i = 0; i < len; i++) {{
                bytes[i] = binaryStr.charCodeAt(i);
            }}

            pdfjsLib.getDocument({{data: bytes}}).promise.then(function(pdf) {{
                let pdfViewerDiv = document.createElement("div");
                pdfViewerDiv.style.cssText = "max-height: 650px; overflow-y: auto; padding: 15px; background: #E2E8F0; border-radius: 8px;";
                container.appendChild(pdfViewerDiv);

                let renderPages = async () => {{
                    for (let num = 1; num <= pdf.numPages; num++) {{
                        let page = await pdf.getPage(num);
                        let viewport = page.getViewport({{scale: 1.2}});
                        let canvas = document.createElement('canvas');
                        let ctx = canvas.getContext('2d');
                        canvas.height = viewport.height;
                        canvas.width = viewport.width;
                        canvas.style.cssText = "width: 100%; max-width: 800px; display: block; margin: 0 auto 15px auto; border-radius: 6px; box-shadow: 0 4px 10px rgba(0,0,0,0.15);";
                        
                        pdfViewerDiv.appendChild(canvas);
                        await page.render({{canvasContext: ctx, viewport: viewport}}).promise;
                    }}
                }};
                renderPages();
            }}).catch(function(err) {{
                container.innerHTML += "<p style='color:red; font-family:sans-serif;'>❌ PDF রেন্ডার করতে সমস্যা হয়েছে: " + err.message + "</p>";
            }});
        }} catch(e) {{
            container.innerHTML += "<p style='color:red; font-family:sans-serif;'>❌ ফাইল লোড করতে সমস্যা হয়েছে।</p>";
        }}
    }}

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
                statusDiv.innerText = "ℹ️ কোনো সেভ করা অফলাইন ফাইল পাওয়া যায়নি।";
                return;
            }}
            let tx = db.transaction("pdf_store", "readonly");
            let store = tx.objectStore("pdf_store");
            let getAllReq = store.getAll();

            getAllReq.onsuccess = function() {{
                let items = getAllReq.result;
                if (!items || items.length === 0) {{
                    statusDiv.innerText = "ℹ️ কোনো সেভ করা অফলাইন ফাইল পাওয়া যায়নি।";
                    return;
                }}
                statusDiv.innerText = "✅ অফলাইন ফাইলসমূহ সফলভাবে লোড হয়েছে!";
                
                items.forEach(item => {{
                    if (selectedCourse === "ALL" || item.course_code === selectedCourse) {{
                        let card = document.createElement("div");
                        card.style.cssText = "background: white; padding: 18px; border-radius: 12px; margin-bottom: 20px; border: 1px solid #CBD5E1; box-shadow: 0 4px 10px rgba(0,0,0,0.05);";
                        card.innerHTML = "<h3 style='margin-top:0; color:#0F172A; font-family:sans-serif;'>" + item.course_code + ": " + item.file_name + "</h3><p style='color:#64748B; font-size:0.85rem; font-family:sans-serif;'>Saved on: " + item.saved_at + "</p>";
                        container.appendChild(card);
                        renderPdfPages(item.base64, card);
                    }}
                }});
            }};
        }};
    }}

    function clearAllOfflineData() {{
        if(confirm("আপনি কি নিশ্চিত যে সকল অফলাইন ফাইল মুছে ফেলতে চান?")) {{
            let request = indexedDB.deleteDatabase("EduHubOfflineDB");
            request.onsuccess = function() {{
                alert("সব সেভ করা অফলাইন ফাইল মুছে ফেলা হয়েছে!");
                location.reload();
            }};
        }}
    }}

    setTimeout(() => {{
        populateDropdown();
        loadOfflinePDFs();
    }}, 300);
    </script>
    """
    components.html(offline_manager_html, height=800, scrolling=True)

# ==========================================================
# TAB 3: 💬 AI Q&A
# ==========================================================
elif tab_selection == "💬 AI Q&A":
    st.subheader(f"💬 AI Assistant for {selected_code}")
    if not raw_text.strip():
        st.info("ℹ️ এই কোর্সের জন্য কোনো ফাইল/টেক্সট পাওয়া যায়নি। আগে PDF আপলোড অথবা যুক্ত করুন।")
    else:
        text_hash = hashlib.md5(raw_text.encode()).hexdigest()
        with st.spinner("🔍 Vector Store তৈরি হচ্ছে..."):
            vector_store = build_vector_store(selected_code, text_hash, raw_text)
        
        user_q = st.text_input("❓ এই কোর্স সম্পর্কিত যেকোনো প্রশ্ন লিখুন (বাংলা/English):")
        if st.button("উত্তর দিন", key="btn_ask") and user_q:
            track("Asked Question", selected_code)
            docs = vector_store.similarity_search(user_q, k=3)
            llm = get_llm(api_key)
            with st.spinner("🤖 AI উত্তর তৈরি করছে..."):
                ans = ask_gemini(llm, docs, user_q)
            st.markdown(f"**🤖 উত্তর:**\n\n{ans}")

# ==========================================================
# TAB 4: 📝 SMART SUMMARY
# ==========================================================
elif tab_selection == "📝 Smart Summary":
    st.subheader(f"📝 Smart Summary - {selected_code}")
    if not raw_text.strip():
        st.info("ℹ️ সারসংক্ষেপ তৈরির জন্য কোনো টেক্সট পাওয়া যায়নি।")
    else:
        if st.button("⚡ সারসংক্ষেপ তৈরি করুন"):
            track("Generated Summary", selected_code)
            llm = get_llm(api_key)
            prompt = f"Provide a comprehensive, well-structured Bengali summary of key concepts from the following course material:\n\n{raw_text[:4000]}"
            with st.spinner("📝 সামারি তৈরি হচ্ছে..."):
                try:
                    res = llm.invoke(prompt)
                    st.markdown(res.content if hasattr(res, 'content') else str(res))
                except Exception as e:
                    st.error(f"Error: {e}")

# ==========================================================
# TAB 5: 📊 LEADERBOARD
# ==========================================================
elif tab_selection == "📊 Leaderboard":
    st.subheader("📊 Top Active Students Leaderboard")
    leaderboard = get_leaderboard()
    if leaderboard:
        st.table([
            {"Rank": idx + 1, "Name": row[0] or "Anonymous", "Roll/ID": row[1], "Total Activities": row[2], "Last Active": row[3]}
            for idx, row in enumerate(leaderboard)
        ])
    else:
        st.info("লিডারবোর্ডে এখনো কোনো ডেটা যুক্ত হয়নি।")
