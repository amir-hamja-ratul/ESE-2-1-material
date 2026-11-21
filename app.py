import os
import glob
import io
import sqlite3
import base64
from datetime import date, datetime
import streamlit as st
import streamlit.components.v1 as components

# ==========================================================
# 1. APP CONFIGURATION & FAST SETUP
# ==========================================================
APP_CONFIG = {
    "title": "EduHub - Academic Workspace",
    "icon": "🎓",
    "department": "Department of Environmental Science and Engineering",
    "default_logo_url": "https://i.ibb.co.com/8DstCsX1/attachment-158389628.png",
    "admin_key": st.secrets.get("ADMIN_PASS", "285277"),
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_model": "gemini-3.6-flash",
}

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
    "MEQ": "Mid Exam Questions",
}

st.set_page_config(
    page_title=APP_CONFIG["title"],
    page_icon=APP_CONFIG["icon"],
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DB_PATH = os.path.join(BASE_DIR, "eduhub_progress.db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Favicon & Global Offline Detector
st.markdown(
    f"""
    <head>
        <link rel="apple-touch-icon" href="{APP_CONFIG['default_logo_url']}">
        <link rel="icon" href="{APP_CONFIG['default_logo_url']}">
    </head>
    """,
    unsafe_allow_html=True,
)

components.html(
    """
    <div id="net-status-banner" style="
        display: none;
        position: fixed;
        top: 0; left: 0; width: 100%;
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
        color: #FFFFFF; text-align: center;
        padding: 9px 14px; font-weight: 700;
        font-family: 'Plus Jakarta Sans', sans-serif;
        z-index: 999999; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        font-size: 0.88rem;
    ">
        📡 Offline Mode: ইন্টারনেট সংযোগ বিচ্ছিন্ন! সেভ করা অফলাইন PDF পড়তে পারবেন।
    </div>
    <script>
    function updateOnlineStatus() {
        const el = document.getElementById("net-status-banner");
        if (el) el.style.display = navigator.onLine ? "none" : "block";
    }
    window.addEventListener("online", updateOnlineStatus);
    window.addEventListener("offline", updateOnlineStatus);
    updateOnlineStatus();
    </script>
    """,
    height=0,
)

# ==========================================================
# 2. DATABASE & ACTIVITY LOGGER
# ==========================================================
@st.cache_resource(show_spinner=False)
def init_db_schema():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                student_name TEXT,
                course_code TEXT NOT NULL,
                action TEXT NOT NULL,
                activity_date TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_act_sid ON activity_log(student_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_act_course ON activity_log(course_code)")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def get_db_connection():
    init_db_schema()
    try:
        return sqlite3.connect(DB_PATH, check_same_thread=False)
    except Exception as e:
        st.error(f"Database Error: {e}")
        return None


def log_activity(student_id: str, student_name: str, course_code: str, action: str):
    if not student_id:
        return
    conn = get_db_connection()
    if conn:
        try:
            conn.execute(
                """
                INSERT INTO activity_log
                (student_id, student_name, course_code, action, activity_date, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    student_id,
                    student_name,
                    course_code,
                    action,
                    date.today().isoformat(),
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()


def track_once(action: str, course_code: str):
    sid = st.session_state.get("student_id")
    sname = st.session_state.get("student_name", "")
    if not sid:
        return
    key = f"trk_{sid}_{course_code}_{action}"
    if not st.session_state.get(key):
        log_activity(sid, sname, course_code, action)
        st.session_state[key] = True


# ==========================================================
# 3. CACHED PDF & AI HELPERS
# ==========================================================
@st.cache_data(show_spinner=False)
def get_pdf_page_count(file_path: str, mtime: float) -> int:
    try:
        import fitz
        with fitz.open(file_path) as doc:
            return len(doc)
    except Exception:
        return 0


@st.cache_data(show_spinner=False)
def render_pdf_page_png(file_path: str, mtime: float, page_index: int, dpi: int) -> bytes:
    import fitz
    with fitz.open(file_path) as doc:
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=dpi)
        return pix.tobytes("png")


@st.cache_data(show_spinner=False)
def extract_text_from_local_pdfs(pdf_paths: tuple, mtimes: tuple):
    from PyPDF2 import PdfReader
    raw, pages = "", 0
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            pages += len(reader.pages)
            for page in reader.pages:
                txt = page.extract_text()
                if txt:
                    raw += txt + "\n"
        except Exception:
            pass
    return raw, pages


@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=APP_CONFIG["embedding_model"])


@st.cache_resource(show_spinner=False)
def get_llm(_api_key: str):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=APP_CONFIG["llm_model"],
        google_api_key=_api_key,
        temperature=0.3,
    )


def display_pdf_fast(file_path: str):
    mtime = os.path.getmtime(file_path)
    total_pages = get_pdf_page_count(file_path, mtime)
    if total_pages <= 0:
        st.warning("এই PDF-এর পেজ সংখ্যা পড়া যাচ্ছে না বা ফাইলটি খালি।")
        return

    st.markdown(f"**📖 মোট পেজ:** `{total_pages}` — দ্রুত ওপেন করতে পেজ রেঞ্জ এবং কোয়ালিটি বেছে নিন")

    c_start, c_end, c_dpi = st.columns([1, 1, 1.2])
    with c_start:
        start_p = st.number_input("শুরু পেজ", min_value=1, max_value=total_pages, value=1, step=1, key="pdf_start_p")
    with c_end:
        default_end = min(total_pages, start_p + 2)
        end_p = st.number_input("শেষ পেজ", min_value=start_p, max_value=total_pages, value=default_end, step=1, key="pdf_end_p")
    with c_dpi:
        quality_label = st.selectbox(
            "রেন্ডার স্পিড / কোয়ালিটি",
            ["⚡ Ultra Fast (85 DPI)", "⚖️ Balanced (110 DPI)", "🔍 Sharp (145 DPI)"],
            index=0,
            key="pdf_quality_sel",
        )
    dpi_map = {
        "⚡ Ultra Fast (85 DPI)": 85,
        "⚖️ Balanced (110 DPI)": 110,
        "🔍 Sharp (145 DPI)": 145,
    }
    dpi = dpi_map[quality_label]

    for p_num in range(start_p, end_p + 1):
        img_bytes = render_pdf_page_png(file_path, mtime, p_num - 1, dpi)
        st.image(img_bytes, caption=f"Page {p_num} of {total_pages}", use_container_width=True)


# ==========================================================
# 4. SMART & HIGH-CONTRAST UI/UX STYLING
# ==========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@600;700;800&display=swap');

        :root {
            --brand-primary: #3730A3;
            --brand-accent: #4F46E5;
            --brand-soft: #EEF2FF;
            --ink-primary: #0F172A;
            --ink-secondary: #334155;
            --ink-muted: #64748B;
            --surface-card: #FFFFFF;
            --surface-border: #CBD5E1;
            --radius-lg: 18px;
            --radius-md: 14px;
            --radius-sm: 10px;
        }

        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: var(--ink-primary);
        }

        h1, h2, h3, h4 {
            font-family: 'Outfit', sans-serif !important;
            color: var(--ink-primary) !important;
            letter-spacing: -0.02em;
        }

        .stApp {
            background: #F8FAFC;
        }

        .block-container {
            padding-top: 1.25rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 1220px;
        }

        #MainMenu, footer, [data-testid="stDeployButton"] {
            visibility: hidden;
            height: 0;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }

        /* Hero Header with High Contrast */
        .hero-banner {
            background: linear-gradient(120deg, #0F172A 0%, #1E1B4B 65%, #312E81 100%);
            padding: 26px 24px;
            border-radius: var(--radius-lg);
            text-align: center;
            color: #FFFFFF !important;
            margin-bottom: 18px;
            border: 1px solid rgba(199, 210, 254, 0.25);
            box-shadow: 0 12px 28px -8px rgba(15, 23, 42, 0.25);
        }
        .hero-logo-box {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #FFFFFF;
            padding: 8px 14px;
            border-radius: 14px;
            margin-bottom: 12px;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.22);
        }
        .hero-logo-box img {
            height: 56px;
            width: auto;
            object-fit: contain;
            display: block;
        }
        .hero-banner h2 {
            color: #FFFFFF !important;
            font-size: 1.65rem !important;
            font-weight: 800 !important;
            margin: 0 0 6px 0;
        }
        .hero-subtitle {
            color: #E0E7FF !important;
            font-size: 0.9rem;
            font-weight: 500;
            margin: 0;
        }

        /* Active Topic / Course Header Card - Crisp Contrast */
        .topic-banner {
            background: #FFFFFF;
            border-left: 6px solid var(--brand-accent);
            border-top: 1px solid #E2E8F0;
            border-right: 1px solid #E2E8F0;
            border-bottom: 1px solid #E2E8F0;
            padding: 18px 22px;
            border-radius: var(--radius-md);
            margin-bottom: 18px;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        }
        .topic-tag {
            display: inline-block;
            background: var(--brand-soft);
            color: var(--brand-primary);
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            padding: 4px 10px;
            border-radius: 999px;
            margin-bottom: 6px;
        }
        .topic-title {
            color: #0F172A !important;
            font-size: 1.4rem !important;
            font-weight: 800 !important;
            margin: 0 !important;
            line-height: 1.25;
        }

        /* Metric Cards */
        .stat-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: var(--radius-md);
            padding: 16px 18px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        }
        .stat-val {
            font-family: 'Outfit', sans-serif;
            font-size: 1.85rem;
            font-weight: 800;
            color: var(--brand-primary);
            line-height: 1.1;
        }
        .stat-lbl {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--ink-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-top: 4px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }

        /* Primary Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #4338CA 0%, #3730A3 100%) !important;
            color: #FFFFFF !important;
            border-radius: var(--radius-sm) !important;
            padding: 9px 18px !important;
            font-weight: 700 !important;
            border: none !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease !important;
        }
        .stButton > button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(55, 48, 163, 0.25) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# 5. HEADER & SIDEBAR NAVIGATION
# ==========================================================
def _load_logo_b64() -> str:
    logo_path = os.path.join(ASSETS_DIR, "university_logo.png")
    if os.path.exists(logo_path):
        try:
            with open(logo_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return ""
    return ""


_logo_b64 = _load_logo_b64()
_logo_html = (
    f'<img src="data:image/png;base64,{_logo_b64}" alt="Department Logo">'
    if _logo_b64
    else f'<img src="{APP_CONFIG["default_logo_url"]}" alt="Department Logo">'
)

st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-logo-box">{_logo_html}</div>
        <h2>{APP_CONFIG["department"]}</h2>
        <p class="hero-subtitle">Smart Course Materials, Fast PDF Reader & Offline Study Hub</p>
    </div>
    """,
    unsafe_allow_html=True,
)

course_options = [f"{code} - {title}" for code, title in COURSES.items()]

with st.sidebar:
    for fname in ("ese10_logo.png", "ese10_logo.jpg"):
        fpath = os.path.join(ASSETS_DIR, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "rb") as f:
                    b64_side = base64.b64encode(f.read()).decode()
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-bottom:14px;">
                        <img src="data:image/png;base64,{b64_side}"
                             style="width:96px; height:96px; border-radius:50%; object-fit:cover; border:3px solid #3730A3;">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                break
            except Exception:
                pass

    st.markdown("#### কোর্স ও নেভিগেশন")
    search_query = st.text_input("🔍 কোর্স খুঁজুন", placeholder="যেমন: hydrology, soil, lab", key="course_search_box")
    if search_query.strip():
        q = search_query.strip().lower()
        matches = [(c, t) for c, t in COURSES.items() if q in c.lower() or q in t.lower()]
        for c, t in matches[:5]:
            if st.button(f"📘 {c} — {t}", key=f"pick_{c}", use_container_width=True):
                st.session_state["course_selectbox"] = f"{c} - {t}"
                st.rerun()
        st.divider()

    selected_option = st.selectbox("📌 কোর্স নির্বাচন করুন", course_options, key="course_selectbox")
    selected_code = selected_option.split(" - ")[0]
    selected_title = COURSES[selected_code]

    st.divider()
    st.markdown("#### শিক্ষার্থী প্রোফাইল")
    student_name_input = st.text_input("নাম", placeholder="e.g. Amir Hamja Ratul", key="prof_name")
    student_roll_input = st.text_input("রোল নম্বর", placeholder="e.g. 25103406", key="prof_roll")

    if student_roll_input.strip():
        st.session_state["student_id"] = student_roll_input.strip()
        st.session_state["student_name"] = student_name_input.strip() or student_roll_input.strip()
        st.caption(f"✅ সক্রিয়: **{st.session_state['student_name']}**")
    else:
        st.session_state["student_id"] = None

    query_params = st.query_params
    admin_pass = (
        st.text_input("🔒 Admin Key", type="password")
        if query_params.get("admin") == "true"
        else ""
    )

# Active Course / Topic Title Card
st.markdown(
    f"""
    <div class="topic-banner">
        <span class="topic-tag">Active Course Topic</span>
        <h3 class="topic-title">{selected_code} — {selected_title}</h3>
    </div>
    """,
    unsafe_allow_html=True,
)

folder_code = selected_code.replace(" ", "_")
course_folder = os.path.join(DATA_DIR, folder_code)
os.makedirs(course_folder, exist_ok=True)

if admin_pass == APP_CONFIG["admin_key"]:
    st.success("⚡ Admin Mode Active")
    uploaded_files = st.file_uploader("📥 PDF আপলোড করুন", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        for u_file in uploaded_files:
            safe_name = os.path.basename(u_file.name)
            with open(os.path.join(course_folder, safe_name), "wb") as f:
                f.write(u_file.getbuffer())
        st.toast("✅ ফাইল সফলভাবে সেভ হয়েছে!", icon="🎉")
        st.rerun()

local_pdfs = sorted(glob.glob(os.path.join(course_folder, "*.pdf")))
files_count = len(local_pdfs)

if files_count > 0:
    mtimes = tuple(os.path.getmtime(p) for p in local_pdfs)
    est_pages = sum(get_pdf_page_count(p, mt) for p, mt in zip(local_pdfs, mtimes))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-val">{files_count}</div><div class="stat-lbl">📁 উপলব্ধ PDF ফাইল</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-val">{est_pages}</div><div class="stat-lbl">📄 মোট পেজ</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

# ==========================================================
# 6. TABS: VIEW/DOWNLOAD & OFFLINE MANAGER
# ==========================================================
tab_selection = st.radio(
    "Navigation",
    ["📖 View & Download", "📲 Offline Saved PDFs"],
    horizontal=True,
    label_visibility="collapsed",
)

if tab_selection == "📖 View & Download":
    st.markdown(f"#### 📖 {selected_code} — {selected_title} (ডকুমেন্ট ভিউয়ার)")
    track_once("View Document", selected_code)

    if local_pdfs:
        selected_pdf = st.selectbox("📄 PDF ফাইল বেছে নিন", local_pdfs, format_func=os.path.basename)
        pdf_name = os.path.basename(selected_pdf)

        with open(selected_pdf, "rb") as f:
            pdf_bytes = f.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")

        col_dl, col_save = st.columns(2)
        with col_dl:
            st.download_button(
                label=f"⬇️ Download {pdf_name}",
                data=pdf_bytes,
                file_name=pdf_name,
                mime="application/pdf",
                use_container_width=True,
            )

        with col_save:
            save_offline_html = f"""
            <button onclick="saveToIndexedDB()" style="
                background: linear-gradient(135deg, #059669 0%, #047857 100%);
                color: #FFFFFF; padding: 10px 18px; border: none; border-radius: 10px;
                font-weight: 700; font-size: 0.9rem; cursor: pointer; width: 100%;
                font-family: 'Plus Jakarta Sans', sans-serif;
            ">
                💾 Save for Offline Reading
            </button>
            <p id="save-status" style="margin:6px 0 0 0; font-weight:700; color:#047857; text-align:center; font-size:0.84rem;"></p>
            <script>
            function saveToIndexedDB() {{
                const statusEl = document.getElementById("save-status");
                const req = indexedDB.open("EduHubOfflineDB", 2);
                req.onupgradeneeded = function(e) {{
                    const db = e.target.result;
                    if (!db.objectStoreNames.contains("pdf_store")) {{
                        db.createObjectStore("pdf_store", {{ keyPath: "id" }});
                    }}
                }};
                req.onsuccess = function(e) {{
                    const db = e.target.result;
                    const tx = db.transaction("pdf_store", "readwrite");
                    const store = tx.objectStore("pdf_store");
                    store.put({{
                        id: "{selected_code}_{pdf_name}",
                        course_code: "{selected_code}",
                        course_title: "{selected_title}",
                        file_name: "{pdf_name}",
                        base64: "{base64_pdf}",
                        saved_at: new Date().toLocaleDateString()
                    }});
                    tx.oncomplete = function() {{
                        if (statusEl) statusEl.innerText = "✅ ব্রাউজারে অফলাইনে সেভ হয়েছে!";
                    }};
                }};
            }}
            </script>
            """
            components.html(save_offline_html, height=72)

        st.divider()
        display_pdf_fast(selected_pdf)
    else:
        st.info("ℹ️ এই কোর্সের জন্য এখনো কোনো PDF ফাইল যুক্ত করা হয়নি।")

elif tab_selection == "📲 Offline Saved PDFs":
    st.markdown("#### 📲 Offline Saved PDFs (দ্রুত অফলাইন রিডার)")
    st.caption("ইন্টারনেট না থাকলেও আগে সেভ করা PDF এখান থেকে সহজে পড়তে পারবেন।")

    courses_js_array = str(list(COURSES.keys()))
    offline_manager_html = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <script>
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    </script>

    <div style="background:#FFFFFF; padding:18px; border-radius:14px; border:1px solid #E2E8F0;">
        <div style="display:flex; gap:10px; align-items:center; margin-bottom:14px; flex-wrap:wrap;">
            <label style="font-weight:700; color:#0F172A;">কোর্স:</label>
            <select id="courseFilter" onchange="loadOfflinePDFs()" style="padding:8px 12px; border-radius:8px; border:1px solid #CBD5E1; font-weight:600; color:#0F172A;">
                <option value="ALL">সব কোর্স</option>
            </select>
            <button onclick="loadOfflinePDFs()" style="background:#3730A3; color:#FFFFFF; padding:8px 14px; border:none; border-radius:8px; font-weight:700; cursor:pointer;">
                🔄 রিফ্রেশ
            </button>
            <button onclick="clearAllOfflineData()" style="background:#DC2626; color:#FFFFFF; padding:8px 14px; border:none; border-radius:8px; font-weight:700; cursor:pointer; margin-left:auto;">
                🗑️ সব মুছুন
            </button>
        </div>
        <div id="status-msg" style="font-weight:700; margin-bottom:10px; color:#3730A3; font-size:0.9rem;"></div>
        <div id="pdf-display-area"></div>
    </div>

    <script>
    const courseList = {courses_js_array};

    function renderPdfPagesFast(base64Data, container) {{
        try {{
            const binaryStr = atob(base64Data);
            const bytes = new Uint8Array(binaryStr.length);
            for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);

            pdfjsLib.getDocument({{ data: bytes }}).promise.then(async function(pdf) {{
                const viewer = document.createElement("div");
                viewer.style.cssText = "max-height:560px; overflow-y:auto; padding:12px; background:#F1F5F9; border-radius:10px;";
                container.appendChild(viewer);

                for (let num = 1; num <= pdf.numPages; num++) {{
                    const page = await pdf.getPage(num);
                    const viewport = page.getViewport({{ scale: 1.0 }});
                    const canvas = document.createElement("canvas");
                    const ctx = canvas.getContext("2d");
                    canvas.height = viewport.height;
                    canvas.width = viewport.width;
                    canvas.style.cssText = "width:100%; max-width:760px; display:block; margin:0 auto 12px auto; border-radius:6px; background:#FFFFFF;";
                    viewer.appendChild(canvas);
                    await page.render({{ canvasContext: ctx, viewport: viewport }}).promise;
                }}
            }});
        }} catch (e) {{
            container.innerHTML += "<p style='color:#DC2626; font-weight:600;'>❌ PDF রেন্ডার করতে সমস্যা হয়েছে।</p>";
        }}
    }}

    function populateDropdown() {{
        const select = document.getElementById("courseFilter");
        if (!select || select.options.length > 1) return;
        courseList.forEach(code => {{
            const opt = document.createElement("option");
            opt.value = code;
            opt.innerText = code;
            select.appendChild(opt);
        }});
        const currentCode = "{selected_code}";
        if (courseList.includes(currentCode)) select.value = currentCode;
    }}

    function loadOfflinePDFs() {{
        const selectedCourse = document.getElementById("courseFilter").value;
        const container = document.getElementById("pdf-display-area");
        const statusDiv = document.getElementById("status-msg");
        container.innerHTML = "";
        statusDiv.innerText = "⏳ অফলাইন ফাইল লোড হচ্ছে...";

        const req = indexedDB.open("EduHubOfflineDB", 2);
        req.onsuccess = function(e) {{
            const db = e.target.result;
            if (!db.objectStoreNames.contains("pdf_store")) {{
                statusDiv.innerText = "ℹ️ কোনো সেভ করা অফলাইন ফাইল নেই।";
                return;
            }}
            const tx = db.transaction("pdf_store", "readonly");
            const store = tx.objectStore("pdf_store");
            const getAllReq = store.getAll();
            getAllReq.onsuccess = function() {{
                const items = getAllReq.result || [];
                const filtered = items.filter(it => selectedCourse === "ALL" || it.course_code === selectedCourse);
                if (!filtered.length) {{
                    statusDiv.innerText = "ℹ️ এই কোর্সে কোনো অফলাইন PDF সেভ করা নেই।";
                    return;
                }}
                statusDiv.innerText = "✅ " + filtered.length + "টি অফলাইন ফাইল প্রস্তুত";
                filtered.forEach(item => {{
                    const card = document.createElement("div");
                    card.style.cssText = "background:#FFFFFF; padding:14px; border-radius:12px; margin-bottom:14px; border:1px solid #CBD5E1;";
                    card.innerHTML = "<h4 style='margin:0; color:#0F172A;'>" + item.course_code + ": " + item.file_name + "</h4><p style='margin:4px 0 10px 0; color:#64748B; font-size:0.82rem;'>Saved: " + item.saved_at + "</p>";
                    container.appendChild(card);
                    renderPdfPagesFast(item.base64, card);
                }});
            }};
        }};
    }}

    function clearAllOfflineData() {{
        if (confirm("সব সেভ করা অফলাইন PDF মুছে ফেলতে চান?")) {{
            const req = indexedDB.deleteDatabase("EduHubOfflineDB");
            req.onsuccess = function() {{ location.reload(); }};
        }}
    }}

    populateDropdown();
    loadOfflinePDFs();
    </script>
    """
    components.html(offline_manager_html, height=740, scrolling=True)
