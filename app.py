import os
import glob
import io
import sqlite3
import base64
from datetime import date, datetime
import streamlit as st
import streamlit.components.v1 as components

# ==========================================================
# 1. CONFIGURATION & PATH SETUP
# ==========================================================
APP_CONFIG = {
    "title": "EduHub - Academic AI Workspace",
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

# Favicon & Offline Banner
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
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white; text-align: center;
        padding: 10px; font-weight: 700; font-family: 'Plus Jakarta Sans', sans-serif;
        z-index: 999999; box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        font-size: 0.9rem;
    ">
        📡 Offline Mode: ইন্টারনেট কানেকশন বিচ্ছিন্ন! সেভ করা অফলাইন PDF পড়তে পারবেন।
    </div>
    <script>
    function updateOnlineStatus() {
        const banner = document.getElementById("net-status-banner");
        if (banner) banner.style.display = navigator.onLine ? "none" : "block";
    }
    window.addEventListener('online', updateOnlineStatus);
    window.addEventListener('offline', updateOnlineStatus);
    updateOnlineStatus();
    </script>
    """,
    height=0,
)

# ==========================================================
# 2. DATABASE & ACTIVITY TRACKING
# ==========================================================
def get_db_connection():
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
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_student ON activity_log(student_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_activity_course ON activity_log(course_code)"
        )
        conn.commit()
        return conn
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


def track_once_per_session(action: str, course_code: str):
    """একই সেশনে বারবার রিফ্রেশে ডুপ্লিকেট লগ কমাতে সাহায্য করে"""
    sid = st.session_state.get("student_id")
    sname = st.session_state.get("student_name", "")
    if not sid:
        return
    key = f"logged_{sid}_{course_code}_{action}"
    if not st.session_state.get(key):
        log_activity(sid, sname, course_code, action)
        st.session_state[key] = True


# ==========================================================
# 3. AI RAG & PDF HELPERS
# ==========================================================
def parse_ai_response(response) -> str:
    content = response.content if hasattr(response, "content") else response
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif hasattr(block, "text"):
                parts.append(str(block.text))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def ask_gemini(llm, docs, question: str) -> str:
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = f"""Role: Expert Academic Assistant.
Context:
{context}

Question: {question}

Instructions:
1. Answer clearly in **English** first.
2. Then provide an accurate **Bengali translation** right below under "### 🇧🇩 বাংলা অনুবাদ".
"""
    try:
        response = llm.invoke(prompt)
        return parse_ai_response(response)
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"


def display_pdf(file_path: str, dpi: int = 120, pages_per_batch: int = 8):
    try:
        import fitz
        from PIL import Image

        doc = fitz.open(file_path)
        total = len(doc)
        st.info(f"📖 **Total Pages:** {total}")

        if total == 0:
            st.warning("PDF-এ কোনো পেজ পাওয়া যায়নি।")
            return

        start_page = st.number_input(
            "Start Page", min_value=1, max_value=total, value=1, step=1
        )
        end_default = min(total, start_page + pages_per_batch - 1)
        end_page = st.number_input(
            "End Page", min_value=start_page, max_value=total, value=end_default, step=1
        )

        for page_num in range(start_page - 1, end_page):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            st.image(img, caption=f"Page {page_num + 1}", use_container_width=True)
    except Exception as e:
        st.error(f"Error viewing PDF: {e}")


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


@st.cache_data(show_spinner=False)
def extract_text_from_local_pdfs(pdf_paths: tuple, cache_key: tuple):
    from PyPDF2 import PdfReader

    raw, pages = "", 0
    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(pdf_path)
            pages += len(reader.pages)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    raw += text + "\n"
        except Exception:
            pass
    return raw, pages


@st.cache_resource(show_spinner=False)
def build_vector_store(course_code: str, text_hash: str, raw_text: str):
    if not raw_text.strip():
        return None
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(raw_text)
    embeddings = get_embeddings_model()
    return FAISS.from_texts(chunks, embedding=embeddings)


# ==========================================================
# 4. STYLING
# ==========================================================
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap');

        :root {
            --primary: #6366F1;
            --primary-gradient: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
            --text-muted: #64748B;
            --radius-lg: 20px;
            --radius-md: 14px;
            --shadow-soft: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.02);
        }

        html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
        h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; letter-spacing: -0.02em; }

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
        .uni-logo-corner { height: 68px; width: auto; display: block; margin: 0 auto; object-fit: contain; }
        .header-box h2 {
            color: #FFFFFF !important;
            font-size: 1.8rem !important;
            font-weight: 800 !important;
            margin: 0;
        }

        section[data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.55) !important;
            backdrop-filter: blur(20px) saturate(180%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.6) !important;
        }

        .course-card {
            background: var(--primary-gradient);
            padding: 20px 28px;
            border-radius: var(--radius-md);
            color: white;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.35);
        }
        .course-card h1 { color: #FFFFFF !important; font-size: 1.45rem; margin: 0; font-weight: 700; }

        .metric-card {
            background: rgba(255, 255, 255, 0.78);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.85);
            border-radius: var(--radius-md);
            padding: 20px;
            text-align: center;
            box-shadow: var(--shadow-soft);
        }
        .metric-card-val {
            font-family: 'Outfit', sans-serif;
            font-size: 2.1rem;
            font-weight: 800;
            color: #4F46E5;
            line-height: 1.1;
        }
        .metric-card-lbl {
            font-size: 0.78rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .stButton > button {
            background: var(--primary-gradient) !important;
            color: white !important;
            border-radius: 12px !important;
            padding: 10px 24px !important;
            font-weight: 700 !important;
            border: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================================
# 5. HEADER & SIDEBAR
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
_logo_img = (
    f'<img src="data:image/png;base64,{_logo_b64}" class="uni-logo-corner" alt="Logo">'
    if _logo_b64
    else f'<img src="{APP_CONFIG["default_logo_url"]}" class="uni-logo-corner" alt="Logo">'
)

st.markdown(
    f"""
    <div class="header-box">
        <div class="uni-logo-wrapper">{_logo_img}</div>
        <h2>{APP_CONFIG["department"]}</h2>
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
                    b64_logo = base64.b64encode(f.read()).decode()
                st.markdown(
                    f"""
                    <div style="text-align:center; margin-bottom:16px;">
                        <img src="data:image/png;base64,{b64_logo}"
                             style="width:110px; height:110px; border-radius:50%; object-fit:cover; border:3px solid #2e7d32;">
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                break
            except Exception:
                pass

    st.markdown("### Workspace Navigation")
    search_query = st.text_input("🔍 Search Courses", placeholder="e.g. hydrology", key="global_search_input")
    if search_query.strip():
        q = search_query.strip().lower()
        matched = [(c, t) for c, t in COURSES.items() if q in c.lower() or q in t.lower()]
        for c, t in matched[:5]:
            if st.button(f"📘 {c} — {t}", key=f"search_{c}", use_container_width=True):
                st.session_state["course_selectbox"] = f"{c} - {t}"
                st.rerun()
        st.divider()

    selected_option = st.selectbox("📌 Select Course Material", course_options, key="course_selectbox")
    selected_code = selected_option.split(" - ")[0]
    selected_title = COURSES[selected_code]

    st.divider()
    st.caption("YOUR PROFILE")
    student_name_input = st.text_input("Your Name", placeholder="e.g. Amir Hamja Ratul", key="student_name_field")
    student_roll_input = st.text_input("Roll Number", placeholder="e.g. 25103406", key="student_roll_field")

    if student_roll_input.strip():
        st.session_state["student_id"] = student_roll_input.strip()
        st.session_state["student_name"] = student_name_input.strip() or student_roll_input.strip()
        st.success(f"Active: {st.session_state['student_name']}")
    else:
        st.session_state["student_id"] = None

    query_params = st.query_params
    admin_pass = (
        st.text_input("🔒 Admin Key", type="password")
        if query_params.get("admin") == "true"
        else ""
    )

st.markdown(
    f"""
    <div class="course-card">
        <span style="font-size:0.78rem; text-transform:uppercase; opacity:0.85; font-weight:700;">Active Course Material</span>
        <h1>🎓 {selected_code}: {selected_title}</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

folder_code = selected_code.replace(" ", "_")
course_folder = os.path.join(DATA_DIR, folder_code)
os.makedirs(course_folder, exist_ok=True)

if admin_pass == APP_CONFIG["admin_key"]:
    st.success("⚡ Admin Mode Active")
    uploaded_files = st.file_uploader("📥 Upload PDFs", accept_multiple_files=True, type="pdf")
    if uploaded_files:
        for u_file in uploaded_files:
            safe_name = os.path.basename(u_file.name)
            with open(os.path.join(course_folder, safe_name), "wb") as f:
                f.write(u_file.getbuffer())
        st.toast("✅ Files saved successfully!", icon="🎉")
        st.rerun()

api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
if not api_key:
    st.warning("⚠️ `GOOGLE_API_KEY` সেট করা নেই। AI ফিচার ব্যবহার করতে Secrets বা Environment Variable-এ যোগ করুন।")
else:
    os.environ["GOOGLE_API_KEY"] = api_key

local_pdfs = sorted(glob.glob(os.path.join(course_folder, "*.pdf")))
raw_text, total_pages = "", 0
files_count = len(local_pdfs)

if local_pdfs:
    mtimes = tuple(os.path.getmtime(p) for p in local_pdfs)
    raw_text, total_pages = extract_text_from_local_pdfs(tuple(local_pdfs), mtimes)

if files_count > 0:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-card-val">{files_count}</div><div class="metric-card-lbl">📁 Documents Loaded</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-card-val">{total_pages}</div><div class="metric-card-lbl">📄 Total Pages Indexed</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

# ==========================================================
# 6. TABS
# ==========================================================
tab_selection = st.radio(
    "Navigation Tabs",
    ["📖 View & Download", "📲 Offline Saved PDFs"],
    horizontal=True,
    label_visibility="collapsed",
)

if tab_selection == "📖 View & Download":
    st.subheader(f"📖 View & Download - {selected_code}")
    track_once_per_session("View Document", selected_code)

    if local_pdfs:
        selected_pdf = st.selectbox("📄 Select PDF File", local_pdfs, format_func=os.path.basename)
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
                background: linear-gradient(135deg, #10B981 0%, #059669 100%);
                color: white; padding: 11px 20px; border: none; border-radius: 12px;
                font-weight: 700; font-size: 0.9rem; cursor: pointer; width: 100%;
                box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25); font-family: 'Plus Jakarta Sans', sans-serif;
            ">
                💾 Save for Offline Reading
            </button>
            <p id="save-status" style="margin-top:6px; font-weight:600; color:#10B981; text-align:center; font-size:0.85rem;"></p>
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
                        if (statusEl) statusEl.innerText = "✅ Saved to Browser Memory!";
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

elif tab_selection == "📲 Offline Saved PDFs":
    st.subheader("📲 Course-Wise Offline PDF Manager")
    st.caption("🌐 নেট কানেকশন না থাকলেও পূর্বে সেভ করা PDF কোর্স অনুযায়ী বেছে পড়তে পারবেন।")

    courses_js_array = str(list(COURSES.keys()))
    offline_manager_html = f"""
    <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
    <script>
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
    </script>

    <div style="background: rgba(255,255,255,0.85); padding:20px; border-radius:16px; border:1px solid #E2E8F0;">
        <div style="display:flex; gap:12px; align-items:center; margin-bottom:18px; flex-wrap:wrap;">
            <label style="font-weight:700; color:#0F172A;">📂 Select Course:</label>
            <select id="courseFilter" onchange="loadOfflinePDFs()" style="padding:10px 14px; border-radius:10px; border:1px solid #6366F1; font-weight:600;">
                <option value="ALL">-- ALL SAVED COURSES --</option>
            </select>
            <button onclick="loadOfflinePDFs()" style="background:#4F46E5; color:white; padding:10px 16px; border:none; border-radius:10px; font-weight:700; cursor:pointer;">
                🔄 Refresh
            </button>
            <button onclick="clearAllOfflineData()" style="background:#EF4444; color:white; padding:10px 16px; border:none; border-radius:10px; font-weight:700; cursor:pointer; margin-left:auto;">
                🗑️ Clear All
            </button>
        </div>
        <div id="status-msg" style="font-weight:600; margin-bottom:12px; color:#4F46E5;"></div>
        <div id="pdf-display-area"></div>
    </div>

    <script>
    const courseList = {courses_js_array};

    function renderPdfPages(base64Data, container) {{
        try {{
            const binaryStr = atob(base64Data);
            const bytes = new Uint8Array(binaryStr.length);
            for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);

            pdfjsLib.getDocument({{ data: bytes }}).promise.then(function(pdf) {{
                const viewer = document.createElement("div");
                viewer.style.cssText = "max-height:600px; overflow-y:auto; padding:12px; background:#E2E8F0; border-radius:8px;";
                container.appendChild(viewer);

                (async () => {{
                    for (let num = 1; num <= pdf.numPages; num++) {{
                        const page = await pdf.getPage(num);
                        const viewport = page.getViewport({{ scale: 1.15 }});
                        const canvas = document.createElement("canvas");
                        const ctx = canvas.getContext("2d");
                        canvas.height = viewport.height;
                        canvas.width = viewport.width;
                        canvas.style.cssText = "width:100%; max-width:780px; display:block; margin:0 auto 12px auto; border-radius:6px;";
                        viewer.appendChild(canvas);
                        await page.render({{ canvasContext: ctx, viewport: viewport }}).promise;
                    }}
                }})();
            }});
        }} catch (e) {{
            container.innerHTML += "<p style='color:red;'>❌ PDF রেন্ডার করতে সমস্যা হয়েছে।</p>";
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
        statusDiv.innerText = "⏳ Reading offline database...";

        const req = indexedDB.open("EduHubOfflineDB", 2);
        req.onsuccess = function(e) {{
            const db = e.target.result;
            if (!db.objectStoreNames.contains("pdf_store")) {{
                statusDiv.innerText = "ℹ️ কোনো সেভ করা অফলাইন ফাইল পাওয়া যায়নি।";
                return;
            }}
            const tx = db.transaction("pdf_store", "readonly");
            const store = tx.objectStore("pdf_store");
            const getAllReq = store.getAll();
            getAllReq.onsuccess = function() {{
                const items = getAllReq.result || [];
                const filtered = items.filter(it => selectedCourse === "ALL" || it.course_code === selectedCourse);
                if (!filtered.length) {{
                    statusDiv.innerText = "ℹ️ এই ফিল্টারে কোনো অফলাইন ফাইল নেই।";
                    return;
                }}
                statusDiv.innerText = "✅ অফলাইন ফাইলসমূহ লোড হয়েছে!";
                filtered.forEach(item => {{
                    const card = document.createElement("div");
                    card.style.cssText = "background:white; padding:16px; border-radius:12px; margin-bottom:16px; border:1px solid #CBD5E1;";
                    card.innerHTML = "<h4 style='margin:0;'>" + item.course_code + ": " + item.file_name + "</h4><p style='color:#64748B; font-size:0.82rem;'>Saved: " + item.saved_at + "</p>";
                    container.appendChild(card);
                    renderPdfPages(item.base64, card);
                }});
            }};
        }};
    }}

    function clearAllOfflineData() {{
        if (confirm("সব অফলাইন ফাইল মুছে ফেলতে চান?")) {{
            const req = indexedDB.deleteDatabase("EduHubOfflineDB");
            req.onsuccess = function() {{ location.reload(); }};
        }}
    }}

    setTimeout(() => {{ populateDropdown(); loadOfflinePDFs(); }}, 250);
    </script>
    """
    components.html(offline_manager_html, height=780, scrolling=True)
