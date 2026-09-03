import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import glob
import fitz   # PyMuPDF
from PIL import Image
import io
import pandas as pd

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="EduHub - Academic AI Assistant", page_icon="🎓", layout="wide")

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
# MODERN PROFESSIONAL UI/UX CSS (Glassmorphism + Refined System)
# ==========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink-950: #05070F;
        --ink-900: #0B0F1E;
        --ink-800: #131A2E;
        --violet: #6D5DFC;
        --violet-deep: #4C3FD7;
        --cyan: #22D3EE;
        --lime: #A3E635;
        --amber: #FBBF24;
        --surface: #FFFFFF;
        --surface-soft: #F6F7FB;
        --border: #E7E9F3;
        --text-main: #10121C;
        --text-muted: #6B7186;
        --radius-xl: 26px;
        --radius-lg: 20px;
        --radius-md: 14px;
        --radius-sm: 10px;
        --shadow-soft: 0 6px 24px -8px rgba(16, 18, 28, 0.08);
        --shadow-elevated: 0 24px 48px -16px rgba(76, 63, 215, 0.35);
        --shadow-glow: 0 0 0 1px rgba(255,255,255,0.06), 0 30px 60px -20px rgba(109, 93, 252, 0.55);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    h1, h2, h3, .header-box h2, .course-card h1 {
        font-family: 'Space Grotesk', 'Plus Jakarta Sans', sans-serif !important;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(109, 93, 252, 0.07), transparent 40%),
            radial-gradient(circle at 95% 15%, rgba(34, 211, 238, 0.08), transparent 40%),
            #FAFBFF;
    }

    .block-container {
        padding-top: 2.2rem !important;
        max-width: 1180px;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    /* ---------------- HERO HEADER (glass + noise) ---------------- */
    .header-box {
        background:
            radial-gradient(ellipse 70% 100% at 10% 0%, rgba(109, 93, 252, 0.55), transparent 60%),
            radial-gradient(ellipse 60% 90% at 100% 100%, rgba(34, 211, 238, 0.35), transparent 55%),
            linear-gradient(155deg, #05070F 0%, #0B0F1E 45%, #131A2E 100%);
        padding: 46px 32px 40px 32px;
        border-radius: var(--radius-xl);
        text-align: center;
        color: white;
        margin-bottom: 22px;
        box-shadow: var(--shadow-glow);
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    .header-box::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image: radial-gradient(rgba(255,255,255,0.08) 1px, transparent 1px);
        background-size: 22px 22px;
        opacity: 0.35;
        pointer-events: none;
    }
    .header-box h2 {
        color: #FFFFFF !important;
        font-size: 1.95rem;
        font-weight: 700;
        margin: 0 0 16px 0;
        letter-spacing: -0.02em;
        position: relative;
    }
    .badge {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(12px);
        color: #E4E7FF !important;
        font-weight: 600;
        font-size: 0.82rem;
        padding: 9px 24px;
        border-radius: 30px;
        display: inline-block;
        letter-spacing: 0.04em;
        border: 1px solid rgba(255,255,255,0.18);
        position: relative;
    }
    .badge::before {
        content: "●";
        color: var(--lime);
        margin-right: 8px;
        font-size: 0.6rem;
    }

    /* ---------------- UNIVERSITY LOGO (top-right corner of header) ---------------- */
    .uni-logo-corner {
        position: absolute;
        top: 18px;
        right: 22px;
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: rgba(255,255,255,0.92);
        padding: 6px;
        box-shadow: 0 8px 20px -4px rgba(0,0,0,0.35), 0 0 0 1px rgba(255,255,255,0.15);
        z-index: 2;
        object-fit: contain;
    }
    @media (max-width: 768px) {
        .uni-logo-corner {
            width: 46px;
            height: 46px;
            top: 12px;
            right: 12px;
        }
    }

    /* ---------------- COURSE TITLE CARD ---------------- */
    .course-card {
        background: linear-gradient(120deg, #6D5DFC 0%, #5847E8 55%, #4C3FD7 100%);
        padding: 28px 34px;
        border-radius: var(--radius-lg);
        color: white;
        margin-bottom: 28px;
        box-shadow: var(--shadow-elevated);
        border: 1px solid rgba(255,255,255,0.14);
        position: relative;
        overflow: hidden;
    }
    .course-card::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -8%;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(255,255,255,0.18), transparent 70%);
        border-radius: 50%;
    }
    .course-card h1 {
        color: #FFFFFF !important;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        position: relative;
    }

    /* ---------------- METRIC CARDS ---------------- */
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 28px 20px;
        text-align: center;
        box-shadow: var(--shadow-soft);
        transition: transform 0.25s cubic-bezier(.2,.8,.2,1), box-shadow 0.25s ease, border-color 0.25s ease;
        position: relative;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 18px 36px -12px rgba(109, 93, 252, 0.3);
        border-color: rgba(109, 93, 252, 0.35);
    }
    .metric-card-val {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.7rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6D5DFC, #22D3EE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        line-height: 1;
    }
    .metric-card-lbl {
        font-size: 0.76rem;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.4px;
    }

    /* ---------------- SIDEBAR ---------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(190deg, #05070F 0%, #0B0F1E 60%, #131A2E 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * {
        color: #C9CDE0 !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #8B90A8 !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] input {
        background-color: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important;
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] div[data-baseweb="select"]:hover > div {
        border-color: rgba(109, 93, 252, 0.5) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.08) !important;
        margin: 20px 0 !important;
    }

    /* ---------------- SINGLE LINE PILL TABS ---------------- */
    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        gap: 10px !important;
        background: transparent !important;
        padding: 6px 4px 16px 4px !important;
        width: 100%;
    }

    div[data-testid="stRadio"] input[type="radio"],
    div[data-testid="stRadio"] div[data-baseweb="radio"] {
        display: none !important;
    }

    div[data-testid="stRadio"] label {
        background-color: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 30px !important;
        padding: 11px 20px !important;
        color: #454A5E !important;
        font-weight: 600 !important;
        font-size: 0.87rem !important;
        white-space: nowrap !important;
        box-shadow: 0 2px 8px rgba(16, 18, 28, 0.04) !important;
        cursor: pointer !important;
        transition: all 0.22s cubic-bezier(.2,.8,.2,1) !important;
    }

    div[data-testid="stRadio"] label:hover {
        border-color: rgba(109, 93, 252, 0.45) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px -6px rgba(109, 93, 252, 0.25) !important;
    }

    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) {
        background: linear-gradient(135deg, #6D5DFC 0%, #4C3FD7 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #4C3FD7 !important;
        box-shadow: 0 10px 22px -6px rgba(109, 93, 252, 0.6) !important;
    }

    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* ---------------- BUTTONS ---------------- */
    .stButton > button, [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #6D5DFC 0%, #4C3FD7 100%) !important;
        color: white !important;
        border-radius: 14px !important;
        padding: 13px 26px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        border: none !important;
        box-shadow: 0 10px 22px -8px rgba(109, 93, 252, 0.5) !important;
        width: 100%;
        transition: transform 0.2s cubic-bezier(.2,.8,.2,1), box-shadow 0.2s ease, filter 0.2s ease !important;
        letter-spacing: 0.01em;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 16px 30px -8px rgba(109, 93, 252, 0.65) !important;
        filter: brightness(1.06);
    }
    .stButton > button:active, [data-testid="stDownloadButton"] > button:active {
        transform: translateY(0px);
    }

    /* ---------------- SECTION HEADINGS ---------------- */
    h3 {
        font-weight: 700 !important;
        color: var(--text-main) !important;
        letter-spacing: -0.015em;
        margin-bottom: 6px !important;
    }
    h3::after {
        content: "";
        display: block;
        width: 42px;
        height: 4px;
        margin-top: 10px;
        border-radius: 4px;
        background: linear-gradient(90deg, #6D5DFC, #22D3EE);
    }

    /* ---------------- INPUTS (selectbox / text input / file uploader) ---------------- */
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border: 1px solid var(--border) !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(109, 93, 252, 0.5) !important;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border-radius: var(--radius-md) !important;
        border: 1.5px dashed rgba(109, 93, 252, 0.35) !important;
        background: linear-gradient(180deg, rgba(109,93,252,0.03), rgba(34,211,238,0.03)) !important;
    }

    /* ---------------- CHAT ---------------- */
    div[data-testid="stChatMessage"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-soft);
        padding: 6px 10px;
        margin-bottom: 4px;
    }
    div[data-testid="stChatInput"] {
        border-radius: var(--radius-md) !important;
    }
    div[data-testid="stChatInput"] textarea {
        border-radius: var(--radius-md) !important;
    }

    /* ---------------- DATAFRAME ---------------- */
    div[data-testid="stDataFrame"] {
        border-radius: var(--radius-md);
        overflow: hidden;
        border: 1px solid var(--border);
        box-shadow: var(--shadow-soft);
    }

    /* ---------------- ALERTS ---------------- */
    div[data-testid="stAlert"] {
        border-radius: var(--radius-sm) !important;
        border: 1px solid var(--border) !important;
    }

    /* ---------------- SCROLLBAR ---------------- */
    ::-webkit-scrollbar { height: 8px; width: 8px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #6D5DFC, #4C3FD7);
        border-radius: 10px;
    }

    /* ---------------- DIVIDER ---------------- */
    hr {
        border-color: var(--border) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# TOP HEADER UI
# ==========================================================
import base64

def _load_logo_b64():
    logo_path = os.path.join(os.path.dirname(__file__), "assets", "university_logo.png")
    try:
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

_logo_b64 = _load_logo_b64()
_logo_html = (
    f'<img src="data:image/png;base64,{_logo_b64}" class="uni-logo-corner">'
    if _logo_b64 else ""
)

st.markdown(f"""
    <div class="header-box">
        {_logo_html}
        <h2>🌱 Department of Environmental Science and Engineering</h2>
        <span class="badge">📚 2nd Year 1st Semester</span>
    </div>
""", unsafe_allow_html=True)

COURSES = {
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
    st.markdown("""
        <div style="display: flex; justify-content: center; margin-bottom: 14px; margin-top: 6px;">
            <div style="width: 84px; height: 84px; border-radius: 22px; background: linear-gradient(135deg, #6D5DFC, #22D3EE); display: flex; align-items: center; justify-content: center; box-shadow: 0 12px 26px -6px rgba(109,93,252,0.55); border: 1px solid rgba(255,255,255,0.15); font-size: 2.4rem; line-height: 1;">
                🎓
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: 0; margin-bottom: 22px;'>Workspace Navigation</h3>", unsafe_allow_html=True)

    selected_option = st.selectbox("📌 Select Course Material", course_options)
    selected_code = selected_option.split(" - ")[0]
    selected_title = COURSES[selected_code]

    st.divider()
    query_params = st.query_params
    admin_pass = st.text_input("🔒 Admin Secret Key", type="password") if query_params.get("admin") == "true" else ""
    st.markdown("<p style='text-align: center; color: #94A3B8; font-size: 0.78rem; margin-top: 26px;'>Designed for ESE-10 Batch.</p>", unsafe_allow_html=True)

st.markdown(f"""
    <div class="course-card">
        <h1>🎓 {selected_code}: {selected_title}</h1>
    </div>
""", unsafe_allow_html=True)

if admin_pass == "285277":
    st.success("⚡ Admin Mode Enabled: Ready to upload new materials.")
    uploaded_files = st.file_uploader("📥 Upload Course Materials (PDF format)", accept_multiple_files=True, type="pdf")
else:
    uploaded_files = None

api_key = st.secrets.get("GOOGLE_API_KEY", None)
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY পাওয়া যায়নি! Streamlit Secrets-এ যোগ করুন।")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

folder_code = selected_code.replace(" ", "_")
course_folder = os.path.join("data", folder_code)
local_pdfs = glob.glob(f"{course_folder}/*.pdf")

raw_text = ""
total_pages = 0
files_count = 0

if uploaded_files:
    files_count = len(uploaded_files)
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        total_pages += len(pdf_reader.pages)
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""
elif local_pdfs:
    files_count = len(local_pdfs)
    for pdf_path in local_pdfs:
        pdf_reader = PdfReader(pdf_path)
        total_pages += len(pdf_reader.pages)
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""

if raw_text.strip():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-val">{files_count}</div>
                <div class="metric-card-lbl">📂 Loaded Documents</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-val">{total_pages}</div>
                <div class="metric-card-lbl">📄 Total Processed Pages</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def ask_gemini(llm, docs, question):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"নিচের তথ্যগুলোর ওপর ভিত্তি করে প্রশ্নের উত্তর দাও:\n\n{context}\n\nপ্রশ্ন: {question}"
    response = llm.invoke(prompt)
    if hasattr(response, 'content'):
        if isinstance(response.content, str):
            return response.content
        elif isinstance(response.content, list):
            return "".join([item.get('text', '') if isinstance(item, dict) else str(item) for item in response.content])
    return str(response)


def display_pdf(file_path):
    doc = fitz.open(file_path)
    st.info(f"📖 **Displaying Total Pages:** {len(doc)}")
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        st.image(image, caption=f"Page {page_num + 1}", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SINGLE LINE HORIZONTAL BUTTON TABS ---
tab_selection = st.radio(
    "Navigation Tabs",
    [
        "📖 View & Download", "💬 AI Q&A", "📝 Smart Summary",
        "🎯 Exam Quiz", "📊 Leaderboard"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.3)
vector_store = None

if raw_text.strip():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)

if tab_selection == "📖 View & Download":
    st.markdown("### 📄 Course Documents Viewer")
    if local_pdfs:
        selected_pdf = st.selectbox("Choose a file to view or download:", local_pdfs, format_func=lambda x: os.path.basename(x))
        with open(selected_pdf, "rb") as f:
            st.download_button(
                label="📥 Download File",
                data=f,
                file_name=os.path.basename(selected_pdf),
                mime="application/pdf"
            )
        st.markdown("---")
        display_pdf(selected_pdf)
    else:
        st.warning(f"📌 **{selected_code}** কোর্সের জন্য বর্তমানে কোনো স্থানীয় PDF ফাইল পাওয়া যায়নি।")

elif tab_selection == "💬 AI Q&A":
    st.markdown("### 💬 Ask Anything About Your Course")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Type your question here..."):
        if vector_store:
            prompt_with_bilingual = f"{user_query}\n\n[অর্ডার: উত্তরটি প্রথমে সহজ ইংরেজিতে (Easy English) দেবে এবং সাথে সাথেই তার বাংলা অনুবাদ (Bangla Translation) নিচে যুক্ত করবে।]"
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Generating smart response..."):
                    docs = vector_store.similarity_search(user_query)
                    res = ask_gemini(llm, docs, prompt_with_bilingual)
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res})
        else:
            st.error("⚠️ আগে ডকুমেন্ট আপলোড করুন বা ফোল্ডারে ফাইল রাখুন যাতে AI সার্চ করতে পারে।")

elif tab_selection == "📝 Smart Summary":
    st.markdown("### 📝 Auto-Generated Course Summary")
    if st.button("✨ Generate Smart Summary", key="sum_btn"):
        if vector_store:
            with st.spinner("Analyzing and summarizing..."):
                docs = vector_store.similarity_search("Summary overview main points")
                summary_res = ask_gemini(llm, docs, "মূল বিষয়বস্তু পয়েন্ট আকারে সহজ ইংরেজিতে (Easy English) লেখো এবং প্রতিটি পয়েন্টের নিচে বাংলা অনুবাদ (Bangla Translation) সাজিয়ে দাও।")
                st.markdown(summary_res)
        else:
            st.warning("⚠️ পর্যাপ্ত ডকুমেন্ট ডেটা নেই।")

elif tab_selection == "🎯 Exam Quiz":
    st.markdown("### 🎯 Exam Preparation Quiz")
    if st.button("📝 Generate Practice Questions", key="quiz_btn"):
        if vector_store:
            with st.spinner("Creating exam questions..."):
                docs = vector_store.similarity_search("Important concepts exam questions")
                quiz_res = ask_gemini(llm, docs, "পরীক্ষার জন্য ৫টি গুরুত্বপূর্ণ প্রশ্ন ও উত্তর সহজ ইংরেজিতে (Easy English) তৈরি করো এবং বাংলা অনুবাদ যুক্ত করো।")
                st.markdown(quiz_res)
        else:
            st.warning("⚠️ পর্যাপ্ত ডকুমেন্ট ডেটা নেই।")

elif tab_selection == "📊 Leaderboard":
    st.markdown("### 📊 Department of Environmental Science and Engineering")
    st.markdown("#### Jatiya Kabi Kazi Nazrul Islam University")
    st.markdown("**Marks of Internal Evaluation (Session: 2024-2025)**")

    courses = [
        "ESE 2101: Hydrology and Hydrogeology",
        "ESE 2103: Oceanography and Limnology",
        "ESE 2105: Ecology",
        "ESE 2102: Ecology - Lab",
        "ESE 2107: Environmental Microbiology",
        "ESE 2104: Environmental Microbiology - Lab",
        "ESE 2109: Survey and Settlement",
        "ESE 2106: Survey and Settlement - Lab",
        "ESE 2111: Soil Mechanics",
        "ESE 2108: Engineering Drawing Lab",
        "ESE 2113: Statistics for Environment",
        "PYQ: Previous Year Questions",
        "MEQ: Mid Exam Questions"
    ]

    selected_course = st.selectbox("📚 কোর্স সিলেক্ট করুন:", courses, key="internal_course_select")

    if selected_course.startswith("ESE 2101"):
        data = {
            "Roll": [
                "25103402", "25103405", "25103406", "25103409", "25103413", "25103413",
                "25103414", "25103415", "25103416", "25103417", "25103420", "25103421",
                "25103422", "25103423", "25103427", "25103429", "25103430", "25103431",
                "25103433", "25103434", "25103435", "25103436", "25103437", "25103438",
                "25103440", "24103403", "24103423"
            ],
            "Name of Students": [
                "FARJANA AKTER MITU", "MOHSINA KHAN", "NOSHIN", "AMIR HAMZA RATUL", "NAZIFA SULTANA", "ELMA",
                "MD. SAIDUR RAHMAN SAID", "MST. FARHANA ISLAM BORSHA", "MD. KAWSER MAHMUD", "SADIA AFRIN PROMI", "JUNAID HASSAN PROVAT", "SIRAZUM MONIRA",
                "SHAD EVENY AHMED SHOWRAV", "MD. MAHADI HASAN", "MURSALIN AL IFTI", "RADUYAN HOSEN", "SANIA AKTER", "HRIDOY MIA",
                "MD. ABU SAIM", "MD. YOUSUF ALI", "MUTAHARA SALSABIL LABIBA", "MAHDI HASAN MARUF", "MST. RATNA AKTER", "MST. KHADILA AKTER",
                "BORSHA AKTER", "UMME SALMA SADIA", "FARIHA TASNUBA"
            ],
            "Attendance (10)": [10, 7, 9, 9, 10, 9, 10, 10, 10, 10, 9, 9, 9, 9, 8, 10, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            "Mid-1 (10)": [10, 8, 10, 10, 10, 9, 9, 10, 8, 8, 10, 10, 9, 10, 10, 10, 10, 8, 10, 10, 10, 9, 9, 10, 10, 10, 10],
            "Mid-2 (10)": [8, 8, 9, 9, 9, 9, 10, 10, 9, 9, 9, 10, 10, 9, 7, 9, 7, 10, 10, 10, 10, 8, 10, 9, 9, 7, 10],
            "Mid-3 (10)": [9, 9, 6, 6, 7, 6, 8, 6, 6, 7, 6, 7, 7, 8, 5, 9, 10, 8, 6, 8, 10, 6, 8, 9, 10, 10, 7],
            "Total Marks (40)": [37, 32, 34, 34, 36, 33, 37, 36, 33, 34, 34, 36, 35, 36, 30, 38, 36, 35, 36, 38, 40, 33, 37, 38, 39, 37, 37]
        }
        df_internal = pd.DataFrame(data)
        df_internal = df_internal.sort_values(by="Total Marks (40)", ascending=False).reset_index(drop=True)
        df_internal.insert(0, "Rank", [f"#{i}" for i in range(1, len(df_internal) + 1)])
        st.dataframe(df_internal, use_container_width=True, hide_index=True)
    else:
        st.info(f"📌 **{selected_course}** কোর্সের ইন্টারনাল মার্কশিট শিঘ্রই যুক্ত করা হবে।")
