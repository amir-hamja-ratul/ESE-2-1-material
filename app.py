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

# Page Configuration
st.set_page_config(page_title="EduHub - Academic AI Assistant", page_icon="🎓", layout="wide")

# Advanced Premium UI/UX CSS: Single Line Orange Buttons
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif; 
    }

    .header-box {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
        backdrop-filter: blur(10px);
        padding: 28px 24px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .header-box h2 {
        color: #FFFFFF !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 12px 0;
    }
    .badge {
        background: linear-gradient(90deg, #38BDF8 0%, #3B82F6 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 6px 18px;
        border-radius: 30px;
        display: inline-block;
    }
    
    .course-card {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        padding: 26px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 25px -8px rgba(79, 70, 229, 0.4);
    }
    .course-card h1 {
        color: #FFFFFF !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .metric-card-val {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4F46E5, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .metric-card-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* --- SINGLE LINE ORANGE BUTTON TABS (RADIO) --- */
    div[data-testid="stRadio"] > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        gap: 8px !important;
        background: transparent !important;
        padding-bottom: 10px !important;
        width: 100%;
    }
    
    div[data-testid="stRadio"] input[type="radio"],
    div[data-testid="stRadio"] div[data-baseweb="radio"] {
        display: none !important;
    }

    div[data-testid="stRadio"] label {
        background-color: #F1F5F9 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 8px 14px !important;
        color: #334155 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        white-space: nowrap !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    div[data-testid="stRadio"] label:hover {
        background-color: #E2E8F0 !important;
    }

    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) {
        background: linear-gradient(135deg, #F97316 0%, #EA580C 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #EA580C !important;
        box-shadow: 0 4px 12px rgba(249, 115, 22, 0.35) !important;
    }

    div[data-testid="stRadio"] label:has(input[type="radio"]:checked) p {
        color: #FFFFFF !important;
    }

    .stButton > button, [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3) !important;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Top Header UI
st.markdown("""
    <div class="header-box">
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
        <div style="display: flex; justify-content: center; margin-bottom: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/3429/3429149.png" width="80">
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; margin-top: 0; margin-bottom: 20px;'>Workspace Navigation</h3>", unsafe_allow_html=True)
    
    selected_option = st.selectbox("📌 Select Course Material", course_options)
    selected_code = selected_option.split(" - ")[0]
    selected_title = COURSES[selected_code]
    
    st.divider()
    query_params = st.query_params
    admin_pass = st.text_input("🔒 Admin Secret Key", type="password") if query_params.get("admin") == "true" else ""
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.8rem; margin-top: 20px;'>Designed for ESE-10 Batch.</p>", unsafe_allow_html=True)

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
    st.error("⚠️ GOOGLE_API_KEY is missing! Please add it to Streamlit Secrets.")
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
    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"
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

# --- LOADING SYSTEM (st.status) FOR VECTOR STORE ---
if raw_text.strip():
    with st.status("🔄 Initializing AI Knowledge Base...", expanded=False) as status:
        st.write("📄 Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        
        st.write("🧠 Generating vector embeddings...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        st.write("⚡ Building FAISS vector database...")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        status.update(label="✅ AI Knowledge Base is ready!", state="complete", expanded=False)

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
        st.warning(f"📌 No local PDF files found for **{selected_code}**.")

elif tab_selection == "💬 AI Q&A":
    st.markdown("### 💬 Ask Anything About Your Course")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Type your question here..."):
        if vector_store:
            prompt_with_bilingual = f"{user_query}\n\n[Instruction: Provide the answer clearly in easy English, followed by its clear translation.]"
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("⏳ Searching documents and generating smart response..."):
                    docs = vector_store.similarity_search(user_query)
                    res = ask_gemini(llm, docs, prompt_with_bilingual)
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res})
        else:
            st.error("⚠️ Please upload documents or place files in the course folder first so the AI can search.")

elif tab_selection == "📝 Smart Summary":
    st.markdown("### 📝 Auto-Generated Course Summary")
    if st.button("✨ Generate Smart Summary", key="sum_btn"):
        if vector_store:
            with st.spinner("⏳ Analyzing course materials and summarizing..."):
                docs = vector_store.similarity_search("Summary overview main points")
                summary_res = ask_gemini(llm, docs, "Provide the main content in bullet points in easy English, with translations included.")
                st.markdown(summary_res)
        else:
            st.warning("⚠️ Insufficient document data available.")

elif tab_selection == "🎯 Exam Quiz":
    st.markdown("### 🎯 Exam Preparation Quiz")
    if st.button("📝 Generate Practice Questions", key="quiz_btn"):
        if vector_store:
            with st.spinner("⏳ Creating practice exam questions..."):
                docs = vector_store.similarity_search("Important concepts exam questions")
                quiz_res = ask_gemini(llm, docs, "Create 5 important exam questions with answers in easy English and include translations.")
                st.markdown(quiz_res)
        else:
            st.warning("⚠️ Insufficient document data available.")

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

    selected_course = st.selectbox("📚 Select Course:", courses, key="internal_course_select")

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
        st.info(f"📌 Internal mark sheet for **{selected_course}** will be added soon.")
