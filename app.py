import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import glob
import fitz  # PyMuPDF
from PIL import Image
import io

# Page Config
st.set_page_config(page_title="EduHub - Academic AI Assistant", page_icon="🎓", layout="wide")

# Advanced Premium UI/UX CSS
st.markdown("""
<style>
/* ট্যাব লিস্টের মূল ব্যাকগ্রাউন্ড এবং গ্লাস ইফেক্ট জোরদার করা */
div[data-testid="stHorizontalBlock"] div[data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.08) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 16px !important;
    padding: 8px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2) !important;
}

/* প্রতিটি ট্যাবের টেক্সট ও প্যাডিং সুন্দর করা */
div[data-baseweb="tab"] {
    border-radius: 10px !important;
    color: inherit !important;
    transition: all 0.3s ease !important;
}

/* সিলেক্ট করা বা অ্যাক্টিভ ট্যাবের স্টাইল */
div[data-baseweb="tab"][aria-selected="true"] {
    background: rgba(255, 255, 255, 0.2) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
}
</style>
""", unsafe_allow_html=True)
</style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Outfit', sans-serif; 
    }
    
    /* Animations */
    @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes scaleIn {
        0% { opacity: 0; transform: scale(0.95); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Department Banner Header (Glassmorphism inspired) */
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
        animation: fadeInUp 0.6s ease-out;
    }
    .header-box h2 {
        color: #FFFFFF !important;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
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
        box-shadow: 0 4px 15px rgba(56, 189, 248, 0.3);
    }
    
    /* Hero Title Card */
    .course-card {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%);
        padding: 26px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 28px;
        box-shadow: 0 12px 25px -8px rgba(79, 70, 229, 0.4);
        animation: scaleIn 0.5s ease-out 0.2s both;
    }
    .course-card h1 {
        color: #FFFFFF !important;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }

    /* 3D Interactive Metric Cards */
    .metric-container {
        display: flex;
        gap: 20px;
        margin-bottom: 30px;
    }
    .metric-card {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInUp 0.5s ease-out 0.4s both;
    }
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1), 0 10px 10px -5px rgba(0,0,0,0.04);
        border-color: #818CF8;
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

    /* Pill-Style Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        padding: 4px 0 16px 0;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F1F5F9;
        border-radius: 30px !important;
        padding: 10px 24px !important;
        border: 1px solid transparent !important;
        color: #475569 !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 15px -3px rgba(79, 70, 229, 0.4) !important;
    }
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: #E2E8F0;
        transform: translateY(-2px);
    }

    /* Premium Button Styling */
    .stButton > button, [data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 10px rgba(16, 185, 129, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px) scale(1.02) !important;
        box-shadow: 0 8px 15px rgba(16, 185, 129, 0.4) !important;
    }

    /* Custom style for Sidebar Collapse/Expand Menu Button */
    [data-testid="collapsedControl"] {
        background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3) !important;
        margin: 10px 0 0 10px !important;
        padding: 4px !important;
    }
    
    [data-testid="collapsedControl"] svg {
        fill: #FFFFFF !important;
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
    # Centered Logo using HTML/CSS
    st.markdown("""
        <div style="display: flex; justify-content: center; margin-bottom: 10px;">
            <img src="https://cdn-icons-png.flaticon.com/512/3429/3429149.png" width="80">
        </div>
    """, unsafe_allow_html=True)
    
    # Centered Title
    st.markdown("<h3 style='text-align: center; margin-top: 0; margin-bottom: 20px;'>Workspace Navigation</h3>", unsafe_allow_html=True)
    
    selected_option = st.selectbox("📌 Select Course Material", course_options)
    selected_code = selected_option.split(" - ")[0]
    selected_title = COURSES[selected_code]
    
    st.divider()
    
    query_params = st.query_params
    admin_pass = st.text_input("🔒 Admin Secret Key", type="password") if query_params.get("admin") == "true" else ""
    
    # Centered Footer Caption
    st.markdown("<p style='text-align: center; color: #64748B; font-size: 0.8rem; margin-top: 20px;'>Designed for ESE-10 Batch.</p>", unsafe_allow_html=True)

# Course Title Banner
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

    with st.spinner("🤖 AI is reading your documents..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    tab0, tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs(["📖 View & Download", "💬 AI Q&A", "📝 Smart Summary", "🎯 Exam Quiz", "🃏 Flashcards", "📐 Formulas","Leaderboard"])
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.3)

    with tab0:
        st.markdown("### 📄 Course Documents Viewer")
        if local_pdfs:
            selected_pdf = st.selectbox("Choose a file to view or download:", local_pdfs, format_func=lambda x: os.path.basename(x))
            
            with open(selected_pdf, "rb") as f:
                st.download_button(
                    label=f"📥 Download File",
                    data=f,
                    file_name=os.path.basename(selected_pdf),
                    mime="application/pdf"
                )
            
            st.markdown("---")
            display_pdf(selected_pdf)

    with tab1:
        st.markdown("### 💬 Ask Anything About Your Course")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Type your question here..."):
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

    with tab2:
        st.markdown("### 📝 Auto-Generated Course Summary")
        if st.button("✨ Generate Smart Summary"):
            with st.spinner("Analyzing and summarizing..."):
                docs = vector_store.similarity_search("Summary overview main points")
                summary_res = ask_gemini(llm, docs, "মূল বিষয়বস্তু পয়েন্ট আকারে সহজ ইংরেজিতে (Easy English) লেখো এবং প্রতিটি পয়েন্টের নিচে বাংলা অনুবাদ (Bangla Translation) সাজিয়ে দাও।")
                st.markdown(summary_res)

    with tab3:
        st.markdown("### 🎯 Exam Preparation Quiz")
        if st.button("📝 Generate Practice Questions"):
            with st.spinner("Creating exam questions..."):
                docs = vector_store.similarity_search("Important concepts exam questions")
                quiz_res = ask_gemini(llm, docs, "পরীক্ষার জন্য ৫টি গুরুত্বপূর্ণ প্রশ্ন ও উত্তর সহজ ইংরেজিতে (Easy English) তৈরি করো এবং বাংলা অনুবাদ যুক্ত করো।")
                st.markdown(quiz_res)

    with tab4:
        st.markdown("### 🃏 Quick Revision Flashcards")
        if st.button("⚡ Generate Study Flashcards"):
            with st.spinner("Crafting flashcards..."):
                docs = vector_store.similarity_search("Key concepts definitions terms")
                flash_res = ask_gemini(llm, docs, "১০টি গুরুত্বপূর্ণ Flashcard সহজ ইংরেজিতে (Easy English) বানাও এবং বাংলা ব্যাখ্যা যুক্ত করো।")
                st.markdown(flash_res)

    with tab5:
        st.markdown("### 📐 Key Formulas & Definitions")
        if st.button("🔍 Extract Important Terms"):
            with st.spinner("Scanning for formulas and definitions..."):
                docs = vector_store.similarity_search("Definitions equations formulas key terms")
                formula_res = ask_gemini(llm, docs, "গুরুত্বপূর্ণ সংজ্ঞা ও গাণিতিক সূত্রগুলো সহজ ইংরেজিতে লেখো এবং বাংলা অর্থ যুক্ত করো।")
                st.markdown(formula_res)
else:
    st.warning(f"📌 **{selected_code}** কোর্সের জন্য বর্তমানে কোনো ডকুমেন্ট লোড করা নেই।")

with tab6:
    import pandas as pd

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
        
        # Sort by Total Marks (Highest to Lowest) and add Rank column
        df_internal = df_internal.sort_values(by="Total Marks (40)", ascending=False).reset_index(drop=True)
        df_internal.insert(0, "Rank", [f"#{i}" for i in range(1, len(df_internal) + 1)])
        
        st.dataframe(df_internal, use_container_width=True, hide_index=True)
    else:
        st.info(f"📌 **{selected_course}** কোর্সের ইন্টারনাল মার্কশিট শিঘ্রই যুক্ত করা হবে।")
