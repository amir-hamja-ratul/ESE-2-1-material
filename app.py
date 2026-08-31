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

# Fixed Contrast & Visible Text CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Plus Jakarta Sans', sans-serif; 
    }
    
    /* Force Light Background */
    .stApp { 
        background-color: #F8FAFC !important; 
    }
    
    /* Force Visible Text Colors Across App */
    h1, h2, h3, h4, h5, h6, p, label, span, div, .stMarkdown {
        color: #0F172A !important;
    }
    
    /* Sidebar Text & Background Fix */
    [data-testid="stSidebar"] { 
        background-color: #FFFFFF !important; 
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }
    
    /* Header Styling Fix */
    .header-container {
        text-align: center;
        padding: 10px 0 20px 0;
    }
    .dept-title {
        color: #0F172A !important;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .sem-badge {
        display: inline-block;
        background-color: #EEF2FF !important;
        color: #4F46E5 !important;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 4px 16px;
        border-radius: 20px;
        border: 1px solid #C7D2FE !important;
    }
    
    /* Hero Banner Fix */
    .hero-card {
        background: linear-gradient(135deg, #1E1B4B 0%, #4338CA 100%) !important;
        padding: 28px 32px;
        border-radius: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 15px -3px rgba(67, 56, 202, 0.2);
    }
    .hero-card h1 { 
        color: #FFFFFF !important; 
        margin: 0; 
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    /* Metric Cards Text Fix */
    .metric-card {
        background: #FFFFFF !important;
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #4F46E5 !important;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B !important;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* Buttons Fix */
    .stButton>button, [data-testid="stDownloadButton"]>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    /* Tabs Text Fix */
    .stTabs [data-baseweb="tab"] {
        color: #475569 !important;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #4F46E5 !important;
    }
    </style>
""", unsafe_allow_html=True)
# Top Department Header
st.markdown("""
    <div class="header-container">
        <div class="dept-title">🌱 Department of Environmental Science and Engineering</div>
        <div class="sem-badge">📚 2nd Year 1st Semester</div>
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
    st.image("https://cdn-icons-png.flaticon.com/512/3429/3429149.png", width=60)
    st.title("Workspace Navigation")
    
    selected_option = st.selectbox("📌 Select Course", course_options)
    selected_code = selected_option.split(" - ")[0]
    selected_title = COURSES[selected_code]
    
    st.divider()
    
    query_params = st.query_params
    admin_pass = st.text_input("🔒 Secret Key", type="password") if query_params.get("admin") == "true" else ""

st.markdown(f"""
    <div class="hero-card">
        <h1>🎓 {selected_code}: {selected_title}</h1>
    </div>
""", unsafe_allow_html=True)

if admin_pass == "285277":
    st.success("⚡ Admin Mode Enabled: Document Upload Access Granted")
    uploaded_files = st.file_uploader("📥 Upload Course Materials (PDF)", accept_multiple_files=True, type="pdf")
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

# Helper function to display PDF pages as images
def display_pdf(file_path):
    doc = fitz.open(file_path)
    st.info(f"📖 **Total Pages:** {len(doc)}")
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        
        st.image(image, caption=f"Page {page_num + 1}", use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

api_key = st.secrets.get("GOOGLE_API_KEY", None)
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY পাওয়া যায়নি! Streamlit Secrets-এ যোগ করুন।")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# Dynamically load PDFs from data/COURSE_CODE/ folder
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
                <div class="metric-value">{files_count}</div>
                <div class="metric-label">📂 Loaded Files</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_pages}</div>
                <div class="metric-label">📄 Processed Pages</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Processing PDF contents..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    # Clean Tabs Layout
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["📖 View & Download PDF", "💬 Interactive Q&A", "📝 Smart Summary", "🎯 Exam Quiz", "🃏 Flashcards", "📐 Formulas & Terms"])
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.3)

    with tab0:
        st.subheader("📄 Course Documents Viewer")
        if local_pdfs:
            selected_pdf = st.selectbox("Select PDF to view:", local_pdfs, format_func=lambda x: os.path.basename(x))
            
            with open(selected_pdf, "rb") as f:
                st.download_button(
                    label=f"📥 Download {os.path.basename(selected_pdf)}",
                    data=f,
                    file_name=os.path.basename(selected_pdf),
                    mime="application/pdf"
                )
            
            st.markdown("---")
            display_pdf(selected_pdf)

    with tab1:
        st.subheader("Ask Anything About Course Materials")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Enter your question here..."):
            prompt_with_bilingual = f"{user_query}\n\n[অর্ডার: উত্তরটি প্রথমে সহজ ইংরেজিতে (Easy English) দেবে এবং সাথে সাথেই তার বাংলা অনুবাদ (Bangla Translation) নিচে যুক্ত করবে।]"
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing document..."):
                    docs = vector_store.similarity_search(user_query)
                    res = ask_gemini(llm, docs, prompt_with_bilingual)
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res})

    with tab2:
        if st.button("Generate Smart Summary"):
            with st.spinner("Processing Summary..."):
                docs = vector_store.similarity_search("Summary overview")
                summary_res = ask_gemini(llm, docs, "মূল বিষয়বস্তু পয়েন্ট আকারে সহজ ইংরেজিতে (Easy English) লেখো এবং প্রতিটি পয়েন্টের নিচে বাংলা অনুবাদ (Bangla Translation) সাজিয়ে দাও।")
                st.markdown(summary_res)

    with tab3:
        if st.button("Generate Exam Questions"):
            with st.spinner("Generating Quiz..."):
                docs = vector_store.similarity_search("Important concepts")
                quiz_res = ask_gemini(llm, docs, "পরীক্ষার জন্য ৫টি গুরুত্বপূর্ণ প্রশ্ন ও উত্তর সহজ ইংরেজিতে (Easy English) তৈরি করো এবং বাংলা অনুবাদ যুক্ত করো।")
                st.markdown(quiz_res)

    with tab4:
        if st.button("Generate Study Flashcards"):
            with st.spinner("Generating Flashcards..."):
                docs = vector_store.similarity_search("Key concepts definitions terms")
                flash_res = ask_gemini(llm, docs, "১০টি গুরুত্বপূর্ণ Flashcard সহজ ইংরেজিতে (Easy English) বানাও এবং বাংলা ব্যাখ্যা যুক্ত করো।")
                st.markdown(flash_res)

    with tab5:
        if st.button("Extract Definitions & Formulas"):
            with st.spinner("Extracting Key Terms..."):
                docs = vector_store.similarity_search("Definitions equations formulas")
                formula_res = ask_gemini(llm, docs, "গুরুত্বপূর্ণ সংজ্ঞা ও গাণিতিক সূত্রগুলো সহজ ইংরেজিতে লেখো এবং বাংলা অর্থ যুক্ত করো।")
                st.markdown(formula_res)
else:
    st.info(f"📌 {selected_code} কোর্সের জন্য বর্তমানে কোনো PDF লোড করা নেই।")
