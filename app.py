import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Page Config
st.set_page_config(page_title="EduHub - Academic AI Assistant", page_icon="🎓", layout="wide")

# Modern Light Theme CSS & Sidebar Toggle Arrow Fix
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp { background-color: #F8F9FA; color: #1E293B !important; }
    [data-testid="stSidebar"] { background-color: #E9ECEF; }
    
    /* Text Color Fixes */
    html, body, p, label, span, h1, h2, h3, h4, .stMarkdown { color: #1E293B !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #1E293B !important; }
    
    /* Header & Sidebar Collapse Arrow Color Fix */
    [data-testid="stHeader"] {
        background-color: #F8F9FA !important;
    }
    [data-testid="stHeader"] button svg,
    [data-testid="stSidebarCollapseButton"] button svg {
        fill: #1E293B !important;
        color: #1E293B !important;
    }
    
    /* File Uploader Style */
    [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color: #FFFFFF !important; }
    [data-testid="stFileUploader"] button, [data-testid="stFileUploader"] button * { color: #FFFFFF !important; fill: #FFFFFF !important; }
    
    /* Hero Banner Styling */
    .hero-card {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        padding: 24px;
        border-radius: 12px;
        color: white !important;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .hero-card h1 { color: white !important; margin: 0; }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background-color: #4F46E5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Top Department Header
st.markdown("<h2 style='text-align: center; color: #1E293B; font-size: 1.8rem; font-weight: 700; margin-bottom: 24px;'>🌱 Department of Environmental Science and Engineering</h2>", unsafe_allow_html=True)

# ESE Department Courses & Resources List
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
    "MEQ": "Mid Exam Questions",
    "GIS": "GIS & Remote Sensing Resources",
    "EIA": "EIA & Environmental Law",
    "LAB": "Lab Manuals & Protocols",
    "FIELD": "Field Work Reports & Data",
    "STD": "Environmental Standards & Guidelines"
}

course_options = [f"{code} - {title}" for code, title in COURSES.items()]

# Sidebar Workspace
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3429/3429149.png", width=65)
    st.title("Workspace Navigation")
    
    selected_option = st.selectbox("📌 Select Course", course_options)
    selected_code = selected_option.split(" - ")[0]
    selected_title = COURSES[selected_code]
    
    st.divider()
    
    # Hidden Admin Panel via URL Parameter
    query_params = st.query_params
    if query_params.get("admin") == "true":
        admin_pass = st.text_input("🔒 Secret Key", type="password")
    else:
        admin_pass = ""

# Top Hero Header
st.markdown(f"""
    <div class="hero-card">
        <h1>🎓 {selected_code}: {selected_title}</h1>
    </div>
""", unsafe_allow_html=True)

# Admin Mode vs Student View
if admin_pass == "285277":
    st.success("⚡ Admin Mode Enabled: Document Upload Access Granted")
    uploaded_files = st.file_uploader("📥 Upload Course Materials (PDF)", accept_multiple_files=True, type="pdf")
else:
    st.info("ℹ️ Student View Mode: Access Pre-loaded Workspace Content")
    uploaded_files = None

def ask_gemini(llm, docs, question):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"নিচের তথ্যগুলোর ওপর ভিত্তি করে প্রশ্নের উত্তর দাও:\n\n{context}\n\nপ্রশ্ন: {question}"
    response = llm.invoke(prompt)
    return response.content

if "messages" not in st.session_state:
    st.session_state.messages = []

# Process uploaded files
if uploaded_files:
    api_key = st.secrets.get("GOOGLE_API_KEY", None)
    if not api_key:
        st.error("⚠️ GOOGLE_API_KEY পাওয়া যায়নি! দয়া করে Streamlit Secrets-এ API Key যোগ করুন।")
        st.stop()
        
    os.environ["GOOGLE_API_KEY"] = api_key

    raw_text = ""
    total_pages = 0
    
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        total_pages += len(pdf_reader.pages)
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""
            
    col1, col2 = st.columns(2)
    col1.metric("📂 Loaded Files", len(uploaded_files))
    col2.metric("📄 Total Processed Pages", total_pages)
    
    if raw_text.strip():
        with st.spinner("Processing PDF contents..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(raw_text)
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Interactive Q&A", "📝 Smart Summary", "🎯 Exam Quiz", "🃏 Flashcards", "📐 Formulas & Terms"])
        
        # Fixed model string formulation
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        with tab1:
            st.subheader("Ask Anything About Course Materials")
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if user_query := st.chat_input("Enter your question here..."):
                st.session_state.messages.append({"role": "user", "content": user_query})
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing document..."):
                        docs = vector_store.similarity_search(user_query)
                        res = ask_gemini(llm, docs, user_query)
                        st.markdown(res)
                        st.session_state.messages.append({"role": "assistant", "content": res})

        with tab2:
            if st.button("Generate Smart Summary"):
                with st.spinner("Processing Summary..."):
                    docs = vector_store.similarity_search("Summary overview")
                    summary_res = ask_gemini(llm, docs, "মূল বিষয়বস্তু পয়েন্ট আকারে সংক্ষেপে বাংলা ও ইংরেজিতে সাজিয়ে দাও।")
                    st.write(summary_res)
                    st.download_button("📥 Download Summary (.txt)", data=summary_res, file_name=f"{selected_code}_Summary.txt")

        with tab3:
            if st.button("Generate Exam Questions"):
                with st.spinner("Generating Quiz..."):
                    docs = vector_store.similarity_search("Important concepts")
                    quiz_res = ask_gemini(llm, docs, "পরীক্ষার জন্য উপযোগী ৫টি গুরুত্বপূর্ণ প্রশ্ন ও উত্তর তৈরি করো।")
                    st.write(quiz_res)
                    st.download_button("📥 Download Quiz (.txt)", data=quiz_res, file_name=f"{selected_code}_Quiz.txt")

        with tab4:
            if st.button("Generate Study Flashcards"):
                with st.spinner("Generating Flashcards..."):
                    docs = vector_store.similarity_search("Key concepts definitions terms")
                    flash_res = ask_gemini(llm, docs, "দ্রুত রিভিশন দেওয়ার জন্য গুরুত্বপূর্ণ ১০টি টপিকের Flashcards (Term: Definition) আকারে সুন্দর করে সাজিয়ে দাও।")
                    st.write(flash_res)
                    st.download_button("📥 Download Flashcards (.txt)", data=flash_res, file_name=f"{selected_code}_Flashcards.txt")

        with tab5:
            if st.button("Extract Definitions & Formulas"):
                with st.spinner("Extracting Key Terms..."):
                    docs = vector_store.similarity_search("Definitions equations formulas")
                    formula_res = ask_gemini(llm, docs, "সব গুরুত্বপূর্ণ সংজ্ঞা এবং গাণিতিক সূত্র আলাদা তালিকা বানিয়ে দাও।")
                    st.write(formula_res)
                    st.download_button("📥 Download Formulas (.txt)", data=formula_res, file_name=f"{selected_code}_Formulas.txt")
