import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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
    .hero-card h1, .hero-card p { color: white !important; margin: 0; }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        background-color: #4F46E5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Top Department Header
st.markdown("<h3 style='text-align: center; color: #1E293B; margin-bottom: 20px;'>🌱 Department of Environmental Science and Engineering</h3>", unsafe_allow_html=True)

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
    "MEQ": "Mid Exam Questions"
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
        
    st.caption("Designed for Academic Excellence 🚀")
    
    # Bottom Left ESE Info Card
    st.markdown("""
        <div style="background-color: #FFFFFF; padding: 12px; border-radius: 8px; border: 1px solid #CBD5E1; text-align: center;">
            <p style="margin: 0; font-size: 0.85rem; font-weight: 600; color: #059669 !important;">🌍 Green Tech & Data Hub</p>
            <p style="margin: 5px 0 0 0; font-size: 0.75rem; color: #64748B !important;">GIS • EIA • Sustainability • Lab Resources</p>
        </div>
    """, unsafe_allow_html=True)

# Top Hero Header
st.markdown(f"""
    <div class="hero-card">
        <h1>🎓 {selected_code}: {selected_title}</h1>
        <p>AI-Powered Workspace • Smart Summaries & Exam Prep</p>
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

# Fetching API Key directly from Streamlit Secrets
if uploaded_files:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Interactive Q&A", "📝 Smart Summary", "🎯 Exam Quiz", "🃏 Flashcards", "📐 Formulas & Terms"])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

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

# Bottom Right/Main Department Feature Cards
st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
        <div style="background-color: #ECFDF5; padding: 16px; border-radius: 10px; border-left: 4px solid #10B981;">
            <h4 style="margin: 0; color: #065F46 !important;">🔬 Core ESE Focus Areas</h4>
            <p style="margin-top: 5px; font-size: 0.9rem; color: #047857 !important;">
                • Hydrology & Water Management<br>
                • GIS & Remote Sensing Analytics<br>
                • Environmental Microbiology & Ecology
            </p>
        </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown("""
        <div style="background-color: #EEF2FF; padding: 16px; border-radius: 10px; border-left: 4px solid #6366F1;">
            <h4 style="margin: 0; color: #3730A3 !important;">📊 Tech Stack Tools</h4>
            <p style="margin-top: 5px; font-size: 0.9rem; color: #4338CA !important;">
                • Python for Environmental Data<br>
                • AutoCAD Engineering Designs<br>
                • Statistical Environmental Modeling
            </p>
        </div>
    """, unsafe_allow_html=True)
