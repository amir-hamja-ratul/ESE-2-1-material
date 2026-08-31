import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import glob

# Page Config
st.set_page_config(page_title="EduHub - Academic AI Assistant", page_icon="🎓", layout="wide")

# Modern Light Theme CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #F8F9FA; color: #1E293B !important; }
    [data-testid="stSidebar"] { background-color: #E9ECEF; }
    
    html, body, p, label, span, h1, h2, h3, h4, .stMarkdown { color: #1E293B !important; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span, [data-testid="stSidebar"] div { color: #1E293B !important; }
    
    .stButton>button, [data-testid="stDownloadButton"]>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        background-color: #4F46E5 !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    
    .hero-card {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        padding: 24px;
        border-radius: 12px;
        color: white !important;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .hero-card h1 { color: white !important; margin: 0; }
    </style>
""", unsafe_allow_html=True)

# Top Department Header
st.markdown("<h2 style='text-align: center; color: #1E293B; font-size: 1.8rem; font-weight: 700; margin-bottom: 24px;'>🌱 Department of Environmental Science and Engineering</h2>", unsafe_allow_html=True)

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
}

course_options = [f"{code} - {title}" for code, title in COURSES.items()]

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3429/3429149.png", width=65)
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
    col1.metric("📂 Loaded Files", files_count)
    col2.metric("📄 Total Processed Pages", total_pages)

    if local_pdfs and not uploaded_files:
        st.subheader("📥 Download Available Resources")
        for pdf_file in local_pdfs:
            with open(pdf_file, "rb") as f:
                file_name = os.path.basename(pdf_file)
                st.download_button(label=f"📄 Download {file_name}", data=f, file_name=file_name, mime="application/pdf")

    with st.spinner("Processing PDF contents..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(raw_text)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Interactive Q&A", "📝 Smart Summary", "🎯 Exam Quiz", "🃏 Flashcards", "📐 Formulas & Terms"])
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0.3)

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
