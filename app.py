import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os

# Page Config
st.set_page_config(page_title="EduHub - Academic AI Assistant", page_icon="🎓", layout="wide")

# Custom UI CSS Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #4F46E5; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

COURSES = {
    "PYQ": "Previous Year Questions",
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
    "ESE 2113": "Statistics for Environment"
}

# Sidebar
# কোড এবং নাম একত্রে লিস্ট তৈরি
course_options = [f"{code} - {title}" for code, title in COURSES.items()]

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3429/3429149.png", width=70)
    st.title("Semester Workspace")
    api_key = st.text_input("🔑 Enter Gemini API Key", type="password")
    
    # নাম বা কোড টাইপ করে সার্চ করার ড্রপডাউন
    selected_option = st.selectbox("📌 Select Course (Code or Name)", course_options)
    
    # অপশন থেকে কোড ও নাম আলাদা করা
    selected_code = selected_option.split(" - ")[0]
    selected_title = COURSES[selected_code]
    
    st.info(f"**Course Title:**\n{selected_title}")
    st.divider()
    st.caption("Developed for Academic Excellence 🚀")

# Main Inputs
uploaded_files = st.file_uploader("📥 আপলোড করুন (PDF Documents)", accept_multiple_files=True, type="pdf")

def ask_gemini(llm, docs, question):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"নিচের তথ্যগুলোর ওপর ভিত্তি করে প্রশ্নের উত্তর দাও:\n\n{context}\n\nপ্রশ্ন: {question}"
    response = llm.invoke(prompt)
    return response.content

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_files and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    raw_text = ""
    total_pages = 0
    
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        total_pages += len(pdf_reader.pages)
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""
            
    # Display Stats
    col1, col2 = st.columns(2)
    col1.metric("📂 Uploaded Files", len(uploaded_files))
    col2.metric("📄 Total Pages Processed", total_pages)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Interactive Chat", "📝 Smart Summary", "🎯 Exam Quiz", "📐 Formulas & Terms"])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # Tab 1: ChatGPT Style Interactive Chat
    with tab1:
        st.subheader("কোর্স সংক্রান্ত যেকোনো প্রশ্ন করুন")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_query := st.chat_input("Ask a question about your uploaded materials..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing document..."):
                    docs = vector_store.similarity_search(user_query)
                    res = ask_gemini(llm, docs, user_query)
                    st.markdown(res)
                    st.session_state.messages.append({"role": "assistant", "content": res})

    # Tab 2: Summary with Download Button
    with tab2:
        if st.button("Generate Course Summary"):
            with st.spinner("Creating Summary..."):
                docs = vector_store.similarity_search("Summary overview")
                summary_res = ask_gemini(llm, docs, "মূল বিষয়বস্তু পয়েন্ট আকারে সংক্ষেপে বাংলা ও ইংরেজিতে সাজিয়ে দাও।")
                st.write(summary_res)
                st.download_button("📥 Download Summary (.txt)", data=summary_res, file_name=f"{selected_code}_Summary.txt")

    # Tab 3: Quiz with Download Button
    with tab3:
        if st.button("Generate Practice Quiz"):
            with st.spinner("Generating Questions..."):
                docs = vector_store.similarity_search("Important concepts")
                quiz_res = ask_gemini(llm, docs, "পরীক্ষার জন্য উপযোগী ৫টি গুরুত্বপূর্ণ প্রশ্ন ও উত্তর তৈরি করো।")
                st.write(quiz_res)
                st.download_button("📥 Download Quiz (.txt)", data=quiz_res, file_name=f"{selected_code}_Quiz.txt")

    # Tab 4: Formulas & Definitions
    with tab4:
        if st.button("Extract Definitions & Formulas"):
            with st.spinner("Extracting Key Terms..."):
                docs = vector_store.similarity_search("Definitions equations formulas")
                formula_res = ask_gemini(llm, docs, "সব গুরুত্বপূর্ণ সংজ্ঞা এবং গাণিতিক সূত্র আলাদা তালিকা বানিয়ে দাও।")
                st.write(formula_res)
                st.download_button("📥 Download Formulas (.txt)", data=formula_res, file_name=f"{selected_code}_Formulas.txt")
