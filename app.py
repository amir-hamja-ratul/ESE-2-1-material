import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import os

st.set_page_config(page_title="EduHub - 2nd Year 1st Sem", page_icon="🎓", layout="wide")

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
    "ESE 2113": "Statistics for Environment"
}

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3429/3429149.png", width=70)
    st.title("Semester Workspace")
    api_key = st.text_input("🔑 Enter Gemini API Key", type="password")
    selected_code = st.selectbox("📌 Select Course Code", list(COURSES.keys()))
    st.info(f"**Course Title:**\n{COURSES[selected_code]}")

st.title(f"🎓 {selected_code}: {COURSES[selected_code]}")
uploaded_files = st.file_uploader("📥 আপলোড করুন (PDF Documents)", accept_multiple_files=True, type="pdf")

if uploaded_files and api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    raw_text = ""
    for pdf in uploaded_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            raw_text += page.extract_text() or ""
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Chat", "📝 Summary", "🎯 Quiz", "📐 Formulas"])
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    chain = load_qa_chain(llm, chain_type="stuff")

    with tab1:
        user_query = st.text_input("প্রশ্ন লিখুন:")
        if user_query:
            docs = vector_store.similarity_search(user_query)
            st.write(chain.run(input_documents=docs, question=user_query))

    with tab2:
        if st.button("Generate Summary"):
            docs = vector_store.similarity_search("Summary overview")
            st.write(chain.run(input_documents=docs, question="মূল বিষয়বস্তু পয়েন্ট আকারে সংক্ষেপে বাংলা ও ইংরেজিতে সাজিয়ে দাও।"))

    with tab3:
        if st.button("Generate Quiz"):
            docs = vector_store.similarity_search("Important concepts")
            st.write(chain.run(input_documents=docs, question="পরীক্ষার জন্য উপযোগী ৫টি গুরুত্বপূর্ণ প্রশ্ন ও উত্তর তৈরি করো।"))

    with tab4:
        if st.button("Extract Formulas & Terms"):
            docs = vector_store.similarity_search("Definitions equations formulas")
            st.write(chain.run(input_documents=docs, question="সব গুরুত্বপূর্ণ সংজ্ঞা এবং গাণিতিক সূত্র আলাদা তালিকা বানিয়ে দাও।"))
