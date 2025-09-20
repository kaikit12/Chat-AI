import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ===== CONFIG =====
st.set_page_config(page_title="Chat với PDF", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
.stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 8px 16px; }
.stTextInput>div>input { border-radius: 8px; border: 1px solid #4CAF50; }
.stFileUploader { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #ddd; }
h1,h2,h3 { color: #2c3e50; font-family: 'Arial', sans-serif; }
.chat-container { max-height: 400px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 10px; background-color: #ffffff; margin-bottom: 20px; }
.user-message { background-color: #4CAF50; color: white; padding: 10px; border-radius: 10px; margin: 5px 10px 5px 50%; max-width: 45%; word-wrap: break-word; }
.ai-message { background-color: #e0e0e0; color: #2c3e50; padding: 10px; border-radius: 10px; margin: 5px 10px 5px 5px; max-width: 45%; word-wrap: break-word; }
</style>
""", unsafe_allow_html=True)

# ===== Session State =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ===== App =====
st.title("📄 Chat với PDF (Powered by Groq Llama3)")

# Sidebar
with st.sidebar:
    st.header("⚙️ Cài đặt")
    st.info("Tải lên file PDF và đặt câu hỏi để nhận câu trả lời từ nội dung PDF.")
    pdf_file = st.file_uploader("📎 Tải lên file PDF", type=["pdf"], help="Chọn file PDF để bắt đầu phân tích.")

    if st.button("🗑️ Xóa lịch sử trò chuyện"):
        st.session_state.chat_history = []
        st.success("Lịch sử trò chuyện đã được xóa!")

# Main
if pdf_file:
    with st.spinner("Đang xử lý file PDF..."):
        pdf_reader = PdfReader(pdf_file)
        text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0
        )

        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        st.success("PDF đã được xử lý thành công!")

    # Chat container
    st.subheader("💬 Trò chuyện với PDF")
    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            cls = "user-message" if message["role"] == "user" else "ai-message"
            st.markdown(f'<div class="{cls}">{message["content"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    query = st.text_input("Hỏi về nội dung PDF:", placeholder="Nhập câu hỏi của bạn...", key="query_input")
    if query:
        with st.spinner("Đang tìm kiếm câu trả lời..."):
            result = qa.run(query)
            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "ai", "content": result})
            st.experimental_rerun()

else:
    st.warning("Vui lòng tải lên file PDF để bắt đầu.")

st.markdown("<hr><p style='text-align: center; color: #7f8c8d;'>Built with ❤️ by Streamlit & Groq</p>", unsafe_allow_html=True)
