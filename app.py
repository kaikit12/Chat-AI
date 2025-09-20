import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
import shutil

app = FastAPI(title="PDF Pal API")

# CORS cho FE gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục lưu PDF và vectorstore
UPLOAD_DIR = "uploads"
VECTOR_DIR = "vectorstore"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_DIR, exist_ok=True)

# ====== GLOBALS ======
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
vectorstore = None  # sẽ lưu FAISS index

# ====== Upload PDF ======
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Đọc PDF
    pdf_reader = PdfReader(file_path)
    text = "".join([page.extract_text() or "" for page in pdf_reader.pages])

    # Chunk & embeddings
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    return {"status": "success", "message": f"{file.filename} uploaded and processed."}

# ====== Query PDF ======
@app.post("/query")
async def query_pdf(question: str):
    global vectorstore, memory
    if vectorstore is None:
        return {"error": "No PDF uploaded yet."}

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

    answer = qa.run(question)
    return {"question": question, "answer": answer}
