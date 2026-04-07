import os
import threading
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

FAISS_PATH = (BASE_DIR / ".." / "faiss_index").resolve()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(str(FAISS_PATH), embeddings, allow_dangerous_deserialization=True)

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
else:
    llm = None

retriever = db.as_retriever(search_kwargs={"k": 3})

system_prompt = '''You are a helpful assistant for answering questions based 
on the provided context. Use the retrieved documents to answer the user's question
 accurately and concisely. If you don't know the answer, say you don't know.
 Context : {context}
'''

prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

if llm:
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)
else:
    rag_chain = None

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Query(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "RAG Chatbot API is running!"}

@app.post("/query")
def query_rag(query: Query):
    if not rag_chain:
        return {"answer": "API key not configured. Please set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."}
    response = rag_chain.invoke({"input": query.text})
    return {"answer": response.get("answer", "No answer found.")}