# main.py — логіка агента CyberMentorAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Завантаження PDF
loader = PyPDFLoader("cybersecurity_basics.pdf")
pages = loader.load()

# 2. Розбиття на чанки
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = text_splitter.split_documents(pages)

# 3. Векторна база (embeddings + FAISS)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 4. Завантаження LLM моделі (українська підтримка)
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/mt5-base",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)
