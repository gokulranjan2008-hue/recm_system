try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from huggingface_hub import InferenceClient

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "ESP32-S3 Water Potability & Treatment Guidelines (Comprehensive).pdf")
DB_DIR = os.path.join(BASE_DIR, "water_rag_db")
vector_db = None

logger = logging.getLogger("uvicorn")

class WaterData(BaseModel):
    ph: float
    tds_ppm: float
    ec_ms_cm: float
    potability: str

class RecommendationResponse(BaseModel):
    status: str
    failed_params: list[str]
    cause: str
    recommendation: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_db
    print("Initializing Cloud Vector DB...")
    
    hf_token = os.getenv("HF_TOKEN")
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=hf_token
    )

    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print(f"Loading existing Vector DB from {DB_DIR}...")
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    else:
        print("Building Vector DB from PDF...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=100,
            separators=["\n\nCondition:", "\n\nRule", "\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)

    print("Vector database ready.")
    yield

app = FastAPI(title="Water Quality Cloud RAG", lifespan=lifespan)

# Allow CORS for web clients and external dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_hf_client():
    token = os.getenv("HF_TOKEN")
    return InferenceClient(token=token) if token else None

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "Water Quality Potability & Treatment Recommendation API",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}

def extract_conditions(ph: float, tds: float, ec: float):
    failed, queries = [], []
    if ph < 6.50:
        failed.append("pH (Acidic)")
        queries.append("Acidic water pH below 6.5 causes and treatment")
    elif ph > 8.50:
        failed.append("pH (Alkaline)")
        queries.append("Alkaline water pH above 8.5 causes and treatment")
    if tds > 500.0:
        failed.append("TDS (High)")
        queries.append("High TDS above 500 ppm causes and reverse osmosis treatment")
    return failed, " ".join(queries)

@app.post("/api/recommend", response_model=RecommendationResponse)
async def get_recommendation(data: WaterData):
    if data.potability == "POTABLE" and (6.5 <= data.ph <= 8.5) and data.tds_ppm <= 500:
        return RecommendationResponse(
            status="POTABLE", failed_params=[],
            cause="None", recommendation="Water is safe for general use."
        )

    failed_params, search_query = extract_conditions(data.ph, data.tds_ppm, data.ec_ms_cm)
    
    retrieved_text = ""
    if vector_db:
        docs = vector_db.similarity_search(search_query, k=2)
        retrieved_text = "\n".join([d.page_content for d in docs])

    prompt = f"Context from Knowledge Base:\n{retrieved_text}\n\nFailed Parameters: {', '.join(failed_params)}.\nBased ONLY on the context, provide a maximum 10-word cause and a maximum 10-word recommendation. Format strictly as:\nCause: [text]\nRecommendation: [text]"
    
    client = get_hf_client()
    ai_output = None
    if client:
        try:
            response = client.chat_completion(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.1
            )
            ai_output = response.choices[0].message.content
        except Exception as e:
            logger.error(f"InferenceClient error: {e}")
            ai_output = None

    if not ai_output:
        ai_output = "Cause: Parameter threshold breached. Recommendation: Retest water after treatment."

    return RecommendationResponse(
        status="NOT POTABLE",
        failed_params=failed_params,
        cause=f"Failed: {', '.join(failed_params)}",
        recommendation=ai_output.strip()
    )