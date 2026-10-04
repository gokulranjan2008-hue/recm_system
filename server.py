import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HF_TOKEN = os.getenv("HF_TOKEN")
PDF_PATH = os.path.join(BASE_DIR, "ESP32-S3 Water Potability & Treatment Guidelines (Comprehensive).pdf")
DB_DIR = os.path.join(BASE_DIR, "water_rag_db")
vector_db = None

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
    
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100,
        separators=["\n\nCondition:", "\n\nRule", "\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(documents)
    
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)
    print("Vector database built successfully.")
    yield

app = FastAPI(title="Water Quality Cloud RAG", lifespan=lifespan)

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=50,
    temperature=0.1,
    huggingfacehub_api_token=HF_TOKEN
)

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
    
    docs = vector_db.similarity_search(search_query, k=2)
    retrieved_text = "\n".join([d.page_content for d in docs])

    prompt = f"Context from Knowledge Base:\n{retrieved_text}\n\nFailed Parameters: {', '.join(failed_params)}.\nBased ONLY on the context, provide a maximum 10-word cause and a maximum 10-word recommendation. Format strictly as:\nCause: [text]\nRecommendation: [text]"
    
    try:
        ai_output = llm.invoke(prompt)
    except:
        ai_output = "Cause: Parameter threshold breached. Recommendation: Retest water after treatment."

    return RecommendationResponse(
        status="NOT POTABLE",
        failed_params=failed_params,
        cause=f"Failed: {', '.join(failed_params)}",
        recommendation=ai_output.strip()
    )