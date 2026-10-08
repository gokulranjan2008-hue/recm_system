try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import shutil
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "water_rag.pdf")
DB_DIR = os.path.join(BASE_DIR, "water_rag_db")
vector_db = None

logger = logging.getLogger("uvicorn")

# Normalize Gemini API key from various common env variable names
GEMINI_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API")
)
if GEMINI_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_KEY
    if "GEMINI_API_KEY" in os.environ:
        del os.environ["GEMINI_API_KEY"]

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

def init_vector_db():
    global vector_db
    if not GEMINI_KEY:
        logger.warning("No Gemini API key configured. Vector DB initialization deferred.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=GEMINI_KEY,
    )

    should_build = False
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        try:
            logger.info(f"Loading existing Vector DB from {DB_DIR}...")
            vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
            # Perform a test search to verify collection dimensions match
            vector_db.similarity_search("water test", k=1)
            logger.info("Existing Vector DB loaded successfully.")
        except Exception as e:
            logger.warning(f"Existing Vector DB incompatible or corrupt ({e}). Rebuilding...")
            should_build = True
            vector_db = None
    else:
        should_build = True

    if should_build:
        if os.path.exists(DB_DIR):
            try:
                shutil.rmtree(DB_DIR)
            except Exception as e:
                logger.warning(f"Could not remove existing DB_DIR: {e}")

        logger.info(f"Building Vector DB from {PDF_PATH}...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\nCondition:", "\n\nRule", "\n\n", "\n", " "]
        )
        chunks = text_splitter.split_documents(documents)
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        logger.info("Vector database built and persisted.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Cloud Vector DB with Gemini...")
    init_vector_db()
    logger.info("Vector database initialization complete.")
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

def get_llm():
    if not GEMINI_KEY:
        return None
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    try:
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GEMINI_KEY,
            max_output_tokens=250,
        )
    except Exception as e:
        logger.error(f"Error initializing ChatGoogleGenerativeAI: {e}")
        return None

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "Water Quality Potability & Treatment Recommendation API",
        "provider": "Google Gemini",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "gemini_configured": bool(GEMINI_KEY),
        "vector_db_ready": vector_db is not None
    }

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
            status="POTABLE",
            failed_params=[],
            cause="None",
            recommendation="Water is safe for general use."
        )

    failed_params, search_query = extract_conditions(data.ph, data.tds_ppm, data.ec_ms_cm)
    
    retrieved_text = ""
    if vector_db:
        try:
            docs = vector_db.similarity_search(search_query, k=2)
            retrieved_text = "\n".join([d.page_content for d in docs])
        except Exception as e:
            logger.error(f"Error querying Chroma vector DB: {e}")

    prompt = f"""Context from Knowledge Base:
{retrieved_text}

Failed Parameters: {', '.join(failed_params)}.
Based ONLY on the context, provide a highly detailed root cause and exactly 3 comprehensive recommendation steps.
CRITICAL: Do NOT include or repeat live sensor readings. Instead, explicitly include the Knowledge Base threshold ranges, ideal target parameters, exact hardware specifications, chemical names, and physical/health risks.

Format strictly as:
Cause:
- [Detailed technical cause including parameter threshold ranges, violation classification, and physical/health risks]

Recommendation:
1. [Detailed technical step including exact hardware specifications, chemical names, and target parameter values]
2. [Detailed technical step detailing equipment mechanisms, maintenance intervals, or dosing protocols]
3. [Detailed technical step focusing on operational limits, hazard mitigation, or telemetry monitoring]"""
    
    llm = get_llm()
    ai_output = None
    if llm:
        try:
            response = llm.invoke(prompt)
            if isinstance(response.content, str):
                ai_output = response.content
            elif isinstance(response.content, list):
                parts = []
                for part in response.content:
                    if isinstance(part, dict):
                        parts.append(part.get("text", ""))
                    elif hasattr(part, "text"):
                        parts.append(part.text)
                    else:
                        parts.append(str(part))
                ai_output = "".join(parts)
            else:
                ai_output = str(response.content)
        except Exception as e:
            logger.error(f"Gemini LLM error: {e}")
            ai_output = None

    if not ai_output or not ai_output.strip():
        ai_output = "Cause:\n- Parameter threshold breached\n\nRecommendation:\n1. Calibrate sensors\n2. Inspect filtration\n3. Retest water sample"

    return RecommendationResponse(
        status="NOT POTABLE",
        failed_params=failed_params,
        cause=f"Failed: {', '.join(failed_params)}",
        recommendation=ai_output.strip()
    )