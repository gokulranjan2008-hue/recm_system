import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API")
)
if not GEMINI_KEY:
    raise ValueError("Gemini API key is not set. Please set GEMINI_API_KEY or GOOGLE_API in your .env file.")

os.environ["GOOGLE_API_KEY"] = GEMINI_KEY
if "GEMINI_API_KEY" in os.environ:
    del os.environ["GEMINI_API_KEY"]

from server import extract_conditions, DB_DIR, init_vector_db, get_llm
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

print("--- Step 1: Initializing ChromaDB Vector Store with Gemini Embeddings ---")
init_vector_db()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_KEY
)
vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
print(f"ChromaDB ready at: {DB_DIR}")

# Test input: Acidic water (pH 5.8)
test_ph, test_tds, test_ec = 5.8, 250.0, 0.5
failed_params, search_query = extract_conditions(test_ph, test_tds, test_ec)

print(f"\nExtracted Failed Parameters: {failed_params}")
print(f"Generated Vector Search Query: '{search_query}'")

docs = vector_db.similarity_search(search_query, k=2)
retrieved_text = "\n".join([d.page_content for d in docs])
print("\n[Retrieved PDF Context]:")
print(retrieved_text)

print("\n--- Step 2: Querying Gemini Model ---")
prompt = f"""Context from Knowledge Base:
{retrieved_text}

Failed Parameters: {', '.join(failed_params)}.
Based ONLY on the context, provide a root cause and exactly 3 recommendation steps.
CRITICAL: To fit on a small microcontroller screen, keep the cause and EACH recommendation step strictly under 8 words. Do not use long sentences.

Format strictly as:
Cause:
- [Short cause under 8 words]

Recommendation:
1. [Short step under 8 words]
2. [Short step under 8 words]
3. [Short step under 8 words]"""

llm = get_llm()
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

print("\n================ FINAL AI TEST OUTPUT ================")
print(ai_output.strip())
print("======================================================")