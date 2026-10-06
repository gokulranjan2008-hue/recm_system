import os

from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set. Please set it in your environment or in a .env file.")

from server import extract_conditions, PDF_PATH, DB_DIR
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma

print("--- Step 1: Loading PDF Knowledge Base ---")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"Successfully loaded {len(documents)} pages from PDF.")

print("\n--- Step 2: Querying ChromaDB Vector Store ---")
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,  
    separators=["\n\nCondition:", "\n\nRule", "\n\n", "\n", " "]
)
chunks = text_splitter.split_documents(documents)
vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_DIR)

# Test input: Acidic water (pH 5.8)
test_ph, test_tds, test_ec =6.5 , 250.0, 0.5
failed_params, search_query = extract_conditions(test_ph, test_tds, test_ec)

print(f"Extracted Failed Parameters: {failed_params}")
print(f"Generated Vector Search Query: '{search_query}'")

docs = vector_db.similarity_search(search_query, k=1)
retrieved_text = docs[0].page_content
print("\n[Retrieved PDF Context]:")
print(retrieved_text)

print("\n--- Step 3: Querying Hugging Face LLM (Phi-3) ---")

from huggingface_hub import InferenceClient

# Use Hugging Face's native client for conversational models
client = InferenceClient(token=os.getenv("HF_TOKEN"))

prompt = f"""Context from Knowledge Base:
{retrieved_text}

Failed Parameters: {', '.join(failed_params)}.
Based ONLY on the context, provide a root cause and exactly 3 recommendation steps.
CRITICAL: To fit on a small microcontroller screen, keep the cause with no sensor malfunction and EACH recommendation step. Uselong sentences.

Format strictly as:
Cause:
- [Short cause]

Recommendation:
1. [Short step under 10 words]
2. [Short step under 10 words
3. [Short step under 10 words]"""

# Call the chat_completion API instead of text-generation
# Call the chat_completion API with a supported free-tier model
response = client.chat_completion(
    model="Qwen/Qwen2.5-72B-Instruct", 
    messages=[{"role": "user", "content": prompt}],
    max_tokens=200,
    temperature=0.1
)

ai_response = response.choices[0].message.content

print("\n================ FINAL AI TEST OUTPUT ================")
print(ai_response.strip())
print("======================================================")