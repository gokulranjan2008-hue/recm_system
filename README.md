# Water Quality Potability & Treatment Recommendation System

A Cloud RAG (Retrieval-Augmented Generation) backend API built with FastAPI, LangChain, ChromaDB, and Google Gemini models. Designed to process water quality telemetry from ESP32-S3 sensors and deliver actionable potability diagnoses, root cause analyses, and water treatment recommendations.

---

## 🌊 Overview

The system evaluates water parameters against standard drinking water safety thresholds:
- **pH**: Safe range `6.5 – 8.5`
- **TDS (Total Dissolved Solids)**: Ideal range `50 – 500 ppm`
- **EC (Electrical Conductivity)**: Ideal range `0 – 800 µS/cm`

When parameters breach safe thresholds, the backend queries a domain-specific knowledge base (`water_rag.pdf`) via ChromaDB vector search with Google Gemini Embeddings (`models/gemini-embedding-001`) and queries Google Gemini (`gemini-3.5-flash-lite`) to generate concise, human-readable causes and corrective filtration/treatment recommendations formatted strictly for microcontroller screens.

---

## 🚀 Features

- **FastAPI Endpoint**: Real-time evaluation endpoint (`POST /api/recommend`) for ESP32-S3 or web clients.
- **RAG Knowledge Base**: Uses ChromaDB and Google Gemini embeddings (`models/gemini-embedding-001`) to ground recommendations on water treatment guidelines.
- **Root Cause & Treatment Pipeline**: Identifies specific failure conditions (acidic, alkaline, high TDS) and suggests appropriate treatment stages (calcite filters, RO membranes, remineralizers).
- **Self-Healing Vector DB**: Automatically detects and rebuilds the vector index from PDF if missing or if embedding dimensions change.
- **Cloud-Ready**: Fully configured for 1-click deployment on Render with automatic environment configuration.

---

## 📁 Repository Structure

```text
├── water_rag.pdf      # Domain knowledge base
├── server.py          # FastAPI application & Gemini RAG pipeline
├── test.py            # Verification & standalone test script
├── requirements.txt   # Python dependencies
├── render.yaml        # Render Blueprint configuration
├── .gitignore         # Git ignore rules (.env, venv, Chroma DB)
└── README.md          # Project documentation
```

---

## 🛠️ Setup & Installation

### 1. Clone Repository

```bash
git clone https://github.com/gokulranjan2008-hue/recm_system.git
cd recm_system
```

### 2. Create Virtual Environment

```bash
python -m venv .venv

# Windows:
.\.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Accepts `GEMINI_API_KEY`, `GOOGLE_API_KEY`, or `GOOGLE_API`)*

---

## 🚦 Running the Application

### Start the FastAPI Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- Swagger UI: `http://localhost:8000/docs`
- Redoc: `http://localhost:8000/redoc`

### Test the Pipeline Standalone

```bash
python test.py
```

---

## 📡 API Reference

### Evaluate Water Sample

- **Endpoint**: `POST /api/recommend`
- **Request Body**:

```json
{
  "ph": 5.4,
  "tds_ppm": 200.0,
  "ec_ms_cm": 0.4,
  "potability": "NOT POTABLE"
}
```

- **Sample Response**:

```json
{
  "status": "NOT POTABLE",
  "failed_params": [
    "pH (Acidic)"
  ],
  "cause": "Failed: pH (Acidic)",
  "recommendation": "Cause:\n- Water pH is acidic (4.0 - 4.99)\n\nRecommendation:\n1. Pass water through a dual-tank limestone bed.\n2. Periodically backwash the limestone bed.\n3. Ensure automated pump maintains neutral pH."
}
```

---

## ☁️ Deploying to Render

You can deploy this FastAPI backend to [Render](https://render.com) using either **Render Blueprints (Automated)** or **Manual Setup**.

### Method 1: Render Blueprints (Recommended - 1 Click)

1. Push your updated code to GitHub:
   ```bash
   git add .
   git commit -m "Switch to Google Gemini and prepare Render deployment"
   git push origin main
   ```
2. Log in to [dashboard.render.com](https://dashboard.render.com).
3. Click **New +** > **Blueprint**.
4. Connect and select your `recm_system` repository.
5. Render will automatically detect [`render.yaml`](./render.yaml).
6. When prompted for `GEMINI_API_KEY`, enter your Google Gemini API key.
7. Click **Apply**. Render will build and deploy the web service with health checks configured.

### Method 2: Manual Web Service Setup

1. In Render Dashboard, click **New +** > **Web Service**.
2. Connect your GitHub repository `recm_system`.
3. Fill in the following settings:
   - **Name**: `water-rag-recommendation-api`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn server:app --host 0.0.0.0 --port $PORT`
   - **Plan**: `Free`
4. In the **Environment Variables** section, add:
   - `PYTHON_VERSION`: `3.11.9`
   - `GEMINI_API_KEY`: *(your Google Gemini API Key)*
5. Click **Create Web Service**.

Your live API will be accessible at:
`https://<your-service-name>.onrender.com` (Interactive docs at `/docs`, health check at `/health`).

---

## 📜 License

This project is licensed under the MIT License.
