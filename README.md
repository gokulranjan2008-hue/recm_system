# Water Quality Potability & Treatment Recommendation System

A Cloud RAG (Retrieval-Augmented Generation) backend API built with FastAPI, LangChain, ChromaDB, and Hugging Face models. Designed to process water quality telemetry from ESP32-S3 sensors and deliver actionable potability diagnoses, root cause analyses, and water treatment recommendations.

---

## 🌊 Overview

The system evaluates water parameters against standard drinking water safety thresholds:
- **pH**: Safe range `6.5 – 8.5`
- **TDS (Total Dissolved Solids)**: Ideal range `50 – 500 ppm`
- **EC (Electrical Conductivity)**: Ideal range `0 – 800 µS/cm`

When parameters breach safe thresholds, the backend queries a domain-specific knowledge base (`ESP32-S3 Water Potability & Treatment Guidelines (Comprehensive).pdf`) via ChromaDB vector search and queries an LLM to generate concise, human-readable causes and corrective filtration/treatment recommendations.

---

## 🚀 Features

- **FastAPI Endpoint**: Real-time evaluation endpoint (`POST /api/recommend`) for ESP32-S3 or web clients.
- **RAG Knowledge Base**: Uses ChromaDB and Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`) to ground recommendations on water treatment guidelines.
- **Root Cause & Treatment Pipeline**: Identifies specific failure conditions (acidic, alkaline, hard water, saline, sensor faults) and suggests appropriate treatment stages (e.g., calcite filters, RO membranes, remineralizers).
- **Extensible & Lightweight**: Easy to run locally or containerize for cloud deployment.

---

## 📁 Repository Structure

```text
├── ESP32-S3 Water Potability & Treatment Guidelines (Comprehensive).pdf  # Domain knowledge base
├── server.py                                                              # FastAPI application & RAG pipeline
├── test.py                                                                # Verification & standalone test script
├── requirements.txt                                                       # Python dependencies
├── .gitignore                                                             # Git ignore rules (.env, venv, Chroma DB)
└── README.md                                                              # Project documentation
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
HF_TOKEN=your_huggingface_access_token_here
```

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
  "ph": 5.8,
  "tds_ppm": 250.0,
  "ec_ms_cm": 0.5,
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
  "recommendation": "Cause: Mild acidity. Recommendation: Install calcite media filter."
}
```

---

## 📜 License

This project is licensed under the MIT License.
