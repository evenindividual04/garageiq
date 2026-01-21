# GarageIQ 🔧🤖

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LangGraph](https://img.shields.io/badge/Agents-LangGraph-orange)
![Groq](https://img.shields.io/badge/Inference-Groq%20Llama3-purple)
![Docker](https://img.shields.io/badge/Deploy-Docker-2496ED)
![Tests](https://img.shields.io/badge/Tests-50%20Passing-brightgreen)

> **AI-Powered Automotive Diagnostic System**  
> Identifies vehicle faults from vague, multilingual voice complaints using a reflective multi-agent workflow.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Project Structure](#-project-structure)

---

## 🎯 Overview

**GarageIQ** solves a real problem in automotive service: mechanics receive vague, multilingual complaints like *"gaadi garam ho rahi hai"* (car is getting hot) or *"cus sts frt lft noise"* and must diagnose the issue accurately.

This project demonstrates:
- **Multi-Agent AI Systems** using LangGraph
- **Retrieval-Augmented Generation (RAG)** for technical knowledge
- **Domain-Specific NLP** for automotive terminology
- **Production Patterns** like prompt injection defense and PII redaction

### The Problem

| Customer Says | AI Must Understand |
|---------------|-------------------|
| "Car won't start, just clicks" | ELECTRICAL → STARTER_MOTOR → NO_START |
| "brk pedal spongy, frt rt pull" | BRAKES → BRAKE_FLUID → LOW_FLUID |
| "gaadi garam ho rahi hai" | COOLING → RADIATOR → OVERHEATING |

### The Solution

GarageIQ uses 4 specialized AI agents working together:
1. **Symptom Analyst** - Extracts structured symptoms from noisy input
2. **Knowledge Agent** - Retrieves relevant repair manuals and TSBs
3. **Historian Agent** - Finds similar past cases from ticket history
4. **Diagnosis Agent** - Synthesizes all evidence into a final diagnosis

---

## ⚡ Key Features

### Core AI Capabilities

| Feature | Description |
|---------|-------------|
| **Agentic Reflection Loop** | If confidence < 70%, the AI self-corrects: refines search, re-checks knowledge base, attempts second diagnosis |
| **Multilingual Support** | Handles Hindi, English, and Hinglish (code-mixing) natively |
| **Voice Input** | Mechanics can speak complaints; uses Groq Whisper for transcription |
| **Hybrid RAG** | Combines semantic search (ChromaDB) with keyword matching for DTC codes |

### Domain-Specific Features

| Feature | Description |
|---------|-------------|
| **Noisy Input Normalization** | Expands abbreviations: "cus sts frt lft" → "customer states front left" |
| **VMRS Code Mapping** | Industry-standard Vehicle Maintenance Reporting Standards codes in every response |
| **TSB Override** | Technical Service Bulletins automatically supersede older manual procedures |
| **Parts Dependency Graph** | "Replace Water Pump" → suggests gasket, coolant, thermostat |
| **VIN Decoding** | Extracts vehicle metadata from VIN or Indian registration numbers |

### Security & Compliance

| Feature | Description |
|---------|-------------|
| **PII Redaction** | Phone, Email, Aadhaar numbers automatically masked before display |
| **Prompt Injection Defense** | Active detection and blocking of malicious inputs |
| **Closed-Loop Learning** | Captures technician feedback for continuous improvement |

---

## 🏗️ Architecture

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed system diagrams.

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                    (Streamlit / Voice Input)                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                             │
│              (Rate Limiting, Request Tracing)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Agent Orchestrator                             │
│                      (LangGraph)                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ Symptom  │→ │Knowledge │→ │Historian │→ │Diagnosis │        │
│  │ Analyst  │  │  Agent   │  │  Agent   │  │  Agent   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│                      ↑           │                               │
│                      └───────────┘ (Reflection Loop)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Knowledge Layer                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ ChromaDB │  │ Manuals  │  │  TSBs    │  │ History  │        │
│  │ Vectors  │  │   .md    │  │   .md    │  │  .json   │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Tech Stack

| Layer | Technology | Why |
|-------|------------|-----|
| **LLM** | Llama-3.1-8b (via Groq) | Fast inference (~0.2s), free tier available |
| **Agents** | LangGraph | Stateful multi-agent orchestration with cycles |
| **Vector DB** | ChromaDB | Lightweight, embedded, Python-native |
| **API** | FastAPI | Async, auto-docs, type-safe |
| **UI** | Streamlit | Rapid prototyping, built-in components |
| **Speech** | Groq Whisper | Fast transcription for voice input |
| **Container** | Docker Compose | One-command deployment |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [Groq API Key](https://console.groq.com) (free tier works)

### Option A: Docker (Recommended)

```bash
# 1. Clone the repo
git clone https://github.com/evenindividual04/garageiq.git
cd garageiq

# 2. Set your API key
export GROQ_API_KEY="your_key_here"

# 3. Run
docker-compose up -d --build

# 4. Access
# UI: http://localhost:8501
# API: http://localhost:8000/docs
```

### Option B: Local Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# 4. Run the UI
PYTHONPATH=./src streamlit run ui/streamlit_app.py

# 5. Or run the API
PYTHONPATH=./src uvicorn automotive_intent.app:app --reload
```

---

## 📡 API Reference

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/classify` | Classify a complaint into structured diagnosis |
| `POST` | `/v1/agent/chat` | Multi-turn diagnostic conversation |
| `GET` | `/v1/ontology` | Get valid system/component/failure_mode paths |
| `GET` | `/v1/vmrs/codes` | Get VMRS code mappings |
| `POST` | `/v1/feedback` | Submit technician correction |
| `GET` | `/v1/analytics` | Shop performance dashboard data |
| `GET` | `/health` | Health check |

### Example Request

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Brake noise when stopping"}'
```

### Example Response

```json
{
  "ticket_id": "TKT-20240121120000",
  "classification_status": "CONFIRMED",
  "intent": {
    "system": "BRAKES",
    "component": "PADS_ROTORS",
    "failure_mode": "SQUEALING",
    "confidence": 0.92,
    "vmrs_code": "013-001-001"
  },
  "triage": {
    "severity": "HIGH",
    "vehicle_state": "DRIVABLE_WITH_CAUTION",
    "suggested_action": "Inspect brake pads for wear"
  }
}
```

---

## 🧪 Testing

```bash
# Run full test suite (50 tests)
PYTHONPATH=./src pytest tests/test_services.py tests/test_api.py tests/test_integration.py -v

# Quick run
PYTHONPATH=./src pytest tests/ --tb=no -q
```

### Test Coverage

| Category | Tests | Description |
|----------|-------|-------------|
| Services | 23 | Normalizer, PII, VIN, Feedback, Hierarchy |
| API | 17 | Endpoints, Schemas, Pipeline |
| Integration | 10 | E2E flows, Security, Performance |

---

## 📁 Project Structure

```
garageiq/
├── src/automotive_intent/
│   ├── agents/              # LangGraph agent definitions
│   │   ├── orchestrator.py  # Main workflow controller
│   │   ├── agents.py        # Individual agent classes
│   │   └── state.py         # Shared state schema
│   ├── core/
│   │   ├── ontology.py      # Valid diagnosis paths
│   │   └── schemas.py       # Pydantic models
│   ├── services/
│   │   ├── normalizer.py    # Abbreviation expansion
│   │   ├── vin_decoder.py   # VIN/registration parsing
│   │   ├── pii_redactor.py  # Privacy protection
│   │   ├── vmrs_codes.py    # Industry code mapping
│   │   ├── feedback_loop.py # Learning system
│   │   └── analytics.py     # Performance metrics
│   ├── pipeline.py          # Classification pipeline
│   └── app.py               # FastAPI application
├── ui/
│   └── streamlit_app.py     # Web interface
├── data/
│   ├── knowledge_base/      # Repair manuals (markdown)
│   ├── tickets/             # Historical tickets (JSON)
│   └── parts_graph.json     # Parts dependencies
├── tests/
│   ├── test_services.py     # Unit tests
│   ├── test_api.py          # API tests
│   └── test_integration.py  # E2E tests
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── ARCHITECTURE.md
```

---

## 🙏 Acknowledgments

- **Groq** for fast LLM inference
- **LangChain/LangGraph** for agent orchestration framework
- **ChromaDB** for vector storage

---

*Built as a portfolio project demonstrating AI/ML engineering skills in the automotive domain.*
