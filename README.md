# 🛒 GrocerAI — Autonomous Multimodal Grocery Shopping Agent

An end-to-end autonomous shopping agent powered by GPT-4V vision, LangGraph multi-agent orchestration, FAISS RAG for user preferences, and real-time price comparison across Kroger APIs.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Railway-brightgreen)](https://grocerai-production.up.railway.app)
[![CI](https://github.com/SparshaAbinethri/GrocerAI/actions/workflows/deploy.yml/badge.svg)](https://github.com/SparshaAbinethri/GrocerAI/actions/workflows/deploy.yml)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

🔗 **Live App:** https://grocerai-production.up.railway.app

---

## Architecture

```
User Upload (photo + grocery list)
        │
        ▼
┌─────────────────────────────────────────────┐
│              LangGraph Pipeline              │
│                                             │
│  [Vision Agent] → [Gap Agent]               │
│       │                │                   │
│  Detects existing   Computes what's         │
│  fridge inventory   actually needed         │
│                        │                   │
│                        ▼                   │
│              [Search Agent]                 │
│         Kroger + Walmart price lookup       │
│                        │                   │
│                        ▼                   │
│               [Cart Agent]                  │
│          Checkout orchestration             │
└─────────────────────────────────────────────┘
        │
        ▼
  FAISS RAG Store (brand prefs, dietary restrictions)
        │
        ▼
  Streamlit UI (photo upload, cart review, checkout)
```

---

## Stack

| Layer | Technology |
|---|---|
| Vision | GPT-4V (OpenAI) |
| Orchestration | LangGraph |
| Preferences | FAISS + LangChain RAG |
| Grocery APIs | Kroger API |
| UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Deployment | Railway |
| CI/CD | GitHub Actions |
| Error Tracking | Sentry |
| Observability | Prometheus metrics, Structured JSON logging |
| Security | Rate limiting, OWASP security headers |

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key (GPT-4V access)
- Kroger API credentials

### 1. Clone & configure
```bash
git clone https://github.com/SparshaAbinethri/GrocerAI.git
cd grocerai
cp .env.example .env
# Fill in your API keys in .env
```

### 2. Run with Docker
```bash
docker-compose up --build
```

### 3. Open the UI
```
http://localhost:8501
```

---

## Development Setup (without Docker)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui/app.py
```

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI key with GPT-4V access |
| `KROGER_CLIENT_ID` | Kroger API client ID |
| `KROGER_CLIENT_SECRET` | Kroger API client secret |
| `KROGER_LOCATION_ID` | Default store location ID |
| `FAISS_INDEX_PATH` | Path to persist FAISS index (default: `./data/faiss_index`) |
| `LOG_LEVEL` | Logging level (default: `INFO`) |
| `SENTRY_DSN` | Sentry DSN for error tracking |
| `ENVIRONMENT` | Deployment environment (default: `production`) |

---

## Production & Observability

| Feature | Details |
|---|---|
| Health Check | `GET /health` — liveness probe |
| Readiness Check | `GET /ready` — readiness probe |
| Metrics | `GET /metrics` — Prometheus-compatible |
| Logging | Structured JSON logs via python-json-logger |
| Error Tracking | Sentry SDK with FastAPI integration |
| Rate Limiting | 100 requests/minute per IP (slowapi) |
| Security Headers | HSTS, CSP, X-Frame-Options, XSS protection |
| CI/CD | GitHub Actions — tests run before every deploy |
| Deployment | Auto-deploy to Railway on push to `main` |

---

## Project Structure

```
grocerai/
├── agents/
│   ├── vision_agent.py       # GPT-4V fridge inventory detection
│   ├── gap_agent.py          # Computes restocking needs
│   ├── search_agent.py       # Multi-store price lookup
│   └── cart_agent.py         # Checkout orchestration
├── rag/
│   ├── preference_store.py   # FAISS vector store for user prefs
│   └── embeddings.py         # Embedding helpers
├── api/
│   ├── kroger.py             # Kroger API client
│   └── walmart.py            # Walmart API client (stub)
├── core/
│   ├── pipeline.py           # LangGraph graph definition
│   ├── state.py              # Shared agent state schema
│   ├── config.py             # App configuration
│   ├── logging_config.py     # Structured JSON logging
│   ├── metrics.py            # Prometheus metrics setup
│   └── security.py           # Rate limiting & security headers
├── ui/
│   ├── app.py                # Main Streamlit app
│   ├── components/           # Reusable UI components
│   └── assets/               # Static assets
├── tests/
│   ├── test_health.py        # Health & readiness endpoint tests
│   ├── test_vision_agent.py
│   ├── test_gap_agent.py
│   ├── test_search_agent.py
│   └── test_rag.py
├── data/                     # Persisted FAISS index (gitignored)
├── docker/
│   └── entrypoint.sh
├── .github/
│   └── workflows/
│       └── deploy.yml        # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Agent Details

### Vision Agent
Uses GPT-4V to analyze fridge/pantry photos. Returns structured JSON inventory with item names, estimated quantities, and confidence scores.

### Gap Agent
Cross-references detected inventory against the user's grocery list. Uses the RAG preference store to filter items by dietary restrictions. Outputs a deduplicated "needs" list.

### Search Agent
Queries Kroger API for real-time pricing. Respects brand preferences from the RAG store. Returns ranked results with price-per-unit comparison.

### Cart Agent
Assembles the final cart from Search Agent results. Handles Kroger OAuth token refresh and cart API calls. Surfaces a review step before checkout.

---

