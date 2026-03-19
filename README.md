# 🛒 GrocerAI — Autonomous Multimodal Grocery Shopping Agent

An end-to-end autonomous shopping agent powered by GPT-4V vision, LangGraph multi-agent orchestration, FAISS RAG for user preferences, and real-time price comparison across Kroger APIs.

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

## Stack

| Layer | Technology |
|---|---|
| Vision | GPT-4V (OpenAI) |
| Orchestration | LangGraph |
| Preferences | FAISS + LangChain RAG |
| Grocery APIs | Kroger API |
| UI | Streamlit |
| Containerization | Docker + Docker Compose |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key (GPT-4V access)
- Kroger API credentials

### 1. Clone & configure
```bash
git clone <repo>
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

## Development Setup (without Docker)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run ui/app.py
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI key with GPT-4V access |
| `KROGER_CLIENT_ID` | Kroger API client ID |
| `KROGER_CLIENT_SECRET` | Kroger API client secret |
| `KROGER_LOCATION_ID` | Default store location ID |
| `FAISS_INDEX_PATH` | Path to persist FAISS index (default: `./data/faiss_index`) |
| `LOG_LEVEL` | Logging level (default: `INFO`) |

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
│   └── config.py             # App configuration
├── ui/
│   ├── app.py                # Main Streamlit app
│   ├── components/           # Reusable UI components
│   └── assets/               # Static assets
├── tests/
│   ├── test_vision_agent.py
│   ├── test_gap_agent.py
│   ├── test_search_agent.py
│   └── test_rag.py
├── data/                     # Persisted FAISS index (gitignored)
├── docker/
│   └── entrypoint.sh
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

## Agent Details

### Vision Agent
Uses GPT-4V to analyze fridge/pantry photos. Returns structured JSON inventory with item names, estimated quantities, and confidence scores.

### Gap Agent
Cross-references detected inventory against the user's grocery list. Uses the RAG preference store to filter items by dietary restrictions. Outputs a deduplicated "needs" list.

### Search Agent
Queries Kroger API for real-time pricing. Respects brand preferences from the RAG store. Returns ranked results with price-per-unit comparison.

### Cart Agent
Assembles the final cart from Search Agent results. Handles Kroger OAuth token refresh and cart API calls. Surfaces a review step before checkout.

## License
MIT
