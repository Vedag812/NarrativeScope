# 🔍 NarrativeScope — Political Narrative Analyzer

An investigative analytics dashboard for analyzing political narratives on Reddit using NLP, semantic search, and network analysis.

## Features

- **Semantic Search** — Find posts by meaning, not just keywords, using sentence-transformer embeddings
- **Topic Modeling** — Discover emerging political themes with LDA-based topic extraction
- **Time-Series Analysis** — Track how narratives spread and evolve over weeks
- **Network Visualization** — Map cross-subreddit information flow and influence patterns
- **Real-time Dashboard** — Interactive analytics overview with key metrics

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python, FastAPI, Uvicorn |
| **NLP** | Sentence-Transformers, scikit-learn (LDA) |
| **Data** | JSONL dataset (10K+ Reddit posts) |
| **Analytics** | NetworkX, pandas, NumPy |
| **API** | RESTful with GZip compression |

## Quick Start

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your Gemini API key

# Run the server
uvicorn app.main:app --reload --port 8000
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | API info |
| `GET /health` | Health check with post count |
| `POST /search` | Semantic search across posts |
| `GET /timeseries` | Narrative trend analysis |
| `GET /topics` | Topic modeling results |
| `GET /network` | Subreddit interaction graph |
| `GET /dashboard` | Aggregated analytics |

## Project Structure

```
NarrativeScope/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI entry point
│   │   ├── core/
│   │   │   ├── config.py    # Settings & CORS config
│   │   │   └── data_loader.py  # JSONL data ingestion
│   │   ├── routers/         # API route handlers
│   │   │   ├── search.py
│   │   │   ├── timeseries.py
│   │   │   ├── topics.py
│   │   │   ├── network.py
│   │   │   └── dashboard.py
│   │   └── services/        # Business logic
│   │       ├── embeddings.py    # Sentence-transformer encoding
│   │       ├── search.py        # Cosine similarity search
│   │       ├── timeseries.py    # Temporal trend analysis
│   │       ├── topics.py        # LDA topic extraction
│   │       ├── network.py       # Graph construction
│   │       └── genai.py         # Gemini AI summaries
│   ├── data/
│   │   └── data.jsonl       # Reddit post dataset
│   └── requirements.txt
└── README.md
```

## Contributors

- [Vedant Agarwal](https://github.com/Vedag812)
- [Tanishka Poddar](https://github.com/Tan1725)

## License

MIT
