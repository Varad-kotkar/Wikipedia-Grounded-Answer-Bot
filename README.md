# WikiTruth AI — Wikipedia-Grounded Answer Bot

> **P53 College Mini-Project** | MIT ADT University | Generative AI Course | Deadline: 7 November 2026

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-green.svg)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**WikiTruth AI** is a Retrieval-Augmented Generation (RAG) application that answers natural-language questions *exclusively* using retrieved Wikipedia passages. It compares grounded and ungrounded LLM responses, generates topic illustrations, and provides a comprehensive evaluation dashboard.

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **RAG Pipeline** | Wikipedia → Chunk → Embed (Sentence-BERT) → ChromaDB → Retrieve → Generate |
| **No API Key Required** | Local Flan-T5 fallback works offline |
| **Grounded vs Ungrounded** | Side-by-side comparison of RAG vs raw LLM answers |
| **Prompt Engineering** | Zero-shot and few-shot grounding prompts compared |
| **Image Generation** | Topic illustrations via HF Inference API (matplotlib fallback) |
| **Evaluation Dashboard** | 20-question evaluation with hallucination rate metrics |
| **Source Links** | Clickable Wikipedia article links for every answer |

## 🏗️ RAG Architecture

```
User Question
     │
     ▼
Question Processing
     │
     ▼
Sentence-BERT Embedding (all-MiniLM-L6-v2)
     │
     ▼
Wikipedia Vector Database (ChromaDB, cosine similarity)
     │
     ▼
Top-K Passage Retrieval
     │
     ▼
Context Validation (similarity threshold check)
     │
     ▼
Grounding Prompt (zero-shot or few-shot)
     │
     ▼
Transformer LLM (Claude / GPT-3.5 / Flan-T5)
     │
     ▼
Answer + Evidence + Source Links
```

## 🚀 Quick Start

### 1. Clone and set up
```bash
git clone <your-repo-url>
cd WikiTruth-AI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (optional)
```bash
cp .env.example .env
# Edit .env to add API keys (optional — app works without them)
```

### 3. Ingest Wikipedia articles
```bash
python main.py ingest
```
This downloads 10 Wikipedia AI/ML articles, chunks them, creates embeddings, and stores in ChromaDB. Takes ~2-5 minutes on first run.

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### 5. CLI usage (optional)
```bash
python main.py query "What is machine learning?"
python main.py query "How does deep learning work?" --top-k 3
python main.py evaluate --auto --output evaluation/results/grounded_results.json
```

## 📱 Application Pages

1. **Home** — Project introduction, RAG architecture, technology stack
2. **Ask WikiTruth** — Ask questions, get grounded answers with source links and illustrations
3. **Grounded vs Ungrounded** — Side-by-side comparison showing RAG vs raw LLM
4. **Evaluation Dashboard** — 20-question evaluation results with charts
5. **Prompt Engineering Lab** — Compare zero-shot vs few-shot prompting
6. **About Project** — Architecture, syllabus units, responsible AI notes

## 🧠 Models Used

| Component | Model | Type |
|-----------|-------|------|
| Embeddings | `all-MiniLM-L6-v2` | Sentence-BERT (local, free) |
| LLM (default) | `google/flan-t5-base` | Local transformer (free) |
| LLM (optional) | Claude Haiku | Anthropic API |
| LLM (optional) | GPT-3.5 Turbo | OpenAI API |
| Image generation | `stable-diffusion-2-1` | HF Inference API (free tier) |

## 📊 Project Units (P53 Syllabus)

- **U1 — Grounding & Prompt Engineering**: Zero-shot/few-shot grounding prompts, hallucination prevention
- **U2 — RAG Implementation**: Complete RAG pipeline with ChromaDB, Sentence-BERT, cosine retrieval
- **U3 — GPT+BERT+Multimodal**: Both model families + image generation
- **U4 — PyTorch + Image Generation**: torch-based embeddings + SD image generation
- **U5 — Responsible AI + Evaluation**: 20-question dual-track evaluation with hallucination metrics

## 🗂️ Project Structure

```
WikiTruth-AI/
├── app.py                    # Streamlit application (6 pages)
├── main.py                   # CLI entry point
├── pipeline.py               # RAG pipeline orchestrator
├── src/
│   ├── config.py             # Configuration
│   ├── data_collection.py    # Wikipedia article fetcher
│   ├── chunking.py           # Token-based chunker (tiktoken)
│   ├── embeddings.py         # Sentence-BERT embeddings
│   ├── vector_store.py       # ChromaDB wrapper
│   ├── retrieval.py          # Cosine similarity retrieval
│   ├── generation.py         # LLM generation (Claude/GPT/Flan-T5)
│   ├── grounding.py          # Grounding prompts + status detection
│   ├── image_generator.py    # Topic illustration generation
│   ├── wikipedia_retriever.py # URL builder + source links
│   └── utils.py              # CSV/JSON export, metric helpers
├── evaluation/
│   ├── test_questions.json   # 20 Q&A evaluation dataset
│   ├── evaluate.py           # Evaluation harness
│   └── results/              # Generated evaluation results
├── tests/                    # Automated tests (pytest)
├── assets/                   # Architecture diagrams
├── .streamlit/config.toml    # Streamlit theme
├── .env.example              # Environment variable template
├── requirements.txt
├── README.md
└── MODEL_CARD.md
```

## ☁️ Deployment (Streamlit Community Cloud)

1. Push to GitHub (public repository)
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Add secrets in the Streamlit Cloud dashboard (not in code):
   - `ANTHROPIC_API_KEY` (optional)
   - `OPENAI_API_KEY` (optional)
   - `HF_TOKEN` (optional)
6. The app works without any secrets using the local Flan-T5 fallback

> **Important**: Run `python main.py ingest` and commit `chroma_db/` and `data/` to the repo, OR implement initialization-on-first-run in the app.

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, or `local` |
| `ANTHROPIC_API_KEY` | — | Optional: Claude API key |
| `OPENAI_API_KEY` | — | Optional: OpenAI API key |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | Embedding backend |
| `HF_TOKEN` | — | Optional: HF token for image generation |

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## 📋 Licence

MIT Licence — see [LICENSE](LICENSE)

This project uses:
- [ChromaDB](https://github.com/chroma-core/chroma) — Apache 2.0
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) — Apache 2.0
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — Apache 2.0
- [Flan-T5](https://huggingface.co/google/flan-t5-base) — Apache 2.0
- [Streamlit](https://streamlit.io) — Apache 2.0

## 🏛️ Academic Context

- **College**: MIT ADT University
- **Course**: Generative AI
- **Project**: P53 — Wikipedia-Grounded Answer Bot
- **Deadline**: 7 November 2026
