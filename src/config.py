"""Central configuration for the RAG system."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_chunks"
CHROMA_DB_DIR = ROOT_DIR / "chroma_db"

# ---------------------------------------------------------------------------
# Wikipedia topics (Phase 1 — 10 AI/ML articles)
# ---------------------------------------------------------------------------
TOPICS = [
    "Artificial intelligence",
    "Machine learning",
    "Deep learning",
    "Neural network",
    "Supervised learning",
    "Unsupervised learning",
    "Reinforcement learning",
    "Data science",
    "Big data",
    "Statistics",
]

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 600    # tokens per chunk
CHUNK_OVERLAP = 75  # overlapping tokens between adjacent chunks
TIKTOKEN_ENCODING = "cl100k_base"

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Set EMBEDDING_PROVIDER env var to "openai" to use OpenAI instead
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers")
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, local
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# ---------------------------------------------------------------------------
# Vector database (ChromaDB)
# ---------------------------------------------------------------------------
COLLECTION_NAME = "rag_wikipedia"
TOP_K = 5  # default number of chunks to retrieve

# ---------------------------------------------------------------------------
# LLM generation
# ---------------------------------------------------------------------------
# Set LLM_PROVIDER env var to "openai" to use OpenAI instead of Anthropic
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
OPENAI_CHAT_MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.1
MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# API keys (read from environment)
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# WikiTruth AI — Project Metadata & Course Details
# ---------------------------------------------------------------------------
APP_NAME = "WikiTruth AI"
APP_VERSION = "1.0.0"
APP_SUBTITLE = "Wikipedia-Grounded Answer Bot"
COLLEGE = "MIT ADT University"
COURSE = "Generative AI"

# ---------------------------------------------------------------------------
# Local LLM fallback (no API key needed)
# ---------------------------------------------------------------------------
LOCAL_LLM_MODEL = "google/flan-t5-base"
LOCAL_LLM_PROVIDER = "local"  # 'local' triggers transformers pipeline

# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------
HF_TOKEN = os.getenv("HF_TOKEN", "")  # optional, for HF inference API
IMAGE_GEN_MODEL = "stabilityai/stable-diffusion-2-1"
IMAGE_GEN_API_URL = f"https://api-inference.huggingface.co/models/{IMAGE_GEN_MODEL}"

# ---------------------------------------------------------------------------
# Retrieval thresholds
# ---------------------------------------------------------------------------
LOW_CONFIDENCE_THRESHOLD = 0.25  # below this → low confidence fallback
SIMILARITY_THRESHOLD = 0.20     # minimum similarity to include a chunk

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

