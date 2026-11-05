"""
app.py — WikiTruth AI
A six-page Streamlit application for the Wikipedia-Grounded Answer Bot.

Pages:
  1. Home               — Project intro, RAG architecture, tech stack
  2. Ask WikiTruth      — Question answering with grounding + images
  3. Grounded vs Ungrounded — Side-by-side comparison
  4. Evaluation Dashboard   — 20-question evaluation results with charts
  5. Prompt Engineering Lab — Zero-shot vs few-shot comparison
  6. About Project          — Architecture, units, responsible AI

P53 College Mini-Project | MIT ADT University | Generative AI Course
"""

import io
import json
import logging
import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to path
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Page config — MUST be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="WikiTruth AI — Wikipedia-Grounded Answer Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 50%, #0a0a1a 100%);
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0d1b2a 100%);
        border-right: 1px solid #1e3a5f;
    }

    /* Cards */
    .wiki-card {
        background: linear-gradient(135deg, #111827 0%, #1a2744 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.75rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    /* Answer box */
    .answer-box {
        background: linear-gradient(135deg, #0d2137 0%, #132a44 100%);
        border: 1px solid #4fc3f7;
        border-left: 4px solid #4fc3f7;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }

    /* Grounded answer */
    .grounded-answer {
        background: linear-gradient(135deg, #0d2137 0%, #132a44 100%);
        border: 1px solid #4caf50;
        border-left: 4px solid #4caf50;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }

    /* Ungrounded answer */
    .ungrounded-answer {
        background: linear-gradient(135deg, #2d1b1b 0%, #3d2020 100%);
        border: 1px solid #ef5350;
        border-left: 4px solid #ef5350;
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a2744 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4fc3f7;
    }

    .metric-label {
        font-size: 0.75rem;
        color: #90caf9;
        margin-top: 0.25rem;
    }

    /* Status badges */
    .badge-grounded {
        background: #1b5e20;
        color: #a5d6a7;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-unsupported {
        background: #b71c1c;
        color: #ef9a9a;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-low-confidence {
        background: #263238;
        color: #90a4ae;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        color: #4fc3f7;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #1e3a5f;
    }

    /* Source link */
    .source-link {
        background: #0d1b2a;
        border: 1px solid #1e3a5f;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        margin: 0.2rem 0;
        font-size: 0.85rem;
    }

    /* Similarity bar */
    .sim-bar-container {
        background: #1a2744;
        border-radius: 4px;
        height: 6px;
        margin-top: 4px;
    }
    .sim-bar-fill {
        background: linear-gradient(90deg, #4fc3f7, #81d4fa);
        border-radius: 4px;
        height: 6px;
    }

    /* Warning box */
    .warning-box {
        background: rgba(239, 83, 80, 0.1);
        border: 1px solid #ef5350;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Info box */
    .info-box {
        background: rgba(79, 195, 247, 0.1);
        border: 1px solid #4fc3f7;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Header gradient */
    .hero-title {
        background: linear-gradient(135deg, #4fc3f7 0%, #81d4fa 50%, #b3e5fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1.2;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(21,101,192,0.4);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1976d2 0%, #1e88e5 100%);
        box-shadow: 0 4px 12px rgba(21,101,192,0.6);
        transform: translateY(-1px);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 8px;
    }

    /* Progress bar */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #4fc3f7, #81d4fa);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        background: #111827;
        border-radius: 8px 8px 0 0;
        color: #90caf9;
    }

    .stTabs [aria-selected="true"] {
        background: #1565c0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading WikiTruth AI pipeline…")
def get_pipeline():
    """Load and cache the RAG pipeline."""
    from pipeline import RAGPipeline
    return RAGPipeline()


@st.cache_resource(show_spinner="Loading vector store…")
def get_vector_store():
    from src.vector_store import get_store
    return get_store()


def is_db_populated() -> bool:
    """Check if the vector store has been populated."""
    try:
        store = get_vector_store()
        return store.is_populated()
    except Exception:
        return False


def get_provider_badge(provider: str) -> str:
    badges = {
        "anthropic": "🟣 Claude",
        "openai": "🟢 GPT-3.5",
        "local": "🔵 Flan-T5 (Local)",
    }
    return badges.get(provider, f"❓ {provider}")


def render_similarity_bar(score: float) -> str:
    pct = int(score * 100)
    color = "#4caf50" if pct > 60 else "#ff9800" if pct > 35 else "#ef5350"
    return f"""
    <div style="display:flex; align-items:center; gap:8px;">
        <div style="flex:1; background:#1a2744; border-radius:4px; height:8px; overflow:hidden;">
            <div style="width:{pct}%; background:{color}; height:100%; border-radius:4px;"></div>
        </div>
        <span style="color:{color}; font-size:0.75rem; font-weight:600; min-width:40px;">{score:.3f}</span>
    </div>
    """


def load_eval_results() -> dict | None:
    """Load pre-generated evaluation results if available."""
    result_file = _ROOT / "evaluation" / "results" / "evaluation_latest.json"
    if result_file.exists():
        with open(result_file, encoding="utf-8") as f:
            return json.load(f)
    return None


def render_db_warning():
    """Show warning if vector DB is not populated."""
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Vector database not populated.</strong>
        Run <code>python main.py ingest</code> to download and index Wikipedia articles,
        then refresh this page.
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2rem;">📚</div>
            <div style="font-size:1.1rem; font-weight:700; color:#4fc3f7;">WikiTruth AI</div>
            <div style="font-size:0.7rem; color:#90caf9; margin-top:0.25rem;">
                Wikipedia-Grounded Answer Bot
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        pages = {
            "🏠 Home": "Home",
            "🔍 Ask WikiTruth": "Ask WikiTruth",
            "⚖️ Grounded vs Ungrounded": "Grounded vs Ungrounded",
            "📊 Evaluation Dashboard": "Evaluation Dashboard",
            "🧪 Prompt Engineering Lab": "Prompt Engineering Lab",
            "ℹ️ About Project": "About Project",
        }

        selected = st.radio(
            "Navigate",
            list(pages.keys()),
            label_visibility="collapsed",
        )

        st.markdown("---")

        # DB status
        populated = is_db_populated()
        if populated:
            try:
                store = get_vector_store()
                count = store.count()
                st.markdown(f"""
                <div style="font-size:0.75rem; color:#4caf50;">
                    ✅ Vector DB: {count} chunks indexed
                </div>
                """, unsafe_allow_html=True)
            except Exception:
                st.markdown('<div style="font-size:0.75rem; color:#4caf50;">✅ Vector DB ready</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="font-size:0.75rem; color:#ef5350;">
                ❌ Vector DB empty<br>
                <code>python main.py ingest</code>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.65rem; color:#546e7a; text-align:center;">
            P53 — MIT ADT University<br>
            Generative AI Course<br>
            November 2026
        </div>
        """, unsafe_allow_html=True)

    return pages[selected]


# ===========================================================================
# PAGE 1: HOME
# ===========================================================================

def page_home():
    # Hero section
    st.markdown("""
    <div style="text-align:center; padding:2rem 0 1rem 0;">
        <div class="hero-title">WikiTruth AI</div>
        <div style="font-size:1.2rem; color:#90caf9; margin-top:0.5rem;">
            Wikipedia-Grounded Answer Bot
        </div>
        <div style="font-size:0.85rem; color:#546e7a; margin-top:0.5rem;">
            P53 College Mini-Project · MIT ADT University · Generative AI Course
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">10</div>
            <div class="metric-label">Wikipedia Articles</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">384</div>
            <div class="metric-label">Embedding Dimensions</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">20</div>
            <div class="metric-label">Evaluation Questions</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">0</div>
            <div class="metric-label">API Keys Required</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:0.75rem;">
                🎯 Project Objective
            </div>
            <p style="color:#e2e8f0; line-height:1.6;">
                WikiTruth AI demonstrates that Large Language Models can be constrained to answer
                questions <strong>exclusively from verified Wikipedia sources</strong>, eliminating
                hallucinations by design. The system retrieves the most relevant Wikipedia passages,
                validates retrieval confidence, and generates answers with strict source attribution.
            </p>
            <p style="color:#90caf9; font-size:0.85rem; line-height:1.6;">
                If Wikipedia doesn't have the answer, the bot explicitly says so —
                it never invents facts.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:0.75rem;">
                ⚙️ How It Works
            </div>
            <div style="color:#e2e8f0; line-height:1.8; font-size:0.9rem;">
                <span style="color:#4fc3f7;">①</span> <strong>Ingest</strong> — 10 Wikipedia AI/ML articles are downloaded, chunked (600 tokens, 75-token overlap), and embedded using Sentence-BERT (<code>all-MiniLM-L6-v2</code>)<br><br>
                <span style="color:#4fc3f7;">②</span> <strong>Retrieve</strong> — Your question is embedded and compared against all chunks using cosine similarity. Top-K most relevant passages are retrieved from ChromaDB<br><br>
                <span style="color:#4fc3f7;">③</span> <strong>Validate</strong> — Retrieval confidence is checked. Low similarity scores trigger a fallback message<br><br>
                <span style="color:#4fc3f7;">④</span> <strong>Generate</strong> — A grounding prompt (zero-shot or few-shot) is used to instruct the LLM to answer only from retrieved context<br><br>
                <span style="color:#4fc3f7;">⑤</span> <strong>Cite</strong> — Sources, similarity scores, and Wikipedia links are displayed alongside the answer
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:1rem;">
                🏗️ RAG Architecture
            </div>
            <div style="font-family:monospace; font-size:0.75rem; color:#90caf9; line-height:2;">
                <div style="color:#e2e8f0;">📝 User Question</div>
                <div style="color:#546e7a;">│</div>
                <div style="color:#e2e8f0;">🔄 Question Processing</div>
                <div style="color:#546e7a;">│</div>
                <div style="color:#4fc3f7;">🧮 Sentence-BERT Embedding</div>
                <div style="color:#546e7a;">│  all-MiniLM-L6-v2 → 384-dim</div>
                <div style="color:#4fc3f7;">🗄️ ChromaDB Vector Search</div>
                <div style="color:#546e7a;">│  cosine similarity</div>
                <div style="color:#e2e8f0;">📄 Top-K Passage Retrieval</div>
                <div style="color:#546e7a;">│</div>
                <div style="color:#ff9800;">✅ Context Validation</div>
                <div style="color:#546e7a;">│  similarity threshold: 0.25</div>
                <div style="color:#ff9800;">📋 Grounding Prompt</div>
                <div style="color:#546e7a;">│  zero-shot / few-shot</div>
                <div style="color:#4caf50;">🤖 Transformer LLM</div>
                <div style="color:#546e7a;">│  Flan-T5 / Claude / GPT</div>
                <div style="color:#e2e8f0;">📊 Answer + Sources + Image</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:0.75rem;">
                🛠️ Technology Stack
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.5rem; font-size:0.8rem;">
                <div style="color:#90caf9;"><strong>Python</strong> 3.10+</div>
                <div style="color:#90caf9;"><strong>Streamlit</strong> 1.28+</div>
                <div style="color:#90caf9;"><strong>ChromaDB</strong> 0.4+</div>
                <div style="color:#90caf9;"><strong>Sentence-BERT</strong></div>
                <div style="color:#90caf9;"><strong>PyTorch</strong> 2.0+</div>
                <div style="color:#90caf9;"><strong>Transformers</strong></div>
                <div style="color:#90caf9;"><strong>Plotly</strong></div>
                <div style="color:#90caf9;"><strong>Flan-T5</strong> local</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Topics covered
    st.markdown("---")
    st.markdown("""
    <div style="color:#4fc3f7; font-size:1rem; font-weight:600; margin-bottom:0.75rem;">
        📚 Wikipedia Articles Indexed
    </div>
    """, unsafe_allow_html=True)

    topics = [
        ("🤖", "Artificial Intelligence"),
        ("📈", "Machine Learning"),
        ("🧠", "Deep Learning"),
        ("🕸️", "Neural Networks"),
        ("🎯", "Supervised Learning"),
        ("🔍", "Unsupervised Learning"),
        ("🏆", "Reinforcement Learning"),
        ("📊", "Data Science"),
        ("💾", "Big Data"),
        ("📐", "Statistics"),
    ]

    cols = st.columns(5)
    for i, (icon, topic) in enumerate(topics):
        with cols[i % 5]:
            st.markdown(f"""
            <div style="background:#111827; border:1px solid #1e3a5f; border-radius:8px;
                        padding:0.75rem; text-align:center; margin:0.25rem 0;">
                <div style="font-size:1.5rem;">{icon}</div>
                <div style="font-size:0.75rem; color:#90caf9; margin-top:0.25rem;">{topic}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Start Asking Questions →", key="home_start"):
        st.info("Use the sidebar to navigate to '🔍 Ask WikiTruth'")


# ===========================================================================
# PAGE 2: ASK WIKITRUTH
# ===========================================================================

def page_ask_wikitruth():
    st.markdown("""
    <div class="hero-title" style="font-size:1.8rem; margin-bottom:0.5rem;">🔍 Ask WikiTruth</div>
    <div style="color:#90caf9; margin-bottom:1.5rem; font-size:0.9rem;">
        Ask any question about AI/ML — answered exclusively from Wikipedia
    </div>
    """, unsafe_allow_html=True)

    if not is_db_populated():
        render_db_warning()
        return

    # Sidebar options
    with st.sidebar:
        st.markdown("---")
        st.markdown("**⚙️ Query Options**")
        top_k = st.slider("Top-K passages", min_value=1, max_value=10, value=5, key="ask_top_k")
        prompt_style = st.radio("Prompt style", ["zero_shot", "few_shot"], key="ask_prompt_style",
                                format_func=lambda x: "🎯 Zero-Shot" if x == "zero_shot" else "📚 Few-Shot")
        generate_img = st.toggle("Generate illustration", value=True, key="ask_gen_img")

    # Question input
    example_questions = [
        "What is machine learning?",
        "How do neural networks learn from data?",
        "What is the difference between supervised and unsupervised learning?",
        "What are the main limitations of deep learning?",
        "How does reinforcement learning work?",
    ]

    col_q, col_ex = st.columns([3, 1])
    with col_q:
        question = st.text_input(
            "Your question",
            placeholder="e.g., What is deep learning?",
            key="ask_question",
            label_visibility="collapsed",
        )
    with col_ex:
        selected_example = st.selectbox(
            "Examples",
            [""] + example_questions,
            key="ask_example",
            label_visibility="collapsed",
        )
        if selected_example:
            question = selected_example

    ask_clicked = st.button("🔍 Ask WikiTruth", key="ask_button", use_container_width=True)

    if ask_clicked and question:
        pipeline = get_pipeline()

        with st.spinner("🔍 Retrieving Wikipedia passages…"):
            try:
                result = pipeline.query(question, top_k=top_k, prompt_style=prompt_style)
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
                return

        # Answer section
        answer = result.get("answer", "")
        chunks = result.get("context_chunks", [])
        grounding_status = result.get("grounding_status", "")
        model = result.get("model", result.get("provider", "unknown"))

        # Grounding status badge
        status_colors = {
            "grounded": ("#4caf50", "✅ Fully Grounded"),
            "partial": ("#ff9800", "⚠️ Partially Grounded"),
            "unsupported": ("#ef5350", "❌ Not Supported"),
            "low_confidence": ("#607d8b", "🔍 Low Confidence"),
        }
        status_color, status_label = status_colors.get(grounding_status, ("#607d8b", "❓ Unknown"))

        col_ans, col_meta = st.columns([2, 1])

        with col_ans:
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem;">
                <div style="color:#4fc3f7; font-weight:600;">📝 Answer</div>
                <span style="background:{status_color}22; color:{status_color}; border:1px solid {status_color}; 
                      padding:0.1rem 0.6rem; border-radius:20px; font-size:0.75rem; font-weight:600;">
                    {status_label}
                </span>
                <span style="color:#546e7a; font-size:0.75rem;">via {get_provider_badge(result.get('provider','?'))}</span>
            </div>
            """, unsafe_allow_html=True)

            # Unsupported warning
            if grounding_status == "unsupported":
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>WikiTruth cannot answer this question from available Wikipedia sources.</strong><br>
                    The retrieved passages do not contain sufficient information. This is by design —
                    the bot refuses to hallucinate.
                </div>
                """, unsafe_allow_html=True)
            elif grounding_status == "low_confidence":
                st.markdown("""
                <div class="warning-box">
                    🔍 <strong>Low retrieval confidence.</strong> The retrieved passages may not be
                    closely related to your question. Consider rephrasing.
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="answer-box">
                <div style="color:#e2e8f0; line-height:1.7;">{answer.replace(chr(10), '<br>')}</div>
            </div>
            """, unsafe_allow_html=True)

            # Prompt style info
            st.markdown(f"""
            <div style="font-size:0.75rem; color:#546e7a; margin-top:0.5rem;">
                Prompt style: <strong style="color:#90caf9;">{'Few-Shot' if prompt_style == 'few_shot' else 'Zero-Shot'}</strong>
                · Top-K: <strong style="color:#90caf9;">{top_k}</strong>
                · Chunks retrieved: <strong style="color:#90caf9;">{len(chunks)}</strong>
            </div>
            """, unsafe_allow_html=True)

        with col_meta:
            # Image generation
            if generate_img and chunks:
                topic = chunks[0].get("source", question) if chunks else question
                with st.spinner(f"🎨 Generating illustration for '{topic}'…"):
                    try:
                        from src.image_generator import generate_image
                        img_bytes = generate_image(topic)
                        if img_bytes:
                            st.image(img_bytes, caption=f"📸 {topic}", use_container_width=True)
                        else:
                            st.info("🖼️ Image generation unavailable")
                    except Exception as e:
                        st.warning(f"Image generation failed: {e}")

            # Quick stats
            if chunks:
                avg_sim = sum(c.get("similarity", 0) for c in chunks) / len(chunks)
                st.markdown(f"""
                <div class="metric-card" style="margin-top:0.5rem;">
                    <div style="font-size:1.5rem; color:#4fc3f7; font-weight:700;">{avg_sim:.3f}</div>
                    <div style="font-size:0.7rem; color:#90caf9;">Avg Retrieval Similarity</div>
                </div>
                """, unsafe_allow_html=True)

        # Retrieved passages
        if chunks:
            st.markdown("---")
            st.markdown("""
            <div class="section-header">📄 Retrieved Wikipedia Passages</div>
            """, unsafe_allow_html=True)

            for i, chunk in enumerate(chunks, 1):
                sim = chunk.get("similarity", 0)
                source = chunk.get("source", "Unknown")
                url = chunk.get("url", f"https://en.wikipedia.org/wiki/{source.replace(' ', '_')}")
                text = chunk.get("text", "")

                with st.expander(f"[{i}] {source} — similarity: {sim:.3f}", expanded=(i == 1)):
                    st.markdown(render_similarity_bar(sim), unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style="color:#e2e8f0; font-size:0.85rem; line-height:1.6; 
                                margin:0.75rem 0; padding:0.75rem; background:#0d1b2a; border-radius:6px;">
                        {text}
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"🔗 [Open on Wikipedia]({url})")

            # Source links
            st.markdown("""
            <div class="section-header">🔗 Wikipedia Source Links</div>
            """, unsafe_allow_html=True)

            try:
                from src.wikipedia_retriever import get_source_links
                sources = get_source_links(chunks)
                for src in sources:
                    st.markdown(f"""
                    <div class="source-link">
                        📖 <a href="{src['url']}" target="_blank" style="color:#4fc3f7; text-decoration:none;">
                            {src['title']}
                        </a>
                        <span style="color:#546e7a; font-size:0.75rem;"> — {src['chunk_count']} passage(s)</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not load source links: {e}")

    elif ask_clicked and not question:
        st.warning("Please enter a question.")


# ===========================================================================
# PAGE 3: GROUNDED vs UNGROUNDED
# ===========================================================================

def page_grounded_vs_ungrounded():
    st.markdown("""
    <div class="hero-title" style="font-size:1.8rem; margin-bottom:0.5rem;">⚖️ Grounded vs Ungrounded</div>
    <div style="color:#90caf9; margin-bottom:1.5rem; font-size:0.9rem;">
        Compare Wikipedia-grounded answers against raw LLM responses to see hallucination in action
    </div>
    """, unsafe_allow_html=True)

    if not is_db_populated():
        render_db_warning()

    # Explanation cards
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div style="background:#0d2137; border:1px solid #4caf50; border-top:3px solid #4caf50;
                    border-radius:8px; padding:1rem; height:100%;">
            <div style="color:#4caf50; font-weight:600; margin-bottom:0.5rem;">✅ Mode B: Grounded (RAG)</div>
            <div style="color:#90caf9; font-size:0.8rem; line-height:1.6;">
                Retrieves relevant Wikipedia passages → Validates context → 
                Generates answer using ONLY retrieved evidence → Cites sources →
                Refuses if context is insufficient
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style="background:#2d1b1b; border:1px solid #ef5350; border-top:3px solid #ef5350;
                    border-radius:8px; padding:1rem; height:100%;">
            <div style="color:#ef5350; font-weight:600; margin-bottom:0.5rem;">❌ Mode A: Ungrounded (Raw LLM)</div>
            <div style="color:#90caf9; font-size:0.8rem; line-height:1.6;">
                Sends question directly to LLM → No retrieval, no context → 
                LLM uses pre-trained knowledge → May hallucinate facts, dates, 
                statistics → No source attribution
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Question input
    example_questions = [
        "What is machine learning?",
        "Who invented artificial intelligence?",
        "What are the applications of deep learning?",
        "What is the bias-variance tradeoff?",
        "How does backpropagation work?",
    ]

    col_q, col_btn = st.columns([4, 1])
    with col_q:
        question = st.text_input(
            "Enter a question to compare",
            placeholder="e.g., Who invented artificial intelligence?",
            key="compare_question",
            label_visibility="collapsed",
        )
    with col_btn:
        selected = st.selectbox("Examples", [""] + example_questions,
                                key="compare_example", label_visibility="collapsed")
        if selected:
            question = selected

    top_k = st.slider("Top-K for grounded retrieval", 1, 10, 5, key="compare_top_k")
    compare_clicked = st.button("⚖️ Compare Both Answers", key="compare_button", use_container_width=True)

    if compare_clicked and question:
        pipeline = get_pipeline()

        col_grounded, col_ungrounded = st.columns(2)

        with col_grounded:
            st.markdown("""
            <div style="color:#4caf50; font-weight:600; font-size:1rem; margin-bottom:0.5rem;">
                ✅ Grounded Answer (Mode B)
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Retrieving Wikipedia passages…"):
                try:
                    grounded_result = pipeline.query(question, top_k=top_k)
                    g_answer = grounded_result.get("answer", "")
                    g_chunks = grounded_result.get("context_chunks", [])
                    g_status = grounded_result.get("grounding_status", "")
                    g_model = grounded_result.get("model", "?")

                    st.markdown(f"""
                    <div class="grounded-answer">
                        <div style="color:#e2e8f0; line-height:1.7;">{g_answer.replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if g_chunks:
                        avg_sim = sum(c.get("similarity", 0) for c in g_chunks) / len(g_chunks)
                        st.markdown(f"""
                        <div style="font-size:0.75rem; color:#546e7a; margin-top:0.5rem;">
                            Model: {get_provider_badge(grounded_result.get('provider','?'))} 
                            · Avg sim: {avg_sim:.3f} · {len(g_chunks)} passages
                        </div>
                        """, unsafe_allow_html=True)

                    # Sources
                    if g_chunks:
                        st.markdown("**Sources:**")
                        from src.wikipedia_retriever import get_source_links
                        for src in get_source_links(g_chunks):
                            st.markdown(f"- 📖 [{src['title']}]({src['url']})")

                    # Retrieved passages
                    with st.expander("📄 View retrieved passages"):
                        for i, chunk in enumerate(g_chunks, 1):
                            sim = chunk.get("similarity", 0)
                            st.markdown(f"**[{i}] {chunk['source']}** (sim: {sim:.3f})")
                            st.markdown(render_similarity_bar(sim), unsafe_allow_html=True)
                            st.caption(chunk.get("text", "")[:300] + "…")

                except Exception as e:
                    st.error(f"Grounded query failed: {e}")

        with col_ungrounded:
            st.markdown("""
            <div style="color:#ef5350; font-weight:600; font-size:1rem; margin-bottom:0.5rem;">
                ❌ Ungrounded Answer (Mode A)
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Querying LLM without retrieval…"):
                try:
                    ungrounded_result = pipeline.query_ungrounded(question)
                    u_answer = ungrounded_result.get("answer", "")

                    st.markdown(f"""
                    <div class="ungrounded-answer">
                        <div style="color:#e2e8f0; line-height:1.7;">{u_answer.replace(chr(10), '<br>')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="font-size:0.75rem; color:#546e7a; margin-top:0.5rem;">
                        Model: {get_provider_badge(ungrounded_result.get('provider','?'))} 
                        · No retrieval · No source attribution
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("""
                    <div class="warning-box" style="margin-top:0.75rem;">
                        ⚠️ <strong>No source attribution.</strong> Claims in this answer are
                        from the LLM's training data and may contain hallucinations.
                    </div>
                    """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Ungrounded query failed: {e}")

    elif compare_clicked and not question:
        st.warning("Please enter a question.")


# ===========================================================================
# PAGE 4: EVALUATION DASHBOARD
# ===========================================================================

def page_evaluation_dashboard():
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd

    st.markdown("""
    <div class="hero-title" style="font-size:1.8rem; margin-bottom:0.5rem;">📊 Evaluation Dashboard</div>
    <div style="color:#90caf9; margin-bottom:1.5rem; font-size:0.9rem;">
        20-question dual-track evaluation: Grounded (Mode B) vs Ungrounded (Mode A)
    </div>
    """, unsafe_allow_html=True)

    # Load pre-generated results or run evaluation
    eval_data = load_eval_results()

    if eval_data is None:
        st.markdown("""
        <div class="info-box">
            ℹ️ <strong>No evaluation results found.</strong><br>
            Run the evaluation to generate results:
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Run Full Evaluation (Grounded + Ungrounded)", key="run_eval"):
                if not is_db_populated():
                    st.error("Vector DB not populated. Run `python main.py ingest` first.")
                else:
                    with st.spinner("Running 20-question dual-track evaluation… (this may take several minutes)"):
                        try:
                            from evaluation.evaluate import run_full_dual_evaluation
                            results = run_full_dual_evaluation(
                                auto_score=True,
                                output_dir="evaluation/results",
                            )
                            st.success("✅ Evaluation complete! Reload the page to see results.")
                            eval_data = load_eval_results()
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
                            return
        with col2:
            if st.button("▶️ Run Grounded Only", key="run_grounded_eval"):
                if not is_db_populated():
                    st.error("Vector DB not populated. Run `python main.py ingest` first.")
                else:
                    with st.spinner("Running grounded evaluation…"):
                        try:
                            from evaluation.evaluate import run_evaluation
                            run_evaluation(
                                auto_score=True,
                                output_file="evaluation/results/grounded_results.json",
                            )
                            st.success("✅ Grounded evaluation complete!")
                        except Exception as e:
                            st.error(f"Evaluation failed: {e}")
        return

    g_metrics = eval_data.get("grounded_metrics", {})
    u_metrics = eval_data.get("ungrounded_metrics", {})
    combined = eval_data.get("combined", [])

    # Top metrics
    st.markdown("""
    <div class="section-header">📈 Key Metrics</div>
    """, unsafe_allow_html=True)

    cols = st.columns(6)
    metrics_to_show = [
        ("Total Questions", g_metrics.get("total_questions", 20), "#4fc3f7"),
        ("Grounded Accuracy", f"{g_metrics.get('overall_accuracy_pct', 0):.1f}%", "#4caf50"),
        ("Ungrounded Accuracy", f"{u_metrics.get('overall_accuracy_pct', 0):.1f}%", "#ef5350"),
        ("Grounded Error Rate", f"{g_metrics.get('error_rate_pct', 0):.1f}%", "#ff9800"),
        ("Grounded Hallucination", f"{g_metrics.get('hallucination_rate_pct', 0):.1f}%", "#ab47bc"),
        ("Avg Retrieval Sim", f"{g_metrics.get('avg_retrieval_similarity', 0):.3f}", "#29b6f6"),
    ]

    for i, (label, value, color) in enumerate(metrics_to_show):
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.5rem; font-weight:700; color:{color};">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row 1
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("""<div class="section-header">📊 Accuracy Comparison</div>""", unsafe_allow_html=True)

        fig_acc = go.Figure()
        categories = ["Correct (2/2)", "Partial (1/2)", "Wrong (0/2)"]
        g_values = [
            g_metrics.get("correct_2", 0),
            g_metrics.get("partial_1", 0),
            g_metrics.get("wrong_0", 0),
        ]
        u_values = [
            u_metrics.get("correct_2", 0),
            u_metrics.get("partial_1", 0),
            u_metrics.get("wrong_0", 0),
        ]

        fig_acc.add_trace(go.Bar(
            name="Grounded (Mode B)",
            x=categories, y=g_values,
            marker_color="#4caf50",
            text=g_values, textposition="outside",
        ))
        fig_acc.add_trace(go.Bar(
            name="Ungrounded (Mode A)",
            x=categories, y=u_values,
            marker_color="#ef5350",
            text=u_values, textposition="outside",
        ))

        fig_acc.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,42,0.5)",
            font=dict(color="#e2e8f0", size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(gridcolor="#1e3a5f", title="Questions"),
            xaxis=dict(gridcolor="#1e3a5f"),
            height=300,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_chart2:
        st.markdown("""<div class="section-header">📉 Error & Hallucination Rates</div>""", unsafe_allow_html=True)

        metrics_comparison = {
            "Error Rate (%)": [g_metrics.get("error_rate_pct", 0), u_metrics.get("error_rate_pct", 0)],
            "Hallucination (%)": [g_metrics.get("hallucination_rate_pct", 0), u_metrics.get("hallucination_rate_pct", 0)],
            "Partial Rate (%)": [g_metrics.get("partial_rate_pct", 0), u_metrics.get("partial_rate_pct", 0)],
        }

        fig_err = go.Figure()
        x_labels = list(metrics_comparison.keys())
        g_errs = [v[0] for v in metrics_comparison.values()]
        u_errs = [v[1] for v in metrics_comparison.values()]

        fig_err.add_trace(go.Bar(name="Grounded", x=x_labels, y=g_errs, marker_color="#4caf50",
                                 text=[f"{v:.1f}%" for v in g_errs], textposition="outside"))
        fig_err.add_trace(go.Bar(name="Ungrounded", x=x_labels, y=u_errs, marker_color="#ef5350",
                                 text=[f"{v:.1f}%" for v in u_errs], textposition="outside"))

        fig_err.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,42,0.5)",
            font=dict(color="#e2e8f0", size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(gridcolor="#1e3a5f", title="Percentage (%)", range=[0, 100]),
            height=300,
        )
        st.plotly_chart(fig_err, use_container_width=True)

    # By difficulty chart
    st.markdown("""<div class="section-header">📚 Performance by Difficulty Level</div>""", unsafe_allow_html=True)

    g_by_diff = g_metrics.get("by_difficulty", {})
    u_by_diff = u_metrics.get("by_difficulty", {})

    if g_by_diff:
        diff_labels = list(g_by_diff.keys())
        g_acc_by_diff = [g_by_diff[d]["accuracy_pct"] for d in diff_labels]
        u_acc_by_diff = [u_by_diff.get(d, {}).get("accuracy_pct", 0) for d in diff_labels]

        fig_diff = go.Figure()
        fig_diff.add_trace(go.Bar(name="Grounded", x=diff_labels, y=g_acc_by_diff,
                                  marker_color="#4caf50", text=[f"{v:.1f}%" for v in g_acc_by_diff],
                                  textposition="outside"))
        fig_diff.add_trace(go.Bar(name="Ungrounded", x=diff_labels, y=u_acc_by_diff,
                                  marker_color="#ef5350", text=[f"{v:.1f}%" for v in u_acc_by_diff],
                                  textposition="outside"))
        fig_diff.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,27,42,0.5)",
            font=dict(color="#e2e8f0", size=11),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20, r=20, t=20, b=40),
            yaxis=dict(gridcolor="#1e3a5f", title="Accuracy (%)", range=[0, 110]),
            height=320,
        )
        st.plotly_chart(fig_diff, use_container_width=True)

    # Detailed results table
    if combined:
        st.markdown("""<div class="section-header">📋 Detailed Question-Level Results</div>""", unsafe_allow_html=True)

        df = pd.DataFrame(combined)
        display_cols = ["id", "question", "difficulty", "grounded_score", "ungrounded_score", "avg_retrieval_similarity"]
        if all(c in df.columns for c in display_cols):
            df_display = df[display_cols].copy()
            df_display.columns = ["ID", "Question", "Difficulty", "Grounded Score", "Ungrounded Score", "Avg Sim"]
            df_display["Question"] = df_display["Question"].apply(lambda x: x[:60] + "…" if len(x) > 60 else x)

            # Color score cells
            def color_score(val):
                if val == 2:
                    return "background-color: #1b5e2033; color: #4caf50;"
                elif val == 1:
                    return "background-color: #f57f1733; color: #ff9800;"
                else:
                    return "background-color: #b71c1c33; color: #ef5350;"

            st.dataframe(
                df_display.style.applymap(color_score, subset=["Grounded Score", "Ungrounded Score"]),
                use_container_width=True,
                height=500,
            )

    # Download buttons
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)
    csv_path = Path("evaluation/results/comparison_results.csv")
    json_path = Path("evaluation/results/evaluation_latest.json")

    with col_dl1:
        if csv_path.exists():
            with open(csv_path, "rb") as f:
                st.download_button("⬇️ Download CSV Results", f, "evaluation_results.csv", "text/csv")

    with col_dl2:
        if json_path.exists():
            with open(json_path, "rb") as f:
                st.download_button("⬇️ Download JSON Results", f, "evaluation_results.json", "application/json")


# ===========================================================================
# PAGE 5: PROMPT ENGINEERING LAB
# ===========================================================================

def page_prompt_engineering_lab():
    st.markdown("""
    <div class="hero-title" style="font-size:1.8rem; margin-bottom:0.5rem;">🧪 Prompt Engineering Lab</div>
    <div style="color:#90caf9; margin-bottom:1.5rem; font-size:0.9rem;">
        Compare zero-shot vs few-shot grounding prompts — U1 syllabus unit
    </div>
    """, unsafe_allow_html=True)

    # Educational section
    tab_lab, tab_edu, tab_viz = st.tabs(["🔬 Compare Prompts", "📚 Education", "📈 Visualization"])

    with tab_lab:
        if not is_db_populated():
            render_db_warning()

        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-weight:600; margin-bottom:0.75rem;">ℹ️ Prompt Techniques Explained</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; font-size:0.85rem;">
                <div>
                    <div style="color:#ff9800; font-weight:600;">🎯 Zero-Shot Prompting</div>
                    <div style="color:#90caf9; margin-top:0.25rem;">Instructs the LLM with rules and constraints only. No examples provided. The LLM must follow instructions without demonstrated patterns.</div>
                </div>
                <div>
                    <div style="color:#4caf50; font-weight:600;">📚 Few-Shot Prompting</div>
                    <div style="color:#90caf9; margin-top:0.25rem;">Provides 2-3 examples of supported and unsupported Q&A pairs. The LLM learns the expected behavior pattern from examples.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        example_q = [
            "What is machine learning?",
            "How do neural networks learn?",
            "Who created the internet?",  # Out of scope
            "What is the difference between supervised and unsupervised learning?",
        ]

        col_q, col_k = st.columns([3, 1])
        with col_q:
            question = st.text_input("Question for prompt comparison", placeholder="Enter your question…",
                                     key="lab_question", label_visibility="collapsed")
        with col_k:
            top_k = st.slider("Top-K", 1, 8, 5, key="lab_top_k")

        selected_q = st.selectbox("Example questions", [""] + example_q, key="lab_example",
                                  label_visibility="collapsed")
        if selected_q:
            question = selected_q

        compare_clicked = st.button("🔬 Compare Prompt Techniques", key="lab_compare", use_container_width=True)

        if compare_clicked and question:
            pipeline = get_pipeline()

            with st.spinner("Running zero-shot and few-shot comparisons…"):
                try:
                    comparison = pipeline.compare_prompt_styles(question, top_k=top_k)
                    zero = comparison["zero_shot"]
                    few = comparison["few_shot"]
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
                    return

            col_z, col_f = st.columns(2)

            def auto_score(answer, keywords=None):
                """Simple scoring for display."""
                ans_lower = answer.lower()
                if any(p in ans_lower for p in ["i cannot answer", "i don't know", "i do not know"]):
                    return 0, "✗ Refused"
                if keywords:
                    hits = sum(1 for k in keywords if k.lower() in ans_lower)
                    if hits / len(keywords) >= 0.5:
                        return 2, "✓ Likely Correct"
                    elif hits / len(keywords) >= 0.25:
                        return 1, "~ Partial"
                return 1, "⚠️ Unverified"

            with col_z:
                st.markdown("""
                <div style="color:#ff9800; font-weight:600; font-size:1rem; margin-bottom:0.5rem;">
                    🎯 Zero-Shot Grounding Prompt
                </div>
                """, unsafe_allow_html=True)

                z_answer = zero.get("answer", "")
                z_chunks = zero.get("context_chunks", [])
                z_score, z_label = auto_score(z_answer)
                z_status = zero.get("grounding_status", "")

                st.markdown(f"""
                <div class="answer-box">
                    <div style="color:#e2e8f0; line-height:1.7; font-size:0.9rem;">{z_answer.replace(chr(10), '<br>')}</div>
                </div>
                <div style="font-size:0.75rem; color:#546e7a; margin-top:0.4rem;">
                    Status: <strong style="color:#90caf9;">{z_status}</strong>
                    · Score: <strong style="color:#ff9800;">{z_label}</strong>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📋 View Zero-Shot Prompt"):
                    try:
                        from src.grounding import build_prompt
                        _, zp = build_prompt(question, z_chunks[:2], style="zero_shot")
                        st.code(zp[:1500] + "…" if len(zp) > 1500 else zp, language="text")
                    except Exception as e:
                        st.error(str(e))

            with col_f:
                st.markdown("""
                <div style="color:#4caf50; font-weight:600; font-size:1rem; margin-bottom:0.5rem;">
                    📚 Few-Shot Grounding Prompt
                </div>
                """, unsafe_allow_html=True)

                f_answer = few.get("answer", "")
                f_chunks = few.get("context_chunks", [])
                f_score, f_label = auto_score(f_answer)
                f_status = few.get("grounding_status", "")

                st.markdown(f"""
                <div class="answer-box">
                    <div style="color:#e2e8f0; line-height:1.7; font-size:0.9rem;">{f_answer.replace(chr(10), '<br>')}</div>
                </div>
                <div style="font-size:0.75rem; color:#546e7a; margin-top:0.4rem;">
                    Status: <strong style="color:#90caf9;">{f_status}</strong>
                    · Score: <strong style="color:#4caf50;">{f_label}</strong>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("📋 View Few-Shot Prompt"):
                    try:
                        from src.grounding import build_prompt
                        _, fp = build_prompt(question, f_chunks[:2], style="few_shot")
                        st.code(fp[:1500] + "…" if len(fp) > 1500 else fp, language="text")
                    except Exception as e:
                        st.error(str(e))

            # Comparison table
            st.markdown("---")
            st.markdown("""<div class="section-header">📊 Prompt Comparison Table</div>""", unsafe_allow_html=True)

            import pandas as pd
            comp_df = pd.DataFrame([{
                "Technique": "Zero-Shot",
                "Prompt Size": "Shorter",
                "Examples": "None",
                "Grounding Status": z_status,
                "Answer Score": z_label,
                "Hallucination Risk": "Medium",
            }, {
                "Technique": "Few-Shot",
                "Prompt Size": "Longer (3 examples)",
                "Examples": "3 (supported, unsupported, partial)",
                "Grounding Status": f_status,
                "Answer Score": f_label,
                "Hallucination Risk": "Lower",
            }])
            st.dataframe(comp_df, use_container_width=True)

        elif compare_clicked and not question:
            st.warning("Please enter a question.")

    with tab_edu:
        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:1rem;">
                🤖 Generative AI vs Discriminative AI
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; font-size:0.85rem;">
                <div>
                    <div style="color:#ff9800; font-weight:600; margin-bottom:0.5rem;">Generative AI</div>
                    <div style="color:#e2e8f0; line-height:1.7;">
                        • Learns the joint distribution P(X, Y)<br>
                        • Can generate new data samples<br>
                        • Examples: GPT, DALL-E, Flan-T5<br>
                        • Used in: text generation, image synthesis<br>
                        • Higher model complexity<br>
                        • Can handle missing data
                    </div>
                </div>
                <div>
                    <div style="color:#4caf50; font-weight:600; margin-bottom:0.5rem;">Discriminative AI</div>
                    <div style="color:#e2e8f0; line-height:1.7;">
                        • Learns the conditional distribution P(Y|X)<br>
                        • Makes predictions/classifications<br>
                        • Examples: BERT, SVM, Logistic Regression<br>
                        • Used in: classification, regression<br>
                        • Often more accurate for classification<br>
                        • Cannot generate new samples
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:1rem;">
                🔄 GenAI Lifecycle
            </div>
            <div style="font-family:monospace; font-size:0.8rem; color:#90caf9; line-height:2;">
                <div style="display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap;">
                    <span style="background:#1565c0; padding:0.3rem 0.8rem; border-radius:20px; color:white;">1. Data Collection</span>
                    <span style="color:#4fc3f7;">→</span>
                    <span style="background:#1565c0; padding:0.3rem 0.8rem; border-radius:20px; color:white;">2. Preprocessing</span>
                    <span style="color:#4fc3f7;">→</span>
                    <span style="background:#1565c0; padding:0.3rem 0.8rem; border-radius:20px; color:white;">3. Model Training</span>
                    <span style="color:#4fc3f7;">→</span>
                    <span style="background:#1565c0; padding:0.3rem 0.8rem; border-radius:20px; color:white;">4. Evaluation</span>
                    <span style="color:#4fc3f7;">→</span>
                    <span style="background:#1565c0; padding:0.3rem 0.8rem; border-radius:20px; color:white;">5. Deployment</span>
                    <span style="color:#4fc3f7;">→</span>
                    <span style="background:#1565c0; padding:0.3rem 0.8rem; border-radius:20px; color:white;">6. Monitoring</span>
                    <span style="color:#4fc3f7;">→ ↺</span>
                </div>
                <div style="margin-top:1rem; color:#546e7a; font-family:sans-serif; font-size:0.8rem;">
                    WikiTruth AI implements steps 1-5: Wikipedia data collection → chunking/preprocessing → 
                    embedding (Sentence-BERT) → RAG evaluation → Streamlit deployment.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:1rem;">
                🧠 Transformer Self-Attention Explained
            </div>
            <div style="color:#e2e8f0; font-size:0.85rem; line-height:1.8;">
                <strong style="color:#ff9800;">What is self-attention?</strong><br>
                Self-attention allows each token in a sequence to "attend" to all other tokens, 
                computing weighted relevance scores. For a sequence of tokens [w₁, w₂, ..., wₙ]:
            </div>
            <div style="font-family:monospace; background:#0d1b2a; padding:1rem; border-radius:6px; 
                        margin:0.75rem 0; font-size:0.8rem; color:#4fc3f7;">
                Q = X · W_Q  (Query matrix)<br>
                K = X · W_K  (Key matrix)<br>
                V = X · W_V  (Value matrix)<br><br>
                Attention(Q,K,V) = softmax(QK^T / √d_k) · V
            </div>
            <div style="color:#90caf9; font-size:0.8rem; line-height:1.7;">
                • <strong>Query</strong>: What am I looking for?<br>
                • <strong>Key</strong>: What do I contain?<br>
                • <strong>Value</strong>: What do I return if matched?<br>
                • The √d_k scaling prevents vanishing gradients in softmax<br>
                • WikiTruth AI uses Sentence-BERT (a BERT variant) which applies self-attention to encode both questions and Wikipedia passages into 384-dim vectors for semantic similarity comparison.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab_viz:
        st.markdown("""
        <div class="section-header">📈 Retrieval Similarity Visualization</div>
        """, unsafe_allow_html=True)

        if not is_db_populated():
            render_db_warning()
        else:
            viz_q = st.text_input(
                "Enter a question to visualize retrieval similarity",
                placeholder="e.g., What is deep learning?",
                key="viz_question",
            )
            viz_k = st.slider("Number of passages to retrieve", 3, 15, 10, key="viz_k")

            if st.button("📊 Visualize Retrieval", key="viz_btn"):
                pipeline = get_pipeline()
                try:
                    from src.retrieval import get_retriever
                    retriever = get_retriever()
                    chunks = retriever.retrieve(viz_q, k=viz_k)

                    if chunks:
                        import plotly.graph_objects as go

                        labels = [f"[{i+1}] {c['source'][:20]}" for i, c in enumerate(chunks)]
                        sims = [c.get("similarity", 0) for c in chunks]

                        fig = go.Figure()
                        colors = ["#4fc3f7" if s > 0.5 else "#ff9800" if s > 0.3 else "#ef5350" for s in sims]

                        fig.add_trace(go.Bar(
                            x=labels, y=sims,
                            marker_color=colors,
                            text=[f"{s:.3f}" for s in sims],
                            textposition="outside",
                        ))

                        # Threshold lines
                        fig.add_hline(y=0.5, line_dash="dash", line_color="#4caf50",
                                      annotation_text="High confidence (0.5)")
                        fig.add_hline(y=0.25, line_dash="dash", line_color="#ef5350",
                                      annotation_text="Minimum threshold (0.25)")

                        fig.update_layout(
                            title=f"Cosine Similarity Scores for: '{viz_q[:50]}'",
                            xaxis_title="Retrieved Passage",
                            yaxis_title="Cosine Similarity",
                            yaxis_range=[0, 1.1],
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(13,27,42,0.5)",
                            font=dict(color="#e2e8f0", size=11),
                            height=400,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.markdown(f"""
                        <div class="info-box">
                            <strong>Interpretation:</strong> Blue bars (sim > 0.5) are highly relevant passages.
                            Orange (0.25-0.5) are moderately relevant. Red (&lt; 0.25) are below the minimum
                            confidence threshold and may not contain the answer.
                            Average similarity: <strong>{sum(sims)/len(sims):.3f}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Visualization failed: {e}")


# ===========================================================================
# PAGE 6: ABOUT PROJECT
# ===========================================================================

def page_about_project():
    st.markdown("""
    <div class="hero-title" style="font-size:1.8rem; margin-bottom:0.5rem;">ℹ️ About WikiTruth AI</div>
    <div style="color:#90caf9; margin-bottom:1.5rem; font-size:0.9rem;">
        P53 — Wikipedia-Grounded Answer Bot | MIT ADT University | Generative AI Course
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Problem Statement", "🎓 Syllabus Units", "⚖️ Responsible AI", "🔧 Limitations"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("""
            <div class="wiki-card">
                <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:0.75rem;">Problem Statement</div>
                <p style="color:#e2e8f0; line-height:1.7;">
                    Large Language Models (LLMs) are powerful but prone to hallucination — generating
                    plausible-sounding but factually incorrect statements. This is especially problematic
                    in educational and research contexts where factual accuracy is essential.
                </p>
                <p style="color:#e2e8f0; line-height:1.7;">
                    <strong style="color:#4fc3f7;">WikiTruth AI</strong> solves this by implementing
                    Retrieval-Augmented Generation (RAG): constraining the LLM to answer exclusively
                    from verified Wikipedia sources, with explicit refusals when sources are insufficient.
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="wiki-card">
                <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:0.75rem;">Objectives</div>
                <div style="color:#e2e8f0; line-height:2; font-size:0.9rem;">
                    ✅ Build a complete RAG pipeline grounded in Wikipedia data<br>
                    ✅ Prevent hallucinations through strict prompt engineering<br>
                    ✅ Compare grounded vs ungrounded LLM performance empirically<br>
                    ✅ Implement zero-shot and few-shot grounding prompts<br>
                    ✅ Generate topic-relevant illustrations via AI image generation<br>
                    ✅ Evaluate using 20-question dataset with hallucination metrics<br>
                    ✅ Demonstrate responsible AI practices in NLP applications
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="wiki-card">
                <div style="color:#4fc3f7; font-size:1rem; font-weight:600; margin-bottom:0.75rem;">Project Details</div>
                <div style="font-size:0.85rem; color:#90caf9; line-height:2;">
                    <strong style="color:#e2e8f0;">Project Code</strong><br>P53<br><br>
                    <strong style="color:#e2e8f0;">Course</strong><br>Generative AI<br><br>
                    <strong style="color:#e2e8f0;">Institution</strong><br>MIT ADT University<br><br>
                    <strong style="color:#e2e8f0;">Deadline</strong><br>7 November 2026<br><br>
                    <strong style="color:#e2e8f0;">App Name</strong><br>WikiTruth AI<br><br>
                    <strong style="color:#e2e8f0;">Licence</strong><br>MIT
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        units = [
            ("U1", "Grounding Prompts & Prompt Engineering", "#4fc3f7", [
                "Zero-shot grounding prompt with strict Wikipedia-only rules",
                "Few-shot grounding prompt with supported/unsupported examples",
                "Side-by-side prompt comparison with scores",
                "GenAI lifecycle diagram",
                "Generative vs Discriminative AI comparison",
                "Embedding similarity visualization for retrieval",
            ]),
            ("U2", "LLM and RAG Implementation", "#ff9800", [
                "Complete RAG pipeline: embed → retrieve → validate → generate",
                "Sentence-BERT (all-MiniLM-L6-v2) for query and passage embedding",
                "ChromaDB vector database with cosine similarity",
                "Top-K configurable retrieval (1-10 passages)",
                "Context validation with confidence threshold (0.25)",
                "Fallback message when retrieval confidence is too low",
                "Retrieval score display per passage",
            ]),
            ("U3", "GPT, BERT & Multimodal", "#4caf50", [
                "GPT-style LLM: Flan-T5 (local) / Claude / GPT-3.5 for answer generation",
                "BERT-family: Sentence-BERT for semantic embeddings",
                "Similarity scores displayed for each retrieved passage",
                "Image generation: Stable Diffusion 2.1 via HF Inference API",
                "Matplotlib topic visualization as image fallback",
                "Generated images displayed alongside answers",
            ]),
            ("U4", "PyTorch + Image Generation", "#ab47bc", [
                "PyTorch backend for Sentence-BERT embeddings",
                "Local Flan-T5 via Hugging Face Transformers (PyTorch)",
                "Stable Diffusion image generation (HF Inference API)",
                "Matplotlib fallback illustration generator",
                "Image regeneration capability in UI",
                "Error handling for API rate limits and timeouts",
            ]),
            ("U5", "Responsible AI & Evaluation", "#ef5350", [
                "20-question evaluation dataset with difficulty levels",
                "Mode A: Ungrounded evaluation (raw LLM answers)",
                "Mode B: Grounded evaluation (RAG pipeline answers)",
                "Scoring rubric: 2=Correct, 1=Partial, 0=Wrong",
                "Accuracy, Error Rate, Hallucination Rate metrics",
                "Comparison charts: bar charts, error-rate charts",
                "CSV and JSON export of results",
                "Human-review capability (manual scoring mode)",
            ]),
        ]

        for code, title, color, features in units:
            st.markdown(f"""
            <div style="background:#111827; border:1px solid {color}33; border-left:4px solid {color};
                        border-radius:8px; padding:1.25rem; margin:0.75rem 0;">
                <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.75rem;">
                    <span style="background:{color}22; color:{color}; padding:0.2rem 0.6rem; 
                                 border-radius:6px; font-size:0.85rem; font-weight:700;">{code}</span>
                    <span style="color:#e2e8f0; font-weight:600;">{title}</span>
                </div>
                <div style="color:#90caf9; font-size:0.82rem; line-height:1.8;">
                    {"".join(f"✓ {f}<br>" for f in features)}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:1rem;">⚖️ Responsible AI Principles</div>

            <div style="margin-bottom:1rem;">
                <div style="color:#ff9800; font-weight:600;">Hallucination Prevention</div>
                <div style="color:#90caf9; font-size:0.85rem; line-height:1.6; margin-top:0.25rem;">
                    WikiTruth AI is designed from the ground up to prevent hallucinations.
                    The LLM is explicitly instructed to answer ONLY from retrieved Wikipedia passages,
                    refuse when context is insufficient, and cite sources for every factual claim.
                    The grounding status is displayed for every answer.
                </div>
            </div>

            <div style="margin-bottom:1rem;">
                <div style="color:#4caf50; font-weight:600;">Transparency & Attribution</div>
                <div style="color:#90caf9; font-size:0.85rem; line-height:1.6; margin-top:0.25rem;">
                    Every answer includes clickable Wikipedia source links.
                    The LLM provider, model name, and prompt style are displayed.
                    Retrieval similarity scores are shown for each passage.
                    All models used are documented in MODEL_CARD.md.
                </div>
            </div>

            <div style="margin-bottom:1rem;">
                <div style="color:#ab47bc; font-weight:600;">Data Privacy</div>
                <div style="color:#90caf9; font-size:0.85rem; line-height:1.6; margin-top:0.25rem;">
                    No user queries are stored or logged.
                    All data is sourced from Wikipedia (CC BY-SA 4.0 licence).
                    No proprietary datasets are used.
                    API keys are managed via environment variables and never committed to code.
                </div>
            </div>

            <div style="margin-bottom:1rem;">
                <div style="color:#ef5350; font-weight:600;">Limitations Disclosure</div>
                <div style="color:#90caf9; font-size:0.85rem; line-height:1.6; margin-top:0.25rem;">
                    Limited to 10 Wikipedia AI/ML articles. The system does not claim to cover
                    all human knowledge. Users are shown explicit warnings when confidence is low.
                    Evaluation metrics are computed automatically and may not perfectly reflect
                    human judgement of answer quality.
                </div>
            </div>

            <div>
                <div style="color:#29b6f6; font-weight:600;">Third-Party Attribution</div>
                <div style="color:#90caf9; font-size:0.85rem; line-height:1.6; margin-top:0.25rem;">
                    This project uses ChromaDB (Apache 2.0), Sentence Transformers (Apache 2.0),
                    Hugging Face Transformers (Apache 2.0), Flan-T5 (Apache 2.0), and Streamlit (Apache 2.0).
                    Wikipedia content is licensed under CC BY-SA 4.0. All third-party licences are preserved.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("""
        <div class="wiki-card">
            <div style="color:#4fc3f7; font-size:1.1rem; font-weight:600; margin-bottom:1rem;">🔧 Known Limitations</div>
            <div style="color:#e2e8f0; font-size:0.85rem; line-height:1.9;">

                <div style="color:#ef5350; font-weight:600;">Knowledge Scope</div>
                The system is limited to 10 Wikipedia AI/ML articles. Questions outside these topics
                will receive "I cannot answer" responses, which is correct behavior — not a bug.<br><br>

                <div style="color:#ff9800; font-weight:600;">Local LLM Quality</div>
                Flan-T5-base (250M params) produces shorter, less nuanced answers than
                Claude or GPT-3.5. The grounding constraints work but answer quality is lower.
                Setting API keys improves answer quality significantly.<br><br>

                <div style="color:#ab47bc; font-weight:600;">Image Generation Rate Limits</div>
                The HF Inference API free tier limits image generation to ~1 image/minute.
                Rate-limited requests fall back to matplotlib visualizations automatically.<br><br>

                <div style="color:#29b6f6; font-weight:600;">Auto-Scoring Limitations</div>
                Keyword-based auto-scoring is a proxy for factual accuracy. It may
                score partially incorrect answers higher if they contain expected keywords.
                Manual review is recommended for academic evaluation.<br><br>

                <div style="color:#4caf50; font-weight:600;">Deployment Considerations</div>
                Streamlit Community Cloud has ~1GB RAM limit. The local Flan-T5 model requires
                ~1GB RAM. Use `--lightweight` mode or API providers for resource-constrained deployments.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# MAIN ROUTER
# ===========================================================================

def main():
    page = render_sidebar()

    if page == "Home":
        page_home()
    elif page == "Ask WikiTruth":
        page_ask_wikitruth()
    elif page == "Grounded vs Ungrounded":
        page_grounded_vs_ungrounded()
    elif page == "Evaluation Dashboard":
        page_evaluation_dashboard()
    elif page == "Prompt Engineering Lab":
        page_prompt_engineering_lab()
    elif page == "About Project":
        page_about_project()


if __name__ == "__main__":
    main()
