"""
src/image_generator.py — WikiTruth AI
Generates topic-relevant illustrations using the HF Inference API (Stable Diffusion)
with a matplotlib fallback when the API is unavailable.
"""

import io
import logging
import os
import random
import textwrap
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _build_image_prompt(topic: str) -> str:
    """Build an art-style prompt from a topic string."""
    styles = [
        "digital art, detailed, vibrant colors",
        "scientific illustration, clean, educational",
        "conceptual art, modern, minimalist",
        "infographic style, blue and white color scheme",
    ]
    style = random.choice(styles)
    return (
        f"A detailed illustration representing the concept of '{topic}'. "
        f"Wikipedia-style educational image. {style}. "
        "High quality, no text, no watermarks."
    )


def generate_image_hf_api(topic: str, timeout: int = 30) -> Optional[bytes]:
    """
    Generate an image using the Hugging Face Inference API.
    Returns image bytes (PNG) or None if unavailable.
    """
    try:
        import requests
        from src.config import IMAGE_GEN_API_URL, HF_TOKEN
    except ImportError:
        logger.warning("requests not installed, cannot use HF API")
        return None

    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    prompt = _build_image_prompt(topic)
    payload = {
        "inputs": prompt,
        "parameters": {
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
        },
    }

    for attempt in range(2):  # 2 attempts
        try:
            response = requests.post(
                IMAGE_GEN_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if response.status_code == 200:
                return response.content
            elif response.status_code == 503:
                # Model loading, wait and retry
                wait_time = response.json().get("estimated_time", 10)
                logger.info(f"Model loading, waiting {wait_time:.0f}s...")
                time.sleep(min(wait_time, 15))
            elif response.status_code == 429:
                logger.warning("HF API rate limit hit, using fallback")
                return None
            else:
                logger.warning(f"HF API error {response.status_code}: {response.text[:200]}")
                return None
        except Exception as e:
            logger.warning(f"HF API request failed (attempt {attempt+1}): {e}")
    
    return None


def generate_image_matplotlib(topic: str) -> Optional[bytes]:
    """
    Generate a topic illustration using matplotlib as a fallback.
    Creates a visually appealing topic card with related keywords.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np
    except ImportError:
        logger.error("matplotlib not installed, cannot generate fallback image")
        return None

    # Topic-to-keywords mapping
    topic_keywords = {
        "artificial intelligence": ["Intelligence", "Learning", "Cognition", "Robots", "Neural", "Reasoning", "Automation", "Perception"],
        "machine learning": ["Algorithms", "Training", "Data", "Models", "Prediction", "Gradient", "Features", "Optimization"],
        "deep learning": ["Layers", "Neurons", "Backprop", "CNN", "RNN", "Transformers", "Features", "Representations"],
        "neural network": ["Neurons", "Weights", "Activation", "Layers", "Connections", "Perceptron", "Bias", "Forward"],
        "reinforcement learning": ["Agent", "Environment", "Reward", "Policy", "Action", "State", "Q-Learning", "Exploration"],
        "supervised learning": ["Labels", "Training", "Classification", "Regression", "Features", "Validation", "Accuracy", "Loss"],
        "unsupervised learning": ["Clustering", "Patterns", "K-means", "PCA", "Dimensionality", "Anomaly", "Autoencoders", "Groups"],
        "data science": ["Analytics", "Visualization", "Statistics", "Python", "Insights", "Big Data", "ETL", "Mining"],
        "big data": ["Volume", "Velocity", "Variety", "Hadoop", "Spark", "Distributed", "Storage", "Streaming"],
        "statistics": ["Probability", "Distribution", "Hypothesis", "Variance", "Regression", "Sampling", "Bayesian", "Inference"],
    }

    # Find best keyword set
    keywords = None
    topic_lower = topic.lower()
    for key, kws in topic_keywords.items():
        if key in topic_lower or topic_lower in key:
            keywords = kws
            break
    
    if keywords is None:
        # Generic keywords from topic words
        words = topic.split()
        keywords = words * (8 // len(words) + 1)
        keywords = [w.capitalize() for w in keywords[:8]]

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.patch.set_facecolor("#0a0a1a")
    ax.set_facecolor("#0a0a1a")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Background gradient effect
    for i in range(100):
        alpha = 0.03
        circle = plt.Circle(
            (5, 3.5),
            radius=(i / 100) * 6,
            color="#1a3a6a",
            alpha=alpha,
            fill=True,
        )
        ax.add_patch(circle)

    # Title
    ax.text(
        5, 6.2, topic.upper(),
        ha="center", va="center",
        fontsize=18, fontweight="bold",
        color="#4fc3f7",
        fontfamily="monospace",
    )
    
    # Wikipedia brand line
    ax.text(
        5, 5.6, "Wikipedia Knowledge Base",
        ha="center", va="center",
        fontsize=9, color="#90caf9", style="italic",
    )

    # Draw keyword bubbles
    positions = [
        (2.0, 4.5), (8.0, 4.5), (1.5, 2.5), (8.5, 2.5),
        (3.5, 1.2), (6.5, 1.2), (5.0, 4.0), (5.0, 2.0),
    ]
    colors = ["#1565c0", "#283593", "#0d47a1", "#1a237e",
              "#1565c0", "#283593", "#0d47a1", "#1a237e"]
    
    for kw, pos, col in zip(keywords[:8], positions[:8], colors[:8]):
        # Bubble background
        bubble = patches.FancyBboxPatch(
            (pos[0] - 0.9, pos[1] - 0.3), 1.8, 0.6,
            boxstyle="round,pad=0.1",
            facecolor=col, edgecolor="#4fc3f7", linewidth=1.5, alpha=0.9,
        )
        ax.add_patch(bubble)
        ax.text(
            pos[0], pos[1], kw,
            ha="center", va="center",
            fontsize=8, color="white", fontweight="bold",
        )
        # Connect to center
        ax.plot(
            [5.0, pos[0]], [3.5, pos[1]],
            color="#4fc3f7", alpha=0.3, linewidth=1, linestyle="--",
        )

    # Center node
    center_circle = plt.Circle((5, 3.5), 0.6, color="#4fc3f7", alpha=0.9)
    ax.add_patch(center_circle)
    ax.text(
        5, 3.5, "W",
        ha="center", va="center",
        fontsize=14, fontweight="bold", color="#0a0a1a",
    )

    # Footer
    ax.text(
        5, 0.3, "WikiTruth AI — Grounded Knowledge",
        ha="center", va="center",
        fontsize=7, color="#546e7a",
    )

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_image(topic: str, prefer_api: bool = True) -> Optional[bytes]:
    """
    Generate a topic-relevant illustration.
    
    Tries HF Inference API first (if prefer_api=True and HF_TOKEN set),
    then falls back to matplotlib visualization.
    
    Args:
        topic: subject to illustrate (e.g., "Machine learning")
        prefer_api: if True, attempt HF API before fallback
    
    Returns:
        PNG image bytes or None on complete failure
    """
    img_bytes = None
    
    if prefer_api:
        try:
            img_bytes = generate_image_hf_api(topic)
        except Exception as e:
            logger.warning(f"HF API generation failed: {e}")
    
    if img_bytes is None:
        logger.info(f"Using matplotlib fallback for topic: {topic}")
        img_bytes = generate_image_matplotlib(topic)
    
    return img_bytes
