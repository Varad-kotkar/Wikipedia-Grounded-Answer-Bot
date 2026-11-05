# WikiTruth AI — Model Card

**Project**: P53 — Wikipedia-Grounded Answer Bot  
**Institution**: MIT ADT University | Generative AI Course  
**Date**: November 2026

---

## Models Used

### 1. Sentence-BERT: `all-MiniLM-L6-v2` (Embedding Model)

| Property | Value |
|----------|-------|
| **Model ID** | `sentence-transformers/all-MiniLM-L6-v2` |
| **Type** | Sentence Transformer (BERT-family) |
| **Task** | Semantic text embedding |
| **Embedding Dim** | 384 |
| **Model Size** | ~90 MB |
| **Licence** | Apache 2.0 |
| **Source** | [HuggingFace](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| **API Key Required** | No |

**Use in WikiTruth AI**: Encodes both Wikipedia passage chunks and user questions into 384-dimensional vectors. Cosine similarity between these vectors is used to retrieve the most relevant passages.

**How it works**: MiniLM is a distilled BERT variant fine-tuned using contrastive learning on sentence pairs. The result is a model that maps semantically similar sentences to nearby points in embedding space.

**Limitations**: 384-dim embeddings may miss fine-grained semantic distinctions compared to larger models like `text-embedding-ada-002`.

---

### 2. Google Flan-T5 Base: `google/flan-t5-base` (Local LLM)

| Property | Value |
|----------|-------|
| **Model ID** | `google/flan-t5-base` |
| **Type** | Seq2Seq Transformer (GPT-style instruction-tuned) |
| **Task** | Text generation (question answering) |
| **Parameters** | 250M |
| **Model Size** | ~1 GB |
| **Licence** | Apache 2.0 |
| **Source** | [HuggingFace](https://huggingface.co/google/flan-t5-base) |
| **API Key Required** | No |

**Use in WikiTruth AI**: Local LLM fallback when no API keys are configured. Generates grounded and ungrounded answers using the retrieved Wikipedia context.

**Fine-tuning status**: NOT fine-tuned. Used as-is from the pre-trained checkpoint. The grounding behavior is enforced entirely through prompt engineering (zero-shot and few-shot prompts).

**Limitations**: Shorter context window than GPT-3.5/Claude. May produce lower-quality answers on complex synthesis questions. Answer length is limited to `MAX_TOKENS` tokens.

---

### 3. Anthropic Claude Haiku (Optional Cloud LLM)

| Property | Value |
|----------|-------|
| **Model ID** | `claude-haiku-4-5-20251001` |
| **Type** | Autoregressive transformer (GPT-style) |
| **Task** | Text generation (question answering) |
| **Licence** | Proprietary — Anthropic API |
| **API Key Required** | Yes (`ANTHROPIC_API_KEY`) |

**Use in WikiTruth AI**: High-quality answer generation when API key is available. Follows the same grounding prompts as the local model.

**Fine-tuning status**: NOT fine-tuned. Original model checkpoint with prompt engineering only.

---

### 4. OpenAI GPT-3.5 Turbo (Optional Cloud LLM)

| Property | Value |
|----------|-------|
| **Model ID** | `gpt-3.5-turbo` |
| **Type** | Autoregressive transformer |
| **Task** | Text generation |
| **Licence** | Proprietary — OpenAI API |
| **API Key Required** | Yes (`OPENAI_API_KEY`) |

**Use in WikiTruth AI**: Alternative to Anthropic Claude when `OPENAI_API_KEY` is set.

---

### 5. Stable Diffusion 2.1 (Image Generation)

| Property | Value |
|----------|-------|
| **Model ID** | `stabilityai/stable-diffusion-2-1` |
| **Type** | Latent Diffusion Model |
| **Task** | Text-to-image generation |
| **Licence** | CreativeML OpenRAIL-M |
| **Source** | [HuggingFace](https://huggingface.co/stabilityai/stable-diffusion-2-1) |
| **API Key Required** | No (free tier) / `HF_TOKEN` for higher limits |

**Use in WikiTruth AI**: Generates topic-relevant illustrations for each answer. Accessed via the HuggingFace Inference API. When unavailable or rate-limited, a matplotlib topic visualization is used as fallback.

**Limitations**: HF free tier rate limits (~1 image/min). Images are generated from text prompts and may not always be topically precise.

---

## Responsible AI Notes

### Grounding and Hallucination Prevention
WikiTruth AI is specifically designed to minimize hallucinations by:
1. Restricting all answers to retrieved Wikipedia passages only
2. Explicitly instructing the LLM to refuse if context is insufficient
3. Flagging low-confidence retrievals (similarity < 0.25)
4. Comparing grounded vs ungrounded responses to demonstrate the value of RAG

### Known Limitations
- The system only covers the 10 Wikipedia AI/ML articles ingested at setup time
- Local Flan-T5 may produce lower-quality answers than paid API models
- Automatic scoring (keyword matching) is a proxy for factual accuracy, not a perfect measure
- Image generation may fail or produce imprecise illustrations

### Data Sources
- Wikipedia articles are licensed under the Creative Commons Attribution-ShareAlike 4.0 International Licence (CC BY-SA 4.0)
- All retrieved passages include attribution to their Wikipedia source articles
- No proprietary or private data is used

### Attribution
This project uses open-source models. All third-party model licences and attribution requirements are preserved. The project does not claim ownership of any third-party model weights, datasets, or software.
