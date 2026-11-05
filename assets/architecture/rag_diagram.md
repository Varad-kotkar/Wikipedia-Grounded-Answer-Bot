# WikiTruth AI — RAG Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     WikiTruth AI — RAG Pipeline                 │
└─────────────────────────────────────────────────────────────────┘

  User Question (natural language)
         │
         ▼
  ┌─────────────────────┐
  │  Question Processing │  ← Tokenization, cleaning
  └─────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────┐
  │  Sentence-BERT Embedding            │
  │  Model: all-MiniLM-L6-v2           │
  │  Output: 384-dim vector            │
  └─────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────┐
  │  ChromaDB Vector Database           │
  │  10 Wikipedia AI/ML articles       │
  │  ~300 chunks, cosine similarity    │
  └─────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────┐
  │  Top-K Passage Retrieval            │
  │  Similarity scores displayed       │
  │  User-adjustable K (1-10)          │
  └─────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────┐
  │  Context Validation                 │
  │  Low confidence → fallback message  │
  └─────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────┐
  │  Grounding Prompt Engineering       │
  │  Zero-shot OR few-shot             │
  │  Strict Wikipedia-only rules       │
  └─────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────┐
  │  Transformer LLM                    │
  │  Local: Flan-T5 (no API key)       │
  │  Cloud: Claude / GPT-3.5 (optional)│
  └─────────────────────────────────────┘
         │
         ▼
  ┌─────────────────────────────────────────────────────┐
  │  Answer Output                                      │
  │  ✓ Grounded answer text                             │
  │  ✓ Retrieved Wikipedia passages with scores        │
  │  ✓ Clickable Wikipedia source links                │
  │  ✓ Grounding status badge                          │
  │  ✓ Generated topic illustration                    │
  └─────────────────────────────────────────────────────┘
```
