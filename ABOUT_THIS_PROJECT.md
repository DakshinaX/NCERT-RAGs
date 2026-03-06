# About This Project

This repository contains a production-oriented Retrieval-Augmented Generation (RAG) system focused on NCERT Class 11 Physics question answering.

## What it does
- Answers textbook-aligned physics questions (conceptual, definitional, reasoning, and numerical).
- Retrieves relevant textbook chunks and uses them as grounded context for generation.
- Returns source-aware responses with citation-style references.

## Core stack
- **Embeddings**: `BAAI/bge-small-en-v1.5`
- **Vector retrieval**: FAISS (semantic) + BM25 (keyword)
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.2` (4-bit quantized option)
- **Evaluation**: retrieval metrics (Recall@k, MRR), ROUGE, and basic faithfulness checks

## Main files
- `ncert_physics_rag_system.py`: end-to-end Python implementation.
- `NCERT_Physics_RAG_Colab (1).ipynb`: Colab notebook workflow.
- `Data.json`: corpus chunks and metadata.
- `Evaluation Set.json`: benchmark questions and references.
- `README.md`, `ARCHITECTURE.md`, `QUICKSTART.md`, `PROJECT_SUMMARY.md`: project docs.

## Intended environment
- Optimized for Google Colab Free tier while still runnable locally.
- Dependency pinning is provided in `requirements.txt`.
