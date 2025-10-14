# AI News Recommendation System

I built this news recommendation system for getting hands-on experience in modern AI technologies including semantic search, neural reranking, multi-model fusion, and agentic AI features. This is an ongoing project that demonstrates practical implementation of recommendation systems and RAG architectures. This project does not use external RAG frameworks for pure understanding of the system.

## Overview

This system combines multiple AI technologies to provide intelligent news recommendations:

- **FAISS Vector Search**: High-performance similarity search
- **Hybrid Retrieval**: BM25+dense fetch.
- **RRF**: Rank Hybrid Retrieval / Combine Diff. Embedding Model Scores
- **Cross-Encoder Reranking**: Neural relevance scoring
- **Custom Graph RAG**: Entity-based recommendation expansion
- **MMR Diversification**: Preventing repetitive recommendations

### Agentic Features (In Progress)
- **Content Validation for Journalists**: AI agents assist in fact-checking and content verification to support journalist reports on the system or open source journalism for qualified users.

## Quick Start

### Prerequisites
```bash
# Create conda environment
conda env create -f environment.yml
conda activate newsenv
```

### Setup and Launch
```bash
# Setup data (one-time)
python scripts/demo.py --setup

# Launch web interface
streamlit run streamlit_app.py
```

## Core Features

### Web Interface
- **Interactive Recommendations**: Select an article and see AI-powered article suggestions with multiple model options.
- **Save and Summarize Article**: AI-generated summaries of selected articles using  transformer models.
- **Live Data Refresh**: Automatic updates from RSS feeds based on number of news you want to see.
<img width="1898" height="866" alt="image" src="https://github.com/user-attachments/assets/e6db368e-4268-46a7-933a-9977b1ea02d1" />


## System Architecture

### Recommendation Pipeline
1. **Content Processing**: Article ingestion and preprocessing
2. **Embedding Generation**: Multi-model semantic representations
3. **Hybrid Search**: Combining BM25, semantic, and graph-based retrieval
4. **Neural Reranking**: Cross-encoder precision refinement
5. **Diversification**: MMR-based result variety
6. **Content Validation**: AI agents assist in journalistic content verification (in progress)

### Evaluation Framework
- **SPICED Dataset Integration**: Standardized evaluation metrics
- **MRR and Hit@K Metrics**: Industry-standard recommendation evaluation
- **Multi-Configuration Testing**: Systematic performance comparison

## Configuration Options

### Recommendation Models
- **Basic**: Fast semantic similarity search without Graph RAG behaviour or cross-encoder reranking
- **Enhanced**: Graph RAG with entity expansion
- **Full**: Complete pipeline with cross-encoder reranking

### Evaluation Pipeline
- **Similarity Detection**: MRR and Hit@K metrics
- **Diversity Assessment**: Topic coverage analysis
- **Performance Comparison**: Multi-configuration benchmarking

## Project Status

This is an ongoing learning project that started for self development. The ideas in this project is subject to use in the Inform Me project, please check the licence.
