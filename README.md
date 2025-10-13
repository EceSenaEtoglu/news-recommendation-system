# AI News Recommendation System

A comprehensive news recommendation system built for learning modern AI technologies including semantic search, neural reranking, multi-model fusion, and agentic AI features. This is an ongoing project that demonstrates practical implementation of advanced recommendation systems and RAG architectures.

## Overview

This system combines multiple AI technologies to provide intelligent news recommendations:
- **Semantic Search**: FAISS-based vector similarity search
- **Neural Reranking**: Cross-encoder models for precision
- **Multi-Model Fusion**: Combining different embedding strategies
- **Graph RAG**: Entity-based recommendation expansion
- **Agentic Features**: AI-powered content analysis and recommendations

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
- **Interactive Recommendations**: AI-powered article suggestions with multiple model options
- **Semantic Search**: Find articles by meaning, not just keywords
- **Article Summarization**: AI-generated summaries using transformer models
- **Real-time Configuration**: Adjust recommendation parameters on-the-fly
- **Live Data Refresh**: Automatic updates from RSS feeds

### AI Technologies
- **FAISS Vector Search**: High-performance similarity search
- **Sentence Transformers**: Multi-model embedding systems
- **Cross-Encoder Reranking**: Neural relevance scoring
- **Graph RAG**: Entity-based recommendation expansion
- **MMR Diversification**: Preventing repetitive recommendations
- **Multi-Model Fusion**: Combining different AI approaches

### Agentic Features (In Progress)
- **Content Validation for Journalists**: AI agents assist in fact-checking and content verification
- **Open Source Journalism Support**: Tools to help qualified users contribute to open journalism
- **Automated Quality Assessment**: AI-powered evaluation of article credibility and accuracy

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
- **Real-time Performance Monitoring**: Continuous system assessment

## Configuration Options

### Recommendation Models
- **Basic**: Fast semantic similarity search
- **Enhanced**: Graph RAG with entity expansion
- **Full**: Complete pipeline with cross-encoder reranking

### Evaluation Pipeline
- **Similarity Detection**: MRR and Hit@K metrics
- **Diversity Assessment**: Topic coverage analysis
- **Performance Comparison**: Multi-configuration benchmarking

## Project Status

This is an ongoing learning project that started for educational purposes. Current focus areas:
- Refining recommendation algorithms
- Developing AI agents for journalistic content validation
- Improving evaluation metrics
- Optimizing system performance

## Technical Stack

- **Python**: Core implementation
- **Streamlit**: Web interface
- **FAISS**: Vector search
- **Transformers**: Neural models
- **spaCy**: NLP processing
- **scikit-learn**: Machine learning utilities

## Contributing

This project is primarily for learning and experimentation. Suggestions and improvements are welcome, especially around:
- Recommendation algorithm enhancements
- Evaluation methodology improvements
- Performance optimizations

