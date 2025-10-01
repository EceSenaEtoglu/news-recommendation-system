# 🚀 AI News MVP

I binge coded this **MVP** in couple of days for hands-on experience with modern AI technologies including **FAISS**, **semantic retrieval**, **neural reranking**, **multi-model fusion**, and **transformer-based summarization**. This project demonstrates practical implementation of advanced recommendation systems and RAG architectures. More features and improvements coming soon!

## 🎯 Quick Start

### Web UI Demo**

The easiest way to explore the system:

```bash
# 0. Install dependencies (one-time)
pip install -r requirements.txt

# 1. Setup data (one-time)
python scripts/demo.py --setup

# 2. Launch interactive web interface
streamlit run streamlit_app.py
```

**Web Interface Features:**
- 🎯 **Interactive AI Recommendations** with multiple models
- 📰 **Featured Articles** with semantic search
- 📚 **Saved Articles** with AI summarization
- ⚙️ **Real-time Configuration** of AI parameters
- 🔄 **Live News Data Refresh** from RSS feeds


## 🧠 Technologies Demonstrated

### **Core AI Technologies**
- **🔍 FAISS Vector Search**: High-performance similarity search
- **🧠 Sentence Transformers**: Multi-model embedding systems
- **⚡ Neural Reranking**: Deep learning for recommendation refinement
- **🔄 Multi-Model Fusion**: Combining multiple AI models
- **📝 Transformer Summarization**: BART-based article summarization
- **🌐 Graph RAG**: Entity-based recommendation expansion

### **Advanced Features**
- **🎯 MMR Diversification**: Preventing repetitive recommendations
- **📊 Logistic Reranking**: 10+ feature-based scoring
- **🔗 Entity Extraction**: spaCy NER for relationship discovery
- **⚖️ Cross-Encoder**: Neural relevance scoring
- **📈 Multi-Model Embeddings**: News-specific vs general models
- **🌐 Graph RAG**: Entity-based recommendation expansion


## 🎮 Web Interface Guide

### **Main Dashboard**
1. **📰 Featured Articles**: Browse curated news with semantic search
2. **🎯 AI Recommendations**: Get personalized suggestions using different AI models
3. **📚 Saved Articles**: Manage your reading list with AI summaries

### **AI Model Options**
- **Basic**: Fast semantic similarity search
- **Enhanced (Neural)**: Deep learning reranking (future use)
- **Multi-Model**: Fusion of multiple embedding models (default)

### **Interactive Features**
- **Real-time Configuration**: Adjust recommendation count, diversity settings
- **Score Explanations**: Understand how AI calculates relevance
- **One-Click Summarization**: AI-powered article summaries
- **Live Data Refresh**: Fetch latest news from RSS feeds


### **Coming Soon**
- 🔮 **User Profiles**: Personalized recommendations
- 🔮 **Real-time Learning**: Continuous model improvement
- 🔮 **Advanced Analytics**: Recommendation insights

## 🐛 Troubleshooting

### **Setup Issues**
```bash
# Install dependencies first
pip install -r requirements.txt

# If no articles found
python scripts/demo.py --setup

# If models fail to load
python scripts/demo.py --fetch
```

### **Common Installation Issues**
- **Streamlit not found**: `pip install streamlit`
- **FAISS installation**: `pip install faiss-cpu` (or `faiss-gpu` for GPU)
- **spaCy model**: `python -m spacy download en_core_web_sm`
- **Transformers**: `pip install transformers torch`

### **Common Issues**
- **Memory**: Close other applications for large models
- **Network**: Ensure internet connection for model downloads
- **Dependencies**: Install all requirements: `pip install -r requirements.txt`

---
