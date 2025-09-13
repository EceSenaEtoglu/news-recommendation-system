# 🚀 RAGify-News AI Demo

A comprehensive demonstration of the AI-powered news recommendation system showcasing neural reranking, multi-model embeddings, and advanced recommendation algorithms.

## 🎯 Quick Start

### 1. Setup Data
```bash
# Setup everything in one command
python scripts/demo.py --setup
```

### 2. Run Demo

#### **Command Line Demo**
```bash
# Run comprehensive demo
python scripts/demo.py --demo

# Or just run the demo (default)
python scripts/demo.py
```

#### **Web UI Demo**
```bash
# Launch Streamlit web interface
streamlit run streamlit_app.py
```

The Streamlit app provides:
- 🎯 **Interactive recommendations** with different AI models
- 🚀 **One-click AI demo** button
- 📊 **Visual interface** for exploring articles
- ⚙️ **Easy data setup** and index rebuilding

## 🧠 What the Demo Shows

### **Demo 1: Multi-Model Embeddings**
- Lists all available embedding models
- Tests different models (general vs news-specific)
- Shows embedding dimensions and capabilities

### **Demo 2: Neural Reranker**
- Initializes neural network reranker
- Trains on synthetic data
- Shows advanced feature extraction (10+ features)

### **Demo 3: Enhanced Recommendations**
- **Basic**: Simple semantic similarity
- **Neural Reranker**: Deep learning reranking
- **MMR Diversification**: Diversity-aware selection
- **Full Enhanced**: Neural + MMR combined

### **Demo 4: Multi-Model Fusion**
- **Weighted Average**: Combines scores with weights
- **Rank Fusion**: Reciprocal rank fusion
- **Max Score**: Conservative maximum approach

### **Demo 5: CLI Commands**
- Lists all available commands
- Shows usage examples
- Provides next steps

## 🎯 Available Commands

| Command | Description |
|---------|-------------|
| `--setup` | Setup data (import fixtures and build index) |
| `--demo` | Run comprehensive demo |
| `--recommend <id>` | Basic recommendations for article ID |
| `--enhanced <id>` | Enhanced recommendations with neural reranker |
| `--multi-model <id>` | Multi-model fusion recommendations |
| `--list-models` | List available embedding models |
| `--model-info` | Show current model information |

## 🔧 Advanced Usage

### Enhanced Recommendations
```bash
# Use neural reranker
python scripts/demo.py --enhanced <article_id>

# Use specific model
python scripts/demo.py --enhanced <article_id> --model news-similarity
```

### Multi-Model Fusion
```bash
# Default models
python scripts/demo.py --multi-model <article_id>

# Custom models
python scripts/demo.py --multi-model <article_id> --models all-MiniLM-L6-v2 news-similarity
```

### Model Management
```bash
# List available models
python scripts/demo.py --list-models

# Show current model info
python scripts/demo.py --model-info
```

## 📊 Expected Output

The demo will show:
- ✅ Database statistics
- 🤖 Available embedding models
- 🧠 Neural reranker training progress
- 🎯 Different recommendation approaches
- 🔄 Multi-model fusion results
- 📋 Available CLI commands

## 🚀 Next Steps

After running the demo:
1. Try enhanced recommendations with specific articles
2. Experiment with different embedding models
3. Compare basic vs enhanced approaches
4. Explore multi-model fusion methods

## 🐛 Troubleshooting

### No Articles Error
```bash
❌ No articles in database!
Please run: python scripts/demo.py --setup
```

### Model Download Issues
- Check internet connection
- Some models may take time to download
- Fallback to default model if specific model fails

### Memory Issues
- Use smaller models: `--model all-MiniLM-L6-v2`
- Reduce batch size in neural config
- Close other applications

---

