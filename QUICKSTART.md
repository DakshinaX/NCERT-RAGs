# 🚀 Quick Start Guide - NCERT Physics RAG System

## For Google Colab Users (Recommended for Beginners)

### Step 1: Upload Files
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `NCERT_Physics_RAG_Colab.ipynb`
3. Upload `Data.json` and `Evaluation_Set.json` (or `Evaluation Set.json`) to `/content/`

### Step 2: Run the Notebook
1. Click "Runtime" → "Run all"
2. Wait 10-15 minutes for setup
3. Start querying!

**That's it!** The notebook handles everything automatically.

---

## For Advanced Users (Local Setup)

### Prerequisites
- Python 3.10 or higher
- 12GB RAM minimum (16GB recommended)
- 15GB free disk space
- Optional: NVIDIA GPU with 6GB+ VRAM

### Installation Steps

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# 4. Place data files
# - Data.json (8.2MB)
# - Evaluation_Set.json (56KB)
```

### Running the System

#### Option A: Using the Main Script

```bash
python ncert_physics_rag_system.py
```

**Important:** First edit the script to set correct paths:
```python
# In the script, update these lines:
config = RAGConfig()
config.data_path = "path/to/Data.json"
config.eval_path = "path/to/Evaluation_Set.json"
```

#### Option B: Interactive Python

```python
# Start Python
python

# Import and initialize
from ncert_physics_rag_system import *

# Setup configuration
config = RAGConfig()
config.data_path = "Data.json"
config.eval_path = "Evaluation_Set.json"

# Load data
loader = DataLoader(config.data_path, config.eval_path)
corpus, _ = loader.load_corpus()
eval_questions, _ = loader.load_evaluation_set()
documents = loader.prepare_documents(corpus)

# Build vector store (takes 3-5 minutes)
vs_builder = VectorStoreBuilder(config)
vector_store = vs_builder.build(documents)

# Load LLM (takes 5-7 minutes)
llm = QuantizedLLM(config)
llm.load()

# Create RAG pipeline
rag = RAGPipeline(vs_builder, llm, config)

# Test a query
response = rag.query("What is a unit in physics?")
print(response['answer'])

# Run evaluation
evaluator = RAGEvaluator(rag, config)
results = evaluator.run_full_evaluation(eval_questions)
```

---

## Common Issues & Solutions

### Issue 1: Out of Memory (OOM)
**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce batch size: `config.eval_batch_size = 1`
- Use smaller model: `config.llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"`
- Enable quantization: `config.use_quantization = True` (should be default)
- Reduce top_k: `config.top_k = 3`

### Issue 2: Slow Inference
**Problem:** Generation takes too long

**Solutions:**
- Use GPU if available (install `faiss-gpu` instead of `faiss-cpu`)
- Reduce max_new_tokens: `config.max_new_tokens = 256`
- Cache vector store: Save after first build, load in subsequent runs
- Disable hybrid search: `config.use_hybrid_search = False`

### Issue 3: Model Download Fails
**Error:** Connection timeout or download interrupted

**Solutions:**
```bash
# Pre-download models
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.2')"
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
```

### Issue 4: Import Errors
**Error:** `ModuleNotFoundError: No module named 'xyz'`

**Solutions:**
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install transformers sentence-transformers faiss-cpu langchain
```

---

## Testing Your Setup

### Quick Test Script

```python
# Save as test_setup.py
import torch
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

print("Testing PyTorch...")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")

print("\nTesting Transformers...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("  ✓ Transformers working")

print("\nTesting Sentence Transformers...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedding = model.encode(["test"])
print(f"  ✓ Generated embedding shape: {embedding.shape}")

print("\n✅ All tests passed! Your setup is ready.")
```

Run with:
```bash
python test_setup.py
```

---

## Sample Queries to Try

Once your system is running, try these questions:

**Conceptual:**
- "What is a unit in physics? Explain fundamental and derived units."
- "State Newton's second law of motion."
- "What is the difference between speed and velocity?"

**Numerical:**
- "A car accelerates from 10 m/s to 30 m/s in 5 seconds. Calculate the acceleration."
- "Calculate the force required to accelerate a 5 kg mass at 2 m/s²."

**Definitional:**
- "Define displacement."
- "What are the seven SI base quantities?"

**Reasoning:**
- "Why is it important to have a standard system of units?"
- "Explain why velocity is a vector quantity."

---

## Performance Expectations

### First-Time Setup
- **Dependency installation:** 5 minutes
- **Model downloads:** 5-7 minutes
  - Mistral-7B: ~4GB
  - BGE embeddings: ~130MB
- **Vector store creation:** 3-5 minutes
- **Total:** ~15-20 minutes

### Query Response Time
- **Retrieval:** <100ms
- **Generation:** 15-30 seconds (Colab Free, CPU)
- **Generation:** 5-10 seconds (with GPU)

### Evaluation Time
- **Full evaluation (60 questions):** 30-45 minutes
- **Sample evaluation (20 questions):** 10-15 minutes

---

## Next Steps

After successful setup:

1. **Explore the Notebook:** Run through all cells to understand the pipeline
2. **Try Custom Queries:** Test with your own physics questions
3. **Run Evaluation:** Check system performance metrics
4. **Read README.md:** Understand architecture and optimization options
5. **Experiment:** Try different configuration parameters

---

## Need Help?

- **Documentation:** See README.md for detailed information
- **Issues:** Check "Common Issues & Solutions" above
- **Architecture:** Review the architecture diagrams in README.md
- **Community:** Open an issue on GitHub

---

**Happy Learning! 🎓**

*Built with ❤️ for physics students*
