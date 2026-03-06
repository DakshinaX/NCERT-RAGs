"""
Production-Ready RAG System for NCERT Class 11 Physics
========================================================

This implementation provides:
- Optimized for Google Colab Free (12GB RAM)
- Local open-source models (Mistral-7B with 4-bit quantization)
- Comprehensive evaluation pipeline with multiple metrics
- Handles conceptual, numerical, and reasoning questions
- Citation and source tracking

Architecture:
- Embeddings: BAAI/bge-small-en-v1.5 (384d)
- Vector Store: FAISS (in-memory)
- LLM: Mistral-7B-Instruct-v0.2 (4-bit quantized)
- Retriever: Top-k semantic search with metadata filtering
"""

#==============================================================================
# SECTION 1: INSTALL DEPENDENCIES
#==============================================================================

import subprocess
import sys

def install_dependencies():
    """Install all required packages for the RAG system."""
    
    packages = [
        "transformers==4.36.0",
        "sentence-transformers==2.2.2",
        "huggingface-hub==0.25.2",  # Compatible with sentence-transformers 2.2.2
        "faiss-cpu==1.7.4",  # Use faiss-gpu if GPU available
        "langchain==0.1.0",
        "langchain-community==0.0.10",
        "bitsandbytes==0.41.3",
        "accelerate==0.25.0",
        "rouge-score==0.1.2",
        "bert-score==0.3.13",
        "scikit-learn==1.3.2",
        "rank-bm25==0.2.2",
        "torch==2.1.2",
        "numpy<2.0",  # Compatibility with older packages
    ]
    
    print("📦 Installing dependencies...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"✓ {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
    
    print("\n✅ All dependencies installed!\n")

# Uncomment to run installation in Colab
# install_dependencies()

#==============================================================================
# SECTION 2: IMPORTS
#==============================================================================

import json
import os
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from tqdm.auto import tqdm

# Transformers and models
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from sentence_transformers import SentenceTransformer

# FAISS
import faiss

# LangChain
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS as LangChainFAISS
from langchain.embeddings.base import Embeddings

# Evaluation
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from sklearn.metrics.pairwise import cosine_similarity

# BM25 for hybrid search
from rank_bm25 import BM25Okapi

#==============================================================================
# SECTION 3: CONFIGURATION
#==============================================================================

@dataclass
class RAGConfig:
    """Configuration for the RAG system."""
    
    # Model configurations
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # Retrieval parameters
    top_k: int = 5
    similarity_metric: str = "cosine"  # cosine or euclidean
    use_hybrid_search: bool = True
    bm25_weight: float = 0.3  # Weight for BM25 in hybrid search
    
    # LLM parameters
    use_quantization: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.1  # Low temp for factual accuracy
    top_p: float = 0.9
    
    # System paths (for Colab)
    data_path: str = "/content/Data.json"
    eval_path: str = "/content/Evaluation_Set.json"
    cache_dir: str = "/content/model_cache"
    
    # Evaluation parameters
    eval_batch_size: int = 5
    
config = RAGConfig()

#==============================================================================
# SECTION 4: CUSTOM EMBEDDINGS WRAPPER
#==============================================================================

class SentenceTransformerEmbeddings(Embeddings):
    """Custom embeddings wrapper for LangChain compatibility."""
    
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = 512
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False).tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        return self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0].tolist()

#==============================================================================
# SECTION 5: DATA LOADING AND PREPROCESSING
#==============================================================================

class DataLoader:
    """Load and preprocess NCERT Physics dataset."""
    
    def __init__(self, data_path: str, eval_path: str):
        self.data_path = self._resolve_path(data_path, ["Data.json"])
        self.eval_path = self._resolve_path(
            eval_path,
            ["Evaluation_Set.json", "Evaluation Set.json"]
        )

    @staticmethod
    def _resolve_path(preferred_path: str, fallback_names: List[str]) -> str:
        """Resolve dataset paths across local/Colab naming variants."""
        if os.path.exists(preferred_path):
            return preferred_path

        search_candidates = []

        # Keep directory from preferred path and try known fallback names.
        preferred_dir = os.path.dirname(preferred_path) or "."
        for name in fallback_names:
            search_candidates.append(os.path.join(preferred_dir, name))

        # Also check current working directory (common for local runs).
        for name in fallback_names:
            search_candidates.append(name)

        for candidate in search_candidates:
            if os.path.exists(candidate):
                print(f"ℹ️ Using detected file path: {candidate}")
                return candidate

        raise FileNotFoundError(
            f"Could not find dataset file. Tried: {preferred_path} and {search_candidates}"
        )
        
    def load_corpus(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """Load the main corpus from Data.json."""
        print("📚 Loading corpus from Data.json...")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        corpus = data['data']
        metadata = data['metadata']
        
        print(f"✓ Loaded {len(corpus)} chunks")
        print(f"✓ Chapters: {metadata['chapters']}")
        print(f"✓ Content types: {metadata['content_types']}")
        
        return corpus, metadata
    
    def load_evaluation_set(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """Load the evaluation dataset."""
        print("\n📝 Loading evaluation set...")
        
        with open(self.eval_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)
        
        questions = eval_data['questions']
        metadata = eval_data['metadata']
        
        print(f"✓ Loaded {len(questions)} evaluation questions")
        print(f"✓ Question types: {metadata['question_type_distribution']}")
        
        return questions, metadata
    
    def prepare_documents(self, corpus: List[Dict]) -> List[Document]:
        """Convert corpus chunks to LangChain Document objects."""
        print("\n🔄 Preparing documents for indexing...")
        
        documents = []
        for chunk in tqdm(corpus, desc="Processing chunks"):
            # Use embedding_text for the content (optimized for RAG)
            content = chunk['rag_optimized']['embedding_text']
            
            # Create rich metadata for filtering and citation
            metadata = {
                'chunk_id': chunk['chunk_id'],
                'chapter_number': chunk['hierarchy']['chapter_number'],
                'chapter_title': chunk['hierarchy']['chapter_title'],
                'section_title': chunk['hierarchy']['section_title'],
                'content_type': chunk['content_type'],
                'is_problem': chunk['problems']['is_problem'],
                'has_diagram': chunk['visuals']['has_diagram'],
                'key_terms': ','.join(chunk['content']['key_terms']),
                'difficulty': chunk['metadata'].get('difficulty', 'unknown'),
                # Store original text for citation
                'original_text': chunk['content']['text'],
                'formulas': ','.join(chunk['content'].get('formulas', [])),
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
        
        print(f"✓ Prepared {len(documents)} documents")
        return documents

#==============================================================================
# SECTION 6: VECTOR STORE CREATION
#==============================================================================

class VectorStoreBuilder:
    """Build and manage FAISS vector store."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = None
        self.vector_store = None
        self.bm25 = None
        self.documents = None
        
    def build(self, documents: List[Document]) -> LangChainFAISS:
        """Build FAISS vector store with embeddings."""
        print("\n🔮 Creating embeddings and building vector store...")
        
        # Initialize embedding model
        print(f"Loading embedding model: {self.config.embedding_model_name}")
        self.embeddings = SentenceTransformerEmbeddings(self.config.embedding_model_name)
        
        # Store documents for BM25
        self.documents = documents
        
        # Create FAISS vector store
        print("Building FAISS index...")
        self.vector_store = LangChainFAISS.from_documents(
            documents,
            self.embeddings,
            distance_strategy="COSINE"
        )
        
        print(f"✓ Vector store built with {len(documents)} documents")
        
        # Build BM25 index for hybrid search
        if self.config.use_hybrid_search:
            print("Building BM25 index for hybrid search...")
            tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            print("✓ BM25 index built")
        
        return self.vector_store
    
    def hybrid_search(self, query: str, k: int) -> List[Document]:
        """Perform hybrid search combining semantic and keyword-based retrieval."""
        
        if not self.config.use_hybrid_search or self.bm25 is None:
            return self.vector_store.similarity_search(query, k=k)
        
        # Semantic search
        semantic_results = self.vector_store.similarity_search_with_score(query, k=k*2)
        semantic_docs = {doc.metadata['chunk_id']: (doc, 1.0 - score) 
                        for doc, score in semantic_results}
        
        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_scores_norm = bm25_scores / max_bm25
        
        # Combine scores
        combined_scores = {}
        for idx, (doc, sem_score) in enumerate(semantic_docs.items()):
            bm25_score = bm25_scores_norm[self.documents.index(sem_score[0])]
            combined_score = (1 - self.config.bm25_weight) * sem_score[1] + \
                           self.config.bm25_weight * bm25_score
            combined_scores[doc] = (sem_score[0], combined_score)
        
        # Sort by combined score
        sorted_docs = sorted(combined_scores.values(), key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in sorted_docs[:k]]

#==============================================================================
# SECTION 7: LLM SETUP (4-bit Quantized)
#==============================================================================

class QuantizedLLM:
    """Load and manage quantized LLM for generation."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def load(self):
        """Load the quantized LLM."""
        print(f"\n🤖 Loading LLM: {self.config.llm_model_name}")
        print("This may take a few minutes...")
        
        # Configure 4-bit quantization
        if self.config.use_quantization:
            print("Using 4-bit quantization (NF4) for memory efficiency...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_name,
            quantization_config=bnb_config if self.config.use_quantization else None,
            device_map="auto",
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if not self.config.use_quantization else None
        )
        
        print(f"✓ Model loaded successfully")
        print(f"✓ Model memory footprint: ~{self.get_memory_footprint():.2f} GB")
        
    def get_memory_footprint(self) -> float:
        """Get model memory usage in GB."""
        if self.model is None:
            return 0.0
        mem_bytes = self.model.get_memory_footprint()
        return mem_bytes / (1024 ** 3)
    
    def generate(self, prompt: str) -> str:
        """Generate response from the LLM."""
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode and extract only the new tokens
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if full_response.startswith(prompt):
            response = full_response[len(prompt):].strip()
        else:
            response = full_response.strip()
        
        return response

#==============================================================================
# SECTION 8: RAG PIPELINE
#==============================================================================

class RAGPipeline:
    """Complete RAG pipeline with retrieval and generation."""
    
    def __init__(self, vector_store_builder: VectorStoreBuilder, llm: QuantizedLLM, config: RAGConfig):
        self.vector_store_builder = vector_store_builder
        self.llm = llm
        self.config = config
        
    def create_prompt(self, query: str, retrieved_docs: List[Document]) -> str:
        """Create the prompt for the LLM with retrieved context."""
        
        # Format retrieved context with citations
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            chapter = doc.metadata.get('chapter_title', 'Unknown')
            section = doc.metadata.get('section_title', '')
            content = doc.metadata.get('original_text', doc.page_content)
            
            citation = f"[Source {i}: Chapter {doc.metadata['chapter_number']} - {chapter}"
            if section:
                citation += f", Section: {section}"
            citation += "]"
            
            context_parts.append(f"{citation}\n{content}")
        
        context_text = "\n\n".join(context_parts)
        
        # Create the prompt with Mistral instruction format
        prompt = f"""<s>[INST] You are a helpful physics tutor answering questions based ONLY on the NCERT Class 11 Physics textbook.

CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the provided context below
2. If the context doesn't contain the answer, say "I cannot answer this based on the provided textbook content"
3. For numerical problems, show step-by-step solution with formulas
4. Cite sources using [Source 1], [Source 2], etc.
5. Be precise and accurate - do not add information not in the context

CONTEXT FROM TEXTBOOK:
{context_text}

QUESTION: {query}

ANSWER (based strictly on the context above): [/INST]"""
        
        return prompt
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        return self.vector_store_builder.hybrid_search(query, k=self.config.top_k)
    
    def generate(self, query: str, retrieved_docs: List[Document]) -> Tuple[str, List[Document]]:
        """Generate answer using retrieved context."""
        prompt = self.create_prompt(query, retrieved_docs)
        answer = self.llm.generate(prompt)
        return answer, retrieved_docs
    
    def query(self, question: str) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve and generate."""
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(question)
        
        # Generate answer
        answer, docs = self.generate(question, retrieved_docs)
        
        # Format response
        response = {
            'question': question,
            'answer': answer,
            'retrieved_chunks': [
                {
                    'chunk_id': doc.metadata['chunk_id'],
                    'chapter': doc.metadata['chapter_title'],
                    'section': doc.metadata.get('section_title', ''),
                    'content_type': doc.metadata['content_type'],
                    'relevance_score': i + 1  # Rank
                }
                for i, doc in enumerate(docs)
            ],
            'num_chunks_retrieved': len(docs)
        }
        
        return response

#==============================================================================
# SECTION 9: EVALUATION METRICS
#==============================================================================

class RAGEvaluator:
    """Comprehensive evaluation of RAG system."""
    
    def __init__(self, rag_pipeline: RAGPipeline, config: RAGConfig):
        self.rag = rag_pipeline
        self.config = config
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def evaluate_retrieval(self, eval_questions: List[Dict]) -> Dict[str, float]:
        """Evaluate retrieval performance (Recall@k, MRR)."""
        print("\n📊 Evaluating Retrieval Performance...")
        
        recall_at_k = []
        mrr_scores = []
        
        for q in tqdm(eval_questions, desc="Retrieval Eval"):
            query = q['question']
            reference_chunk_id = q['reference_chunk_id']
            
            # Retrieve documents
            retrieved_docs = self.rag.retrieve(query)
            retrieved_chunk_ids = [doc.metadata['chunk_id'] for doc in retrieved_docs]
            
            # Recall@k: Is the reference chunk in top-k?
            is_retrieved = reference_chunk_id in retrieved_chunk_ids
            recall_at_k.append(1.0 if is_retrieved else 0.0)
            
            # MRR: Reciprocal rank of the reference chunk
            if is_retrieved:
                rank = retrieved_chunk_ids.index(reference_chunk_id) + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
        
        metrics = {
            f'Recall@{self.config.top_k}': np.mean(recall_at_k),
            'MRR': np.mean(mrr_scores),
            'Retrieval_Success_Rate': np.mean(recall_at_k)
        }
        
        return metrics
    
    def evaluate_answer_quality(self, eval_questions: List[Dict]) -> Dict[str, float]:
        """Evaluate answer quality using ROUGE and semantic similarity."""
        print("\n📊 Evaluating Answer Quality...")
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for q in tqdm(eval_questions[:20], desc="Answer Quality Eval (sample)"):  # Sample for speed
            query = q['question']
            reference_answer = q['reference_answer']
            
            # Generate answer
            response = self.rag.query(query)
            generated_answer = response['answer']
            
            # ROUGE scores
            scores = self.rouge_scorer.score(reference_answer, generated_answer)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        metrics = {
            'ROUGE-1': np.mean(rouge1_scores),
            'ROUGE-2': np.mean(rouge2_scores),
            'ROUGE-L': np.mean(rougeL_scores),
        }
        
        return metrics
    
    def evaluate_faithfulness(self, eval_questions: List[Dict]) -> Dict[str, float]:
        """Evaluate if answers are grounded in retrieved context."""
        print("\n📊 Evaluating Faithfulness (Context Grounding)...")
        
        # This is a simplified version - full implementation would use NLI models
        faithfulness_scores = []
        
        for q in tqdm(eval_questions[:10], desc="Faithfulness Eval (sample)"):
            query = q['question']
            
            # Generate answer
            response = self.rag.query(query)
            answer = response['answer']
            
            # Check for hallucination indicators
            hallucination_phrases = [
                "I cannot answer",
                "not in the context",
                "based on the provided textbook",
                "Source 1", "Source 2"  # Citation present
            ]
            
            has_grounding = any(phrase.lower() in answer.lower() for phrase in hallucination_phrases)
            faithfulness_scores.append(1.0 if has_grounding else 0.5)
        
        metrics = {
            'Faithfulness_Indicator': np.mean(faithfulness_scores),
        }
        
        return metrics
    
    def evaluate_numerical_accuracy(self, eval_questions: List[Dict]) -> Dict[str, float]:
        """Evaluate numerical problem solving accuracy."""
        print("\n📊 Evaluating Numerical Problem Accuracy...")
        
        numerical_questions = [q for q in eval_questions if q['type'] == 'numerical']
        
        if not numerical_questions:
            return {'Numerical_Accuracy': 0.0, 'Numerical_Count': 0}
        
        correct_count = 0
        total_count = 0
        
        for q in tqdm(numerical_questions[:10], desc="Numerical Eval (sample)"):
            query = q['question']
            reference_answer = q['reference_answer']
            
            # Generate answer
            response = self.rag.query(query)
            generated_answer = response['answer']
            
            # Extract numerical values (simplified)
            ref_numbers = self.extract_numbers(reference_answer)
            gen_numbers = self.extract_numbers(generated_answer)
            
            # Check if key numbers are present (tolerance-based)
            if ref_numbers and gen_numbers:
                # Check if any reference number is close to generated numbers
                for ref_num in ref_numbers[:3]:  # Check first 3 key numbers
                    for gen_num in gen_numbers:
                        if abs(ref_num - gen_num) / (abs(ref_num) + 1e-10) < 0.05:  # 5% tolerance
                            correct_count += 1
                            break
                total_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        metrics = {
            'Numerical_Accuracy': accuracy,
            'Numerical_Questions_Evaluated': total_count
        }
        
        return metrics
    
    def extract_numbers(self, text: str) -> List[float]:
        """Extract numerical values from text."""
        # Pattern to match numbers (including scientific notation)
        pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, text)
        
        numbers = []
        for match in matches:
            try:
                numbers.append(float(match))
            except:
                continue
        
        return numbers
    
    def run_full_evaluation(self, eval_questions: List[Dict]) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        print("\n" + "="*80)
        print("RUNNING COMPREHENSIVE EVALUATION")
        print("="*80)
        
        results = {}
        
        # 1. Retrieval Evaluation
        retrieval_metrics = self.evaluate_retrieval(eval_questions)
        results.update(retrieval_metrics)
        
        # 2. Answer Quality Evaluation
        answer_metrics = self.evaluate_answer_quality(eval_questions)
        results.update(answer_metrics)
        
        # 3. Faithfulness Evaluation
        faithfulness_metrics = self.evaluate_faithfulness(eval_questions)
        results.update(faithfulness_metrics)
        
        # 4. Numerical Accuracy Evaluation
        numerical_metrics = self.evaluate_numerical_accuracy(eval_questions)
        results.update(numerical_metrics)
        
        return results

#==============================================================================
# SECTION 10: MAIN EXECUTION PIPELINE
#==============================================================================

def main():
    """Main execution pipeline for the RAG system."""
    
    print("="*80)
    print("NCERT CLASS 11 PHYSICS RAG SYSTEM")
    print("Production-Ready Implementation")
    print("="*80)
    
    # Step 1: Load Data
    loader = DataLoader(config.data_path, config.eval_path)
    corpus, corpus_metadata = loader.load_corpus()
    eval_questions, eval_metadata = loader.load_evaluation_set()
    documents = loader.prepare_documents(corpus)
    
    # Step 2: Build Vector Store
    vs_builder = VectorStoreBuilder(config)
    vector_store = vs_builder.build(documents)
    
    # Step 3: Load LLM
    llm = QuantizedLLM(config)
    llm.load()
    
    # Step 4: Create RAG Pipeline
    rag = RAGPipeline(vs_builder, llm, config)
    
    # Step 5: Test Query (Demo)
    print("\n" + "="*80)
    print("DEMO QUERY")
    print("="*80)
    demo_question = "What is a unit in physics? Explain fundamental and derived units."
    print(f"Question: {demo_question}\n")
    
    response = rag.query(demo_question)
    print(f"Answer: {response['answer']}\n")
    print(f"Retrieved {response['num_chunks_retrieved']} chunks:")
    for chunk_info in response['retrieved_chunks']:
        print(f"  - {chunk_info['chapter']} ({chunk_info['content_type']})")
    
    # Step 6: Run Evaluation
    evaluator = RAGEvaluator(rag, config)
    eval_results = evaluator.run_full_evaluation(eval_questions)
    
    # Step 7: Print Results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print("\n📈 RETRIEVAL METRICS:")
    print(f"  Recall@{config.top_k}: {eval_results[f'Recall@{config.top_k}']:.2%}")
    print(f"  MRR (Mean Reciprocal Rank): {eval_results['MRR']:.3f}")
    print(f"  Retrieval Success Rate: {eval_results['Retrieval_Success_Rate']:.2%}")
    
    print("\n📝 ANSWER QUALITY METRICS:")
    print(f"  ROUGE-1: {eval_results['ROUGE-1']:.3f}")
    print(f"  ROUGE-2: {eval_results['ROUGE-2']:.3f}")
    print(f"  ROUGE-L: {eval_results['ROUGE-L']:.3f}")
    
    print("\n✅ FAITHFULNESS METRICS:")
    print(f"  Faithfulness Indicator: {eval_results['Faithfulness_Indicator']:.2%}")
    
    print("\n🔢 NUMERICAL ACCURACY:")
    print(f"  Numerical Accuracy: {eval_results['Numerical_Accuracy']:.2%}")
    print(f"  Questions Evaluated: {eval_results['Numerical_Questions_Evaluated']}")
    
    # Step 8: Interpretation
    print("\n" + "="*80)
    print("PERFORMANCE INTERPRETATION")
    print("="*80)
    
    recall = eval_results[f'Recall@{config.top_k}']
    rouge_l = eval_results['ROUGE-L']
    
    if recall >= 0.85:
        print("✅ EXCELLENT: Retrieval performance is strong (>85%)")
    elif recall >= 0.70:
        print("✓ GOOD: Retrieval performance is acceptable (70-85%)")
    else:
        print("⚠ NEEDS IMPROVEMENT: Retrieval performance is below target (<70%)")
    
    if rouge_l >= 0.4:
        print("✅ EXCELLENT: Answer quality is strong (ROUGE-L > 0.4)")
    elif rouge_l >= 0.3:
        print("✓ GOOD: Answer quality is acceptable (ROUGE-L 0.3-0.4)")
    else:
        print("⚠ NEEDS IMPROVEMENT: Answer quality could be better")
    
    # Step 9: Suggestions
    print("\n" + "="*80)
    print("SUGGESTED IMPROVEMENTS")
    print("="*80)
    
    suggestions = [
        "1. Fine-tune embedding model on physics domain for better retrieval",
        "2. Implement cross-encoder reranking for top-k results",
        "3. Add query expansion for complex questions",
        "4. Use larger LLM (13B/70B) for better reasoning on numerical problems",
        "5. Implement chain-of-thought prompting for multi-step problems",
        "6. Add context compression to fit more relevant info in prompt",
        "7. Use RAG-specific fine-tuned LLMs (e.g., RAG-tuned Mistral)",
        "8. Implement citation verification using NLI models",
        "9. Add metadata filtering (e.g., retrieve only from specific chapters)",
        "10. Create specialized retrievers for different question types"
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    print("\n" + "="*80)
    print("FUTURE SCALABILITY IDEAS")
    print("="*80)
    
    scalability = [
        "• Multi-tenancy: Support multiple textbooks/subjects",
        "• Streaming: Implement streaming responses for better UX",
        "• Caching: Cache frequent queries and embeddings",
        "• API: Wrap in FastAPI for production deployment",
        "• Monitoring: Add logging, metrics tracking (Prometheus/Grafana)",
        "• A/B Testing: Compare different retrieval strategies",
        "• User Feedback Loop: Collect ratings to improve system",
        "• Adaptive Retrieval: Adjust k based on query complexity",
        "• Multi-modal: Support diagram understanding with vision models",
        "• Personalization: User-specific context and learning history"
    ]
    
    for item in scalability:
        print(f"  {item}")
    
    print("\n✅ Evaluation Complete!")
    
    return rag, evaluator, eval_results

# Entry point
if __name__ == "__main__":
    # Note: In Colab, run main() after uploading files
    print("⚠️  Before running main(), please:")
    print("1. Upload Data.json and Evaluation_Set.json (or Evaluation Set.json) to /content/")
    print("2. Run: install_dependencies()")
    print("3. Run: rag_pipeline, evaluator, results = main()")
    print("\nOr simply run: python ncert_physics_rag_system.py")
