# RRF score
def rrf(rank,k_rrf): 
      return 1.0 / (k_rrf + rank)


def build_bm25_index(db):
    """Build BM25 index from database articles"""
    from rank_bm25 import BM25Okapi
    import pickle
    import os
    from datetime import datetime, timezone
    from src.storage import ArticleDB
    
    # Get recent articles for BM25 index
    articles = db.get_recent_articles(limit=1000, hours_back=24*30)  # 30 days to include fixture data
    
    if not articles:
        print("No articles found for BM25 index")
        return
    
    # Build corpus
    corpus = []
    bm25_articles = []
    
    for article in articles:
        text = f"{article.title} {article.description or ''} {article.content[:500]}"
        # Use same tokenization as MultiRAGRetriever
        from src.retrieval import _tokenize
        doc_tokens = _tokenize(text)
        corpus.append(doc_tokens)
        bm25_articles.append(article)
    
    # Build BM25 index
    bm25_index = BM25Okapi(corpus)
    
    # Save to cache
    cache_path = "db/bm25_cache.pkl"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache_data = {
        'index': bm25_index,
        'articles': bm25_articles,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"BM25 index built with {len(bm25_articles)} articles")


# ================ Approval Agent Helper Functions ================

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import re

from ..config import ApprovalConfig, LLMConfig
from ..embeddings import EmbeddingSystem
from ..storage import ArticleDB


def now_iso() -> str:
    """Get current UTC time as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def basic_title_consistency(submitted_title: Optional[str], synthesized_content: str) -> float:
    """Compute basic title consistency score."""
    if not submitted_title or not synthesized_content:
        return 0.0
    title = submitted_title.strip().lower()
    content = synthesized_content.strip().lower()
    if not title or not content:
        return 0.0
    if title in content:
        return 1.0
    toks_t = set(title.split())
    toks_c = set(content.split())
    if not toks_t or not toks_c:
        return 0.0
    overlap = len(toks_t.intersection(toks_c)) / max(1, len(toks_t))
    return min(1.0, max(0.0, overlap))


def compute_evidence_quality(evidence_fetch_outcomes: List[Dict], cfg: ApprovalConfig) -> float:
    """Compute evidence quality score based on number of usable evidence sources."""
    if not evidence_fetch_outcomes:
        return 0.0
    usable = [e for e in evidence_fetch_outcomes if (e.get("extracted_length", 0) or 0) >= cfg.min_extract_length]
    if not usable:
        return 0.0
    score = 0.5 if len(usable) >= 1 else 0.0
    if len(usable) >= 2:
        score = 0.8
    if len(usable) >= 3:
        score = 1.0
    return score


def compute_coherence(embeddings: EmbeddingSystem, evidence_fetch_outcomes: List[Dict], cfg: ApprovalConfig) -> float:
    """Compute coherence using embeddings for pairwise similarity between evidence texts."""
    usable = [e for e in evidence_fetch_outcomes if (e.get("extracted_length", 0) or 0) >= cfg.min_coherence_length]
    if len(usable) < 2:
        return 0.0
    
    texts = [e.get("extracted_text", "").strip() for e in usable]
    texts = [t for t in texts if t]  # filter empty
    
    if len(texts) < 2:
        return 0.0
    
    try:
        # Get embeddings for all texts
        embeddings_list = []
        for text in texts:
            emb = embeddings.get_embedding(text[:2000])  # truncate for efficiency
            if emb is not None:
                embeddings_list.append(emb)
        
        if len(embeddings_list) < 2:
            return 0.0
        
        # Compute pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings_list)):
            for j in range(i + 1, len(embeddings_list)):
                # Cosine similarity
                dot_product = sum(a * b for a, b in zip(embeddings_list[i], embeddings_list[j]))
                norm_i = sum(a * a for a in embeddings_list[i]) ** 0.5
                norm_j = sum(b * b for b in embeddings_list[j]) ** 0.5
                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (norm_i * norm_j)
                    similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Return average pairwise similarity
        return sum(similarities) / len(similarities)
    
    except Exception:
        return 0.0


def synthesize_content(content_raw: Optional[str], evidence_fetch_outcomes: List[Dict]) -> str:
    """Synthesize content from raw content or evidence."""
    if content_raw and len(content_raw.strip()) >= 400:
        return content_raw.strip()
    
    parts: List[str] = []
    for e in evidence_fetch_outcomes:
        text = (e.get("extracted_text") or "").strip()
        if text:
            parts.append(text[:600])
        if len("\n\n".join(parts)) >= 900:
            break
    return ("\n\n".join(parts))[:4000]


def dup_max_cosine(embeddings: EmbeddingSystem, text: str) -> float:
    """Compute maximum cosine similarity with existing articles."""
    if not text or not isinstance(text, str):
        return 0.0
    try:
        res = embeddings.semantic_search(text, k=1)
        if not res:
            return 0.0
        return float(res[0][1])
    except Exception:
        return 0.0


def get_duplicate_preview(embeddings: EmbeddingSystem, text: str, k: int = 3) -> List[Dict]:
    """Get preview of most similar articles for duplicate detection."""
    if not text or not isinstance(text, str):
        return []
    try:
        res = embeddings.semantic_search(text, k=k)
        if not res:
            return []
        
        preview = []
        for article_id, score in res:
            preview.append({
                "article_id": article_id,
                "similarity_score": float(score),
                "title": "",  # Could be populated from DB if needed
            })
        return preview
    except Exception:
        return []


def extract_title_from_evidence(evidence_fetch_outcomes: List[Dict]) -> Optional[str]:
    """Extract the best title from evidence texts."""
    titles = []
    for e in evidence_fetch_outcomes:
        title = e.get("extracted_title", "").strip()
        if title and len(title) >= 10:  # reasonable minimum
            titles.append(title)
    
    if not titles:
        return None
    
    # Return the longest title (often most descriptive)
    return max(titles, key=len)


def compute_title_alignment(submitted_title: str, extracted_title: Optional[str]) -> float:
    """Compute alignment between submitted and extracted titles."""
    if not submitted_title or not extracted_title:
        return 0.0
    
    submitted = submitted_title.strip().lower()
    extracted = extracted_title.strip().lower()
    
    if not submitted or not extracted:
        return 0.0
    
    # Exact match
    if submitted == extracted:
        return 1.0
    
    # Substring match
    if submitted in extracted or extracted in submitted:
        return 0.8
    
    # Word overlap
    submitted_words = set(submitted.split())
    extracted_words = set(extracted.split())
    
    if not submitted_words or not extracted_words:
        return 0.0
    
    overlap = len(submitted_words.intersection(extracted_words))
    union = len(submitted_words.union(extracted_words))
    
    if union == 0:
        return 0.0
    
    jaccard = overlap / union
    return min(1.0, jaccard * 2)  # Scale up for better scoring


def llm_content_analysis(content: str, evidence_texts: List[str], llm_config: Optional[LLMConfig] = None) -> Dict[str, Any]:
    """Perform LLM-based content analysis and fact-checking."""
    if not content or not evidence_texts:
        return {"fact_check_score": 0.0, "contradictions": [], "synthesis_quality": 0.0}
    
    contradictions = []
    fact_check_score = 0.7
    synthesis_quality = 0.6
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        if llm_config is None:
            from ..config import LLMConfig
            llm_config = LLMConfig()
        
        tokenizer = AutoTokenizer.from_pretrained(llm_config.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            llm_config.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        evidence_context = "\n\n".join([f"Evidence {i+1}: {text[:300]}" for i, text in enumerate(evidence_texts[:2])])
        prompt = f"Content: {content[:400]} Evidence: {evidence_context[:400]} Score:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        llm_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Parse scores from response
        if any(s in llm_response for s in ["0.8", "0.9", "1.0"]):
            fact_check_score = 0.8
        elif any(s in llm_response for s in ["0.6", "0.7"]):
            fact_check_score = 0.6
        elif any(s in llm_response for s in ["0.4", "0.5"]):
            fact_check_score = 0.4
        elif any(s in llm_response for s in ["0.0", "0.1", "0.2", "0.3"]):
            fact_check_score = 0.2
        
        # Check for contradictions
        if any(word in llm_response.lower() for word in ["contradiction", "inconsistent", "disagree", "conflict", "wrong"]):
            contradictions.append("LLM detected contradictions")
            fact_check_score = min(fact_check_score, 0.4)
        
        # Adjust based on content quality
        if len(content) < 100:
            contradictions.append("Content too short")
            fact_check_score = min(fact_check_score, 0.3)
        
        if len(evidence_texts) < 2:
            contradictions.append("Insufficient evidence")
            fact_check_score = min(fact_check_score, 0.4)
        
        if len(content) > 500 and len(evidence_texts) >= 2:
            fact_check_score = min(1.0, fact_check_score + 0.1)
            synthesis_quality = min(1.0, synthesis_quality + 0.1)
        
    except ImportError:
        contradictions.append("LLM not available - using fallback")
        fact_check_score = 0.6
        synthesis_quality = calculate_synthesis_quality(content, evidence_texts)
        
    except Exception as e:
        contradictions.append(f"LLM error: {str(e)}")
        fact_check_score = 0.3
        synthesis_quality = calculate_synthesis_quality(content, evidence_texts)
    
    return {
        "fact_check_score": fact_check_score,
        "contradictions": contradictions,
        "synthesis_quality": synthesis_quality,
        "confidence": fact_check_score * synthesis_quality
    }


def calculate_synthesis_quality(content: str, evidence_texts: List[str]) -> float:
    """Calculate synthesis quality based on content structure and evidence coverage."""
    if not content:
        return 0.0
    
    length_score = min(1.0, len(content.strip()) / 1000.0)
    sentences = len(re.findall(r'[.!?]+', content))
    structure_score = min(1.0, sentences / 10.0)
    evidence_coverage = min(1.0, len(evidence_texts) / 3.0)
    
    return min(1.0, max(0.0, 0.4 * length_score + 0.3 * structure_score + 0.3 * evidence_coverage))


def adaptive_threshold_calculation(db: ArticleDB) -> Dict[str, float]:
    """Calculate adaptive thresholds based on historical performance."""
    try:
        recent_decisions = db.get_recent_submission_decisions(limit=100)
        if not recent_decisions:
            return {"tau_auto": 0.80, "tau_review": 0.55, "tau_dup": 0.95, "tau_title_align": 0.75}
        
        # Simplified: adjust based on recent performance (placeholder logic)
        auto_accuracy = 0.8  # Would calculate from actual data
        
        if auto_accuracy > 0.85:
            tau_auto = 0.75
        elif auto_accuracy < 0.7:
            tau_auto = 0.85
        else:
            tau_auto = 0.80
        
        return {
            "tau_auto": tau_auto,
            "tau_review": max(0.5, tau_auto - 0.25),
            "tau_dup": 0.95,
            "tau_title_align": 0.75
        }
    except Exception:
        return {"tau_auto": 0.80, "tau_review": 0.55, "tau_dup": 0.95, "tau_title_align": 0.75}


def calculate_confidence_level(signals: Dict[str, Any], synthesized_content: str) -> float:
    """Calculate overall confidence level for the submission."""
    evidence_quality = signals.get("evidence_quality", 0.0)
    coherence = signals.get("cross_evidence_coherence", 0.0)
    title_alignment = signals.get("title_alignment", 0.0)
    content_factor = min(1.0, len(synthesized_content) / 1000.0)
    
    return min(1.0, max(0.0, 
        0.3 * evidence_quality + 
        0.25 * coherence + 
        0.2 * title_alignment + 
        0.25 * content_factor
    ))
