from __future__ import annotations

import json
import uuid
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import numpy as np
from langgraph.graph import StateGraph, END

from ..storage import ArticleDB
from ..data_models import Article, Source, SourceCategory, ContentType, SubmissionStatus, DecisionType, ArticleProvenance, SubmissionModel
from ..embeddings import EmbeddingSystem
from ..config import ApprovalConfig, JOURNALIST_DEFAULT_CREDIBILITY, LLMConfig
from ..utils.content_extract import fetch_and_extract

@dataclass
class ApprovalOutcome:
    status: SubmissionStatus
    decision_type: Optional[DecisionType]
    validity_score: float
    reasons: List[str]
    blockers: List[str]
    final_title: str
    synthesized_content: str
    signals: Dict


@dataclass
class AgentState:
    submission: SubmissionModel
    signals: Dict
    blockers: List[str]
    reasons: List[str]
    synthesized_content: str
    validity_score: float
    status: Optional[SubmissionStatus]
    decision_type: Optional[DecisionType]
    # fields for enhanced agentic behavior
    llm_analysis: Optional[Dict] = None
    adaptive_thresholds: Optional[Dict] = None
    confidence_level: float = 0.0


def apply_reviewer_decision(db: ArticleDB,
                            submission_id: str,
                            decision: str,
                            *,
                            final_title: Optional[str] = None,
                            notes: Optional[str] = None) -> bool:
    """Apply a human reviewer decision for a submission in JEEDS_REVIEW"""
    sub = db.get_submission(submission_id)
    if not sub:
        return False

    if notes:
        db.set_submission_review_notes(submission_id, notes)

    if decision == "approve":
        # Promote to Article with reviewer decision
        submitted_title: str = (sub.submitted_title or "").strip()
        title = (final_title or submitted_title or ("Journalist Report " + submission_id[:8]))
        evidence_urls: List[str] = sub.evidence_urls or []

        src = Source(
            id="journalist",
            name="Journalist Report",
            url="",
            category=SourceCategory.GENERAL,
            credibility_score=JOURNALIST_DEFAULT_CREDIBILITY,
        )
        article = Article(
            id=str(uuid.uuid4()),
            title=title,
            content=(sub.synthesized_content or ""),
            description=sub.description,
            url="#",
            source=src,
            published_at=datetime.now(timezone.utc),
            content_type=ContentType.FACTUAL,
            provenance_source=ArticleProvenance.JOURNALIST_REPORT,
            decision_type=DecisionType.REVIEWED,
            evidence_urls=evidence_urls,
        )
        ok = db.save_article(article)
        db.update_submission_decision(
            submission_id,
            status=SubmissionStatus.APPROVED.value,
            decision_type=DecisionType.REVIEWED.value,
            validity_score=sub.validity_score,
            reasons=sub.reasons,
            blockers=sub.blockers,
            decided_at_iso=datetime.now(timezone.utc).isoformat(),
        )
        return ok

    if decision == "reject":
        return db.update_submission_decision(
            submission_id,
            status=SubmissionStatus.REJECTED.value,
            decision_type=None,
            validity_score=sub.validity_score,
            reasons=sub.reasons,
            blockers=sub.blockers,
            decided_at_iso=datetime.now(timezone.utc).isoformat(),
        )

    if decision == "request_more_evidence":
        return db.update_submission_decision(
            submission_id,
            status=SubmissionStatus.PENDING.value,
            decision_type=None,
            validity_score=sub.validity_score,
            reasons=sub.reasons,
            blockers=sub.blockers,
            decided_at_iso=datetime.now(timezone.utc).isoformat(),
        )

    return False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _basic_title_consistency(submitted_title: Optional[str], synthesized_content: str) -> float:
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


def _compute_evidence_quality(evidence_fetch_outcomes: List[Dict], cfg: ApprovalConfig) -> float:
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


def _compute_coherence_stub(evidence_fetch_outcomes: List[Dict], cfg: ApprovalConfig) -> float:
    usable = [e for e in evidence_fetch_outcomes if (e.get("extracted_length", 0) or 0) >= cfg.min_extract_length]
    if len(usable) >= 2:
        return 0.6
    return 0.0


def _compute_coherence_real(embeddings: EmbeddingSystem, evidence_fetch_outcomes: List[Dict], cfg: ApprovalConfig) -> float:
    """Compute real coherence using embeddings for pairwise similarity between evidence texts."""
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


def _synthesize_content_stub(content_raw: Optional[str], evidence_fetch_outcomes: List[Dict]) -> Tuple[str, Optional[str]]:
    if content_raw and len(content_raw.strip()) >= 400:
        return content_raw.strip(), None
    parts: List[str] = []
    for e in evidence_fetch_outcomes:
        text = (e.get("extracted_text") or "").strip()
        if text:
            parts.append(text[:600])
        if len("\n\n".join(parts)) >= 900:
            break
    synthesized = ("\n\n".join(parts))[:4000]
    return synthesized, None


def _dup_max_cosine(embeddings: EmbeddingSystem, text: str) -> float:
    if not text or not isinstance(text, str):
        return 0.0
    try:
        res = embeddings.semantic_search(text, k=1)
        if not res:
            return 0.0
        return float(res[0][1])
    except Exception:
        return 0.0


def _get_duplicate_preview(embeddings: EmbeddingSystem, text: str, k: int = 3) -> List[Dict]:
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


def _extract_title_from_evidence(evidence_fetch_outcomes: List[Dict]) -> Optional[str]:
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


def _compute_title_alignment(submitted_title: str, extracted_title: Optional[str]) -> float:
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



def _llm_content_analysis(content: str, evidence_texts: List[str], embeddings: EmbeddingSystem, llm_config: Optional[LLMConfig] = None) -> Dict:
    """Perform actual LLM-based content analysis and fact-checking."""
    if not content or not evidence_texts:
        return {"fact_check_score": 0.0, "contradictions": [], "synthesis_quality": 0.0}
    
    contradictions = []
    fact_check_score = 0.8  # Base score
    
    try:
        # Use Hugging Face Transformers directly
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Use provided config or default
        if llm_config is None:
            from src.config import LLMConfig
            llm_config = LLMConfig()
        
        # Load model and tokenizer (cached after first use)
        model_name = llm_config.model_name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not available
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Prepare evidence context
        evidence_context = "\n\n".join([f"Evidence {i+1}: {text[:500]}" for i, text in enumerate(evidence_texts[:3])])
        
        # Fact-checking prompt
        fact_check_prompt = f"""
        Analyze the following synthesized content for factual accuracy against the provided evidence.
        
        SYNTHESIZED CONTENT:
        {content[:1500]}
        
        EVIDENCE:
        {evidence_context}
        
        Please provide:
        1. A fact-check score (0.0-1.0) based on how well the content aligns with the evidence
        2. Any contradictions or inconsistencies you find
        3. A synthesis quality score (0.0-1.0) based on how well the content synthesizes the evidence
        
        Respond in JSON format:
        {{
            "fact_check_score": 0.8,
            "contradictions": ["specific contradiction 1", "specific contradiction 2"],
            "synthesis_quality": 0.7,
            "reasoning": "brief explanation"
        }}
        """
        
        # Tokenize and generate
        inputs = tokenizer(fact_check_prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=llm_config.max_tokens,
                temperature=llm_config.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        llm_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            parsed_result = json.loads(json_match.group())
            fact_check_score = float(parsed_result.get("fact_check_score", 0.5))
            contradictions = parsed_result.get("contradictions", [])
            synthesis_quality = float(parsed_result.get("synthesis_quality", 0.5))
        else:
            # Fallback parsing
            fact_check_score = 0.5
            synthesis_quality = 0.5
            contradictions = ["Failed to parse LLM response"]
        
    except ImportError:
        # Fallback if transformers not available
        contradictions.append("Transformers not available - using fallback")
        fact_check_score = 0.6
        synthesis_quality = _calculate_synthesis_quality(content, evidence_texts)
        
    except Exception as e:
        contradictions.append(f"LLM analysis error: {str(e)}")
        fact_check_score = 0.3
        synthesis_quality = _calculate_synthesis_quality(content, evidence_texts)
    
    return {
        "fact_check_score": fact_check_score,
        "contradictions": contradictions,
        "synthesis_quality": synthesis_quality,
        "confidence": fact_check_score * synthesis_quality
    }


def _calculate_synthesis_quality(content: str, evidence_texts: List[str]) -> float:
    """Calculate synthesis quality based on content structure and evidence coverage."""
    if not content:
        return 0.0
    
    # Base quality from content length and structure
    content_length = len(content.strip())
    length_score = min(1.0, content_length / 1000.0)
    
    # Check for good structure (paragraphs, sentences)
    import re
    sentences = len(re.findall(r'[.!?]+', content))
    paragraphs = len(content.split('\n\n'))
    
    structure_score = min(1.0, sentences / 10.0)  # Expect at least 10 sentences
    
    # Evidence coverage score
    evidence_coverage = min(1.0, len(evidence_texts) / 3.0)  # Expect 3+ evidence sources
    
    # Combine scores
    synthesis_quality = (
        0.4 * length_score +
        0.3 * structure_score +
        0.3 * evidence_coverage
    )
    
    return min(1.0, max(0.0, synthesis_quality))


def _adaptive_threshold_calculation(db: ArticleDB, submission_type: str = "journalist_report") -> Dict:
    """Calculate adaptive thresholds based on historical performance."""
    try:
        # Get recent submission decisions for learning
        # This is a simplified version - in production, you'd query actual historical data
        recent_decisions = db.get_recent_submission_decisions(limit=100)
        
        if not recent_decisions:
            # Return default thresholds
            return {
                "tau_auto": 0.80,
                "tau_review": 0.55,
                "tau_dup": 0.95,
                "tau_title_align": 0.75
            }
        
        # Analyze accuracy of auto-approvals vs human decisions
        auto_approved = [d for d in recent_decisions if d.get("decision_type") == "auto_approved"]
        human_reviewed = [d for d in recent_decisions if d.get("decision_type") == "reviewed"]
        
        # Calculate accuracy metrics
        auto_accuracy = 0.8  # Placeholder - would calculate from actual data
        human_accuracy = 0.9  # Placeholder
        
        # Adjust thresholds based on performance
        if auto_accuracy > 0.85:
            # High confidence in auto-approval, can lower threshold
            tau_auto = 0.75
        elif auto_accuracy < 0.7:
            # Low confidence, raise threshold
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
        # Fallback to default thresholds
        return {
            "tau_auto": 0.80,
            "tau_review": 0.55,
            "tau_dup": 0.95,
            "tau_title_align": 0.75
        }


def _calculate_confidence_level(state: AgentState) -> float:
    """Calculate overall confidence level for the submission."""
    signals = state.signals
    
    # Weight different factors
    evidence_quality = signals.get("evidence_quality", 0.0)
    coherence = signals.get("cross_evidence_coherence", 0.0)
    title_alignment = signals.get("title_alignment", 0.0)
    content_length = len(state.synthesized_content)
    
    # Normalize content length factor
    content_factor = min(1.0, content_length / 1000.0)
    
    # Weighted confidence calculation
    confidence = (
        0.3 * evidence_quality +
        0.25 * coherence +
        0.2 * title_alignment +
        0.25 * content_factor
    )
    
    return min(1.0, max(0.0, confidence))


# ---------------- Agent Nodes ----------------
def n_schema_validate(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    sub = state.submission
    submitted_title = (sub.submitted_title or "").strip()
    evidence_urls = sub.evidence_urls or []
    content_raw = sub.content_raw or ""
    
    # Validate title
    if not submitted_title:
        state.blockers.append("missing_title")
    elif len(submitted_title) < 10:
        state.blockers.append("title_too_short")
    elif len(submitted_title) > 200:
        state.blockers.append("title_too_long")
    
    # Validate evidence URLs
    if evidence_urls:
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        
        invalid_urls = [url for url in evidence_urls if not url_pattern.match(url)]
        if invalid_urls:
            state.blockers.append("invalid_urls")
            state.reasons.append(f"Invalid URLs: {invalid_urls}")
    
    # Validate content
    if not evidence_urls and not content_raw.strip():
        state.blockers.append("no_evidence_and_no_content")
    elif content_raw and len(content_raw.strip()) < 50:
        state.blockers.append("content_too_short")
    
    return state

# run the text extraction for the evidence urls in parallel
# returns the outcomes of the text extraction
def n_fetch_evidence(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    sub = state.submission
    evidence_urls: List[str] = sub.evidence_urls or []
    outcomes: List[Dict] = []
    
    if evidence_urls:
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(evidence_urls))) as ex:
                futs = [ex.submit(fetch_and_extract, u, timeout=cfg.fetch_timeout_s, retries=cfg.fetch_retries, max_redirects=cfg.fetch_max_redirects) for u in evidence_urls]
                for i, f in enumerate(futs):
                    try:
                        result = f.result(timeout=cfg.fetch_timeout_s + 5)  # Add buffer
                        outcomes.append(result)
                    except Exception as e:
                        # Include the original URL in error
                        original_url = evidence_urls[i] if i < len(evidence_urls) else "unknown"
                        outcomes.append({
                            "url": original_url,
                            "status_code": None,
                            "extracted_text": "",
                            "error": str(e),
                            "extracted_length": 0,
                            "boilerplate_ratio": 1.0,
                            "num_links": 0.0,
                            "extractor_used": "error"
                        })
                        print(f"Error fetching evidence from {original_url}: {e}")
        except Exception as e:
            print(f"Error in evidence fetching: {e}")
            state.blockers.append("evidence_fetch_error")
            state.reasons.append(f"Evidence fetching failed: {str(e)}")
    
    state.signals["evidence_fetch"] = outcomes
    return state


def n_synthesize(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    sub = state.submission
    content_raw: Optional[str] = sub.content_raw
    outcomes: List[Dict] = state.signals.get("evidence_fetch", [])
    synthesized, _ = _synthesize_content_stub(content_raw, outcomes)
    if not synthesized:
        synthesized = (sub.description or "").strip()
    state.synthesized_content = synthesized
    return state


def n_signals(state: AgentState, cfg: ApprovalConfig, embeddings: EmbeddingSystem) -> AgentState:
    sub = state.submission
    submitted_title: str = (sub.submitted_title or "").strip()
    outcomes: List[Dict] = state.signals.get("evidence_fetch", [])
    
    # Compute evidence quality
    state.signals["evidence_quality"] = _compute_evidence_quality(outcomes, cfg)
    
    # Compute real coherence using embeddings
    state.signals["cross_evidence_coherence"] = _compute_coherence_real(embeddings, outcomes, cfg)
    
    # Extract title from evidence and compute alignment
    extracted_title = _extract_title_from_evidence(outcomes)
    state.signals["extracted_title"] = extracted_title
    state.signals["title_alignment"] = _compute_title_alignment(submitted_title, extracted_title)
    
    # Legacy title consistency (keep for backward compatibility)
    state.signals["title_consistency"] = _basic_title_consistency(submitted_title, state.synthesized_content)
    
    return state


def n_llm_analysis(state: AgentState, cfg: ApprovalConfig, embeddings: EmbeddingSystem, llm_config: Optional[LLMConfig] = None) -> AgentState:
    """Perform LLM-based content analysis and fact-checking."""
    outcomes: List[Dict] = state.signals.get("evidence_fetch", [])
    evidence_texts = [e.get("extracted_text", "") for e in outcomes if e.get("extracted_text")]
    
    # Perform LLM analysis
    llm_analysis = _llm_content_analysis(state.synthesized_content, evidence_texts, embeddings, llm_config)
    state.llm_analysis = llm_analysis
    
    # Update signals with LLM insights
    state.signals["fact_check_score"] = llm_analysis["fact_check_score"]
    state.signals["synthesis_quality"] = llm_analysis["synthesis_quality"]
    state.signals["llm_confidence"] = llm_analysis["confidence"]
    
    # Add LLM-based reasons
    if llm_analysis["fact_check_score"] >= 0.8:
        state.reasons.append("+llm_fact_check")
    if llm_analysis["synthesis_quality"] >= 0.7:
        state.reasons.append("+llm_synthesis")
    
    return state


def n_adaptive_thresholds(state: AgentState, cfg: ApprovalConfig, db: ArticleDB) -> AgentState:
    """Calculate adaptive thresholds based on historical performance."""
    adaptive_thresholds = _adaptive_threshold_calculation(db, "journalist_report")
    state.adaptive_thresholds = adaptive_thresholds
    
    # Update confidence level
    state.confidence_level = _calculate_confidence_level(state)
    state.signals["confidence_level"] = state.confidence_level
    
    return state


def n_evidence_gate(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    """Conditional branching based on evidence quality."""
    evidence_quality = state.signals.get("evidence_quality", 0.0)
    confidence_level = state.confidence_level
    
    # Use adaptive thresholds if available
    thresholds = state.adaptive_thresholds or {}
    tau_auto = thresholds.get("tau_auto", cfg.tau_auto)
    tau_review = thresholds.get("tau_review", cfg.tau_review)
    
    # Gate based on evidence quality and confidence
    if evidence_quality < 0.3 or confidence_level < 0.4:
        state.signals["evidence_gate"] = "human_review"  # Skip to human review
    elif evidence_quality < 0.6 or confidence_level < 0.7:
        state.signals["evidence_gate"] = "additional_verification"  # Extra checks
    else:
        state.signals["evidence_gate"] = "continue_automated"  # Normal flow
    
    return state


def n_deduplicate(state: AgentState, cfg: ApprovalConfig, embeddings: EmbeddingSystem) -> AgentState:
    dup_cos = _dup_max_cosine(embeddings, state.synthesized_content)
    state.signals["dup_max_cosine_vs_corpus"] = dup_cos
    
    # Add duplicate preview for review
    duplicate_preview = _get_duplicate_preview(embeddings, state.synthesized_content, k=3)
    state.signals["duplicate_preview"] = duplicate_preview
    
    if dup_cos >= cfg.tau_dup:
        state.blockers.append("exact_duplicate")
    return state


def n_score_and_decide(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    # Use improved title alignment instead of basic consistency
    title_score = state.signals.get("title_alignment", state.signals.get("title_consistency", 0.0))
    
    # Get LLM analysis scores if available
    llm_fact_check = state.signals.get("fact_check_score", 0.0)
    llm_synthesis = state.signals.get("synthesis_quality", 0.0)
    
    # Enhanced scoring with LLM components
    positives = (
        cfg.w_evidence_quality * state.signals.get("evidence_quality", 0.0) +
        cfg.w_coherence * state.signals.get("cross_evidence_coherence", 0.0) +
        cfg.w_title_consistency * title_score +
        cfg.w_content_quality * (1.0 if len(state.synthesized_content) >= 800 else 0.5 if len(state.synthesized_content) >= 300 else 0.0) +
        0.1 * llm_fact_check +  # Add LLM fact-checking bonus
        0.1 * llm_synthesis     # Add LLM synthesis bonus
    )
    
    dup_cos = state.signals.get("dup_max_cosine_vs_corpus", 0.0)
    dup_pen = min(1.0, max(0.0, (dup_cos - 0.80) / 0.15)) if dup_cos > 0.80 else 0.0
    penalty = cfg.w_penalties_dup_safety * max(dup_pen, 0.0)
    
    # Apply confidence level adjustment
    confidence_adjustment = state.confidence_level * 0.1  # Boost high-confidence submissions
    
    state.validity_score = max(0.0, min(1.0, positives + penalty + confidence_adjustment))

    # Enhanced reasoning with new signals
    if state.signals.get("evidence_quality", 0.0) >= 0.8:
        state.reasons.append("+evidence_quality")
    if state.signals.get("cross_evidence_coherence", 0.0) >= 0.5:
        state.reasons.append("+coherence")
    if title_score >= cfg.tau_title_align:
        state.reasons.append("+title_alignment")
    if llm_fact_check >= 0.8:
        state.reasons.append("+llm_fact_check")
    if llm_synthesis >= 0.7:
        state.reasons.append("+llm_synthesis")
    if state.confidence_level >= 0.8:
        state.reasons.append("+high_confidence")
    if dup_pen > 0.0:
        state.reasons.append("-near_duplicate")
    
    # Add extracted title to signals for review
    extracted_title = state.signals.get("extracted_title")
    if extracted_title:
        state.signals["suggested_final_title"] = extracted_title

    # Use adaptive thresholds if available
    thresholds = state.adaptive_thresholds or {}
    tau_auto = thresholds.get("tau_auto", cfg.tau_auto)
    tau_review = thresholds.get("tau_review", cfg.tau_review)

    if state.blockers:
        state.status = SubmissionStatus.REJECTED
        state.decision_type = None
    elif state.validity_score >= tau_auto:
        state.status = SubmissionStatus.AUTO_APPROVED
        state.decision_type = DecisionType.AUTO_APPROVED
    elif state.validity_score >= tau_review:
        state.status = SubmissionStatus.NEEDS_REVIEW
        state.decision_type = None
    else:
        state.status = SubmissionStatus.REJECTED
        state.decision_type = None
    return state


def n_promote_if_auto(state: AgentState, db: ArticleDB) -> AgentState:
    if state.status != SubmissionStatus.AUTO_APPROVED:
        return state
    sub = state.submission
    submitted_title: str = (sub.submitted_title or "").strip()
    evidence_urls: List[str] = sub.evidence_urls or []
    
    # Use suggested title if available, otherwise submitted title
    final_title = state.signals.get("suggested_final_title") or submitted_title or ("Journalist Report " + str(uuid.uuid4())[:8])
    
    article_id = str(uuid.uuid4())
    src = Source(
        id="journalist",
        name="Journalist Report",
        url="",
        category=SourceCategory.GENERAL,
        credibility_score=JOURNALIST_DEFAULT_CREDIBILITY,
    )
    article = Article(
        id=article_id,
        title=final_title,
        content=state.synthesized_content or (submitted_title or ""),
        description=sub.description,
        url="#",
        source=src,
        published_at=datetime.now(timezone.utc),
        content_type=ContentType.FACTUAL,
        provenance_source=ArticleProvenance.JOURNALIST_REPORT,
        decision_type=DecisionType.AUTO_APPROVED,
        evidence_urls=evidence_urls,
    )
    db.save_article(article)
    return state


def run_approval_agent(db: ArticleDB, embeddings: EmbeddingSystem, submission_id: str, config: Optional[ApprovalConfig] = None, llm_config: Optional[LLMConfig] = None) -> ApprovalOutcome:
    cfg = config or ApprovalConfig()
    sub = db.get_submission(submission_id)
    if not sub:
        raise ValueError(f"Submission not found: {submission_id}")

    state = AgentState(
        submission=sub,
        signals={},
        blockers=[],
        reasons=[],
        synthesized_content="",
        validity_score=0.0,
        status=None,
        decision_type=None,
    )


    def to_dict(s: AgentState) -> dict:
        return {
            "submission": s.submission,
            "signals": s.signals,
            "blockers": s.blockers,
            "reasons": s.reasons,
            "synthesized_content": s.synthesized_content,
            "validity_score": s.validity_score,
            "status": s.status,
            "decision_type": s.decision_type,
            "llm_analysis": s.llm_analysis,
            "adaptive_thresholds": s.adaptive_thresholds,
            "confidence_level": s.confidence_level,
        }

    def from_dict(d: dict) -> AgentState:
        return AgentState(
            submission=d.get("submission"),  # Should always be SubmissionModel
            signals=d.get("signals", {}),
            blockers=d.get("blockers", []),
            reasons=d.get("reasons", []),
            synthesized_content=d.get("synthesized_content", ""),
            validity_score=d.get("validity_score", 0.0),
            status=d.get("status"),
            decision_type=d.get("decision_type"),
            llm_analysis=d.get("llm_analysis"),
            adaptive_thresholds=d.get("adaptive_thresholds"),
            confidence_level=d.get("confidence_level", 0.0),
        )

    g = StateGraph(dict)

    def wrap(fn):
        def _inner(d: dict) -> dict:
            s = from_dict(d)
            s2 = fn(s)
            return to_dict(s2)
        return _inner

    # bind cfg/embeddings/db via closures
    g.add_node("schema", wrap(lambda s: n_schema_validate(s, cfg)))
    g.add_node("fetch", wrap(lambda s: n_fetch_evidence(s, cfg)))
    g.add_node("synth", wrap(lambda s: n_synthesize(s, cfg)))
    g.add_node("signals", wrap(lambda s: n_signals(s, cfg, embeddings)))
    g.add_node("llm_analysis", wrap(lambda s: n_llm_analysis(s, cfg, embeddings, llm_config)))
    g.add_node("adaptive_thresholds", wrap(lambda s: n_adaptive_thresholds(s, cfg, db)))
    g.add_node("evidence_gate", wrap(lambda s: n_evidence_gate(s, cfg)))
    g.add_node("dedup", wrap(lambda s: n_deduplicate(s, cfg, embeddings)))
    g.add_node("score", wrap(lambda s: n_score_and_decide(s, cfg)))
    g.add_node("promote", wrap(lambda s: n_promote_if_auto(s, db)))
    g.add_node("human_review", wrap(lambda s: s))  # Placeholder for human review

    g.set_entry_point("schema")
    g.add_edge("schema", "fetch")
    g.add_edge("fetch", "synth")
    g.add_edge("synth", "signals")
    g.add_edge("signals", "llm_analysis")
    g.add_edge("llm_analysis", "adaptive_thresholds")
    g.add_edge("adaptive_thresholds", "evidence_gate")
    
    # Conditional branching from evidence_gate
    g.add_conditional_edges(
        "evidence_gate",
        lambda state: state.get("evidence_gate", "continue_automated"),
        {
            "human_review": "human_review",
            "additional_verification": "dedup",
            "continue_automated": "dedup"
        }
    )
    
    g.add_edge("dedup", "score")
    g.add_edge("score", "promote")
    g.add_edge("promote", END)
    g.add_edge("human_review", END)

    app = g.compile()
    try:
        out_dict = app.invoke(to_dict(state))
        state = from_dict(out_dict)
    except Exception as e:
        print(f"Error during agent execution: {e}")
        # Set error state
        state.status = SubmissionStatus.REJECTED
        state.decision_type = None
        state.blockers.append("execution_error")
        state.reasons.append(f"Agent execution failed: {str(e)}")
        state.validity_score = 0.0

    # Persist submission decision and set review token if needed
    db.update_submission_decision(
        submission_id,
        status=(state.status or SubmissionStatus.REJECTED).value,  # default to rejected
        decision_type=(state.decision_type.value if state.decision_type else None),
        validity_score=state.validity_score,
        reasons=state.reasons,
        blockers=state.blockers,
        synthesized_content=state.synthesized_content,
        extracted_title=None,
        signals=state.signals,
        decided_at_iso=_now_iso(),
    )

    if state.status == SubmissionStatus.NEEDS_REVIEW:
        token = str(uuid.uuid4())
        db.set_submission_review_token(submission_id, token)

    submitted_title: str = (state.submission.submitted_title or "").strip()
    return ApprovalOutcome(
        status=(state.status or SubmissionStatus.REJECTED),  # default to rejected
        decision_type=(state.decision_type if state.decision_type else None),
        validity_score=state.validity_score,
        reasons=state.reasons,
        blockers=state.blockers,
        final_title=submitted_title or "",
        synthesized_content=state.synthesized_content,
        signals=state.signals,
    )


# Backward-compatible wrapper name
def evaluate_submission(db: ArticleDB, embeddings: EmbeddingSystem, submission_id: str, config: Optional[ApprovalConfig] = None) -> ApprovalOutcome:
    return run_approval_agent(db, embeddings, submission_id, config)


