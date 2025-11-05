from __future__ import annotations

import uuid
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict, Any
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END

from ..storage import ArticleDB
from ..data_models import Article, Source, SourceCategory, ContentType, SubmissionStatus, DecisionType, ArticleProvenance, SubmissionModel
from ..embeddings import EmbeddingSystem
from ..config import ApprovalConfig, JOURNALIST_DEFAULT_CREDIBILITY, LLMConfig
from ..utils.content_extract import fetch_and_extract
from ..utils.helpers import (
    now_iso,
    basic_title_consistency,
    compute_evidence_quality,
    compute_coherence,
    synthesize_content,
    dup_max_cosine,
    get_duplicate_preview,
    extract_title_from_evidence,
    compute_title_alignment,
    llm_content_analysis,
    calculate_synthesis_quality,
    adaptive_threshold_calculation,
    calculate_confidence_level,
)

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


class AgentState(TypedDict):
    """State passed through the approval agent workflow.
    """
    submission: SubmissionModel
    signals: Dict[str, Any]
    blockers: List[str]
    reasons: List[str]
    synthesized_content: str
    validity_score: float
    status: Optional[SubmissionStatus]
    decision_type: Optional[DecisionType]
    # Enhanced agentic behavior fields
    llm_analysis: Optional[Dict[str, Any]]
    adaptive_thresholds: Optional[Dict[str, Any]]
    confidence_level: float


def apply_reviewer_decision(db: ArticleDB,
                            submission_id: str,
                            decision: str,
                            *,
                            final_title: Optional[str] = None,
                            notes: Optional[str] = None) -> bool:
    """Apply a human reviewer decision for a submission in NEEDS_REVIEW"""
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


# ================ Agent Workflow Nodes ================
# Each node processes and modifies the state, passing it to the next node

def n_schema_validate(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    """Validate submission schema (title, URLs, content)."""
    sub = state["submission"]
    submitted_title = (sub.submitted_title or "").strip()
    evidence_urls = sub.evidence_urls or []
    content_raw = sub.content_raw or ""
    
    blockers = state.setdefault("blockers", [])
    reasons = state.setdefault("reasons", [])
    
    # Validate title
    if not submitted_title:
        blockers.append("missing_title")
    elif len(submitted_title) < 10:
        blockers.append("title_too_short")
    elif len(submitted_title) > 200:
        blockers.append("title_too_long")
    
    # Validate evidence URLs
    if evidence_urls:
        url_pattern = re.compile(
            r'^https?://(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
            r'localhost|\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::\d+)?(?:/?|[/?]\S+)$',
            re.IGNORECASE)
        
        invalid_urls = [url for url in evidence_urls if not url_pattern.match(url)]
        if invalid_urls:
            blockers.append("invalid_urls")
            reasons.append(f"Invalid URLs: {invalid_urls}")
    
    # Validate content
    if not evidence_urls and not content_raw.strip():
        blockers.append("no_evidence_and_no_content")
    elif content_raw and len(content_raw.strip()) < 50:
        blockers.append("content_too_short")
    
    return state

def n_fetch_evidence(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    """Fetch and extract content from evidence URLs in parallel."""
    sub = state["submission"]
    evidence_urls: List[str] = sub.evidence_urls or []
    outcomes: List[Dict] = []
    
    if evidence_urls:
        try:
            with ThreadPoolExecutor(max_workers=min(4, len(evidence_urls))) as ex:
                futs = [ex.submit(fetch_and_extract, u, timeout=cfg.fetch_timeout_s, 
                                 retries=cfg.fetch_retries, max_redirects=cfg.fetch_max_redirects) 
                        for u in evidence_urls]
                for i, f in enumerate(futs):
                    try:
                        outcomes.append(f.result(timeout=cfg.fetch_timeout_s + 5))
                    except Exception as e:
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
        except Exception as e:
            state.setdefault("blockers", []).append("evidence_fetch_error")
            state.setdefault("reasons", []).append(f"Evidence fetching failed: {str(e)}")
    
    state.setdefault("signals", {})["evidence_fetch"] = outcomes
    return state


def n_synthesize(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    """Synthesize content from raw content or fetched evidence."""
    sub = state["submission"]
    content_raw: Optional[str] = sub.content_raw
    outcomes: List[Dict] = state.get("signals", {}).get("evidence_fetch", [])
    synthesized = synthesize_content(content_raw, outcomes)
    if not synthesized:
        synthesized = (sub.description or "").strip()
    state["synthesized_content"] = synthesized
    return state


def n_signals(state: AgentState, cfg: ApprovalConfig, embeddings: EmbeddingSystem) -> AgentState:
    """Compute quality signals from evidence and content."""
    sub = state["submission"]
    submitted_title: str = (sub.submitted_title or "").strip()
    signals = state.setdefault("signals", {})
    outcomes: List[Dict] = signals.get("evidence_fetch", [])
    
    signals["evidence_quality"] = compute_evidence_quality(outcomes, cfg)
    signals["cross_evidence_coherence"] = compute_coherence(embeddings, outcomes, cfg)
    
    extracted_title = extract_title_from_evidence(outcomes)
    signals["extracted_title"] = extracted_title
    signals["title_alignment"] = compute_title_alignment(submitted_title, extracted_title)
    signals["title_consistency"] = basic_title_consistency(submitted_title, state.get("synthesized_content", ""))
    
    return state


def n_llm_analysis(state: AgentState, cfg: ApprovalConfig, embeddings: EmbeddingSystem, llm_config: Optional[LLMConfig] = None) -> AgentState:
    """Perform LLM-based content analysis and fact-checking."""
    signals = state.setdefault("signals", {})
    outcomes: List[Dict] = signals.get("evidence_fetch", [])
    evidence_texts = [e.get("extracted_text", "") for e in outcomes if e.get("extracted_text")]
    
    llm_analysis = llm_content_analysis(state.get("synthesized_content", ""), evidence_texts, llm_config)
    state["llm_analysis"] = llm_analysis
    
    signals["fact_check_score"] = llm_analysis["fact_check_score"]
    signals["synthesis_quality"] = llm_analysis["synthesis_quality"]
    signals["llm_confidence"] = llm_analysis["confidence"]
    
    reasons = state.setdefault("reasons", [])
    if llm_analysis["fact_check_score"] >= 0.8:
        reasons.append("+llm_fact_check")
    if llm_analysis["synthesis_quality"] >= 0.7:
        reasons.append("+llm_synthesis")
    
    return state


def n_adaptive_thresholds(state: AgentState, cfg: ApprovalConfig, db: ArticleDB) -> AgentState:
    """Calculate adaptive thresholds and confidence level."""
    state["adaptive_thresholds"] = adaptive_threshold_calculation(db)
    state["confidence_level"] = calculate_confidence_level(state.get("signals", {}), state.get("synthesized_content", ""))
    state.setdefault("signals", {})["confidence_level"] = state["confidence_level"]
    return state


def n_evidence_gate(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    """Determine workflow path based on evidence quality and confidence."""
    signals = state.setdefault("signals", {})
    evidence_quality = signals.get("evidence_quality", 0.0)
    confidence_level = state.get("confidence_level", 0.0)
    
    if evidence_quality < 0.3 or confidence_level < 0.4:
        signals["evidence_gate"] = "human_review"
    elif evidence_quality < 0.6 or confidence_level < 0.7:
        signals["evidence_gate"] = "additional_verification"
    else:
        signals["evidence_gate"] = "continue_automated"
    
    return state


def n_deduplicate(state: AgentState, cfg: ApprovalConfig, embeddings: EmbeddingSystem) -> AgentState:
    """Check for duplicate content using semantic similarity."""
    signals = state.setdefault("signals", {})
    content = state.get("synthesized_content", "")
    
    dup_cos = dup_max_cosine(embeddings, content)
    signals["dup_max_cosine_vs_corpus"] = dup_cos
    signals["duplicate_preview"] = get_duplicate_preview(embeddings, content, k=3)
    
    if dup_cos >= cfg.tau_dup:
        state.setdefault("blockers", []).append("exact_duplicate")
    return state


def n_score_and_decide(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    """Calculate validity score and make approval decision."""
    signals = state.setdefault("signals", {})
    reasons = state.setdefault("reasons", [])
    blockers = state.get("blockers", [])
    
    # Calculate score components
    title_score = signals.get("title_alignment", signals.get("title_consistency", 0.0))
    content_length = len(state.get("synthesized_content", ""))
    content_quality = 1.0 if content_length >= 800 else (0.5 if content_length >= 300 else 0.0)
    
    positives = (
        cfg.w_evidence_quality * signals.get("evidence_quality", 0.0) +
        cfg.w_coherence * signals.get("cross_evidence_coherence", 0.0) +
        cfg.w_title_consistency * title_score +
        cfg.w_content_quality * content_quality +
        0.1 * signals.get("fact_check_score", 0.0) +
        0.1 * signals.get("synthesis_quality", 0.0)
    )
    
    # Apply duplicate penalty
    dup_cos = signals.get("dup_max_cosine_vs_corpus", 0.0)
    dup_pen = min(1.0, max(0.0, (dup_cos - 0.80) / 0.15)) if dup_cos > 0.80 else 0.0
    penalty = cfg.w_penalties_dup_safety * dup_pen
    confidence_adjustment = state.get("confidence_level", 0.0) * 0.1
    
    state["validity_score"] = max(0.0, min(1.0, positives - penalty + confidence_adjustment))
    
    # Add reasons
    if signals.get("evidence_quality", 0.0) >= 0.8:
        reasons.append("+evidence_quality")
    if signals.get("cross_evidence_coherence", 0.0) >= 0.5:
        reasons.append("+coherence")
    if title_score >= cfg.tau_title_align:
        reasons.append("+title_alignment")
    if signals.get("fact_check_score", 0.0) >= 0.8:
        reasons.append("+llm_fact_check")
    if signals.get("synthesis_quality", 0.0) >= 0.7:
        reasons.append("+llm_synthesis")
    if state.get("confidence_level", 0.0) >= 0.8:
        reasons.append("+high_confidence")
    if dup_pen > 0.0:
        reasons.append("-near_duplicate")
    
    # Set suggested title
    if signals.get("extracted_title"):
        signals["suggested_final_title"] = signals["extracted_title"]
    
    # Make decision using adaptive thresholds
    thresholds = state.get("adaptive_thresholds") or {}
    tau_auto = thresholds.get("tau_auto", cfg.tau_auto)
    tau_review = thresholds.get("tau_review", cfg.tau_review)

    if blockers:
        state["status"] = SubmissionStatus.REJECTED
        state["decision_type"] = None
    elif state["validity_score"] >= tau_auto:
        state["status"] = SubmissionStatus.AUTO_APPROVED
        state["decision_type"] = DecisionType.AUTO_APPROVED
    elif state["validity_score"] >= tau_review:
        state["status"] = SubmissionStatus.NEEDS_REVIEW
        state["decision_type"] = None
    else:
        state["status"] = SubmissionStatus.REJECTED
        state["decision_type"] = None
    
    return state


def n_promote_if_auto(state: AgentState, db: ArticleDB) -> AgentState:
    """Promote auto-approved submissions to articles."""
    if state.get("status") != SubmissionStatus.AUTO_APPROVED:
        return state
    
    sub = state["submission"]
    submitted_title: str = (sub.submitted_title or "").strip()
    signals = state.get("signals", {})
    final_title = signals.get("suggested_final_title") or submitted_title or ("Journalist Report " + str(uuid.uuid4())[:8])
    
    article = Article(
        id=str(uuid.uuid4()),
        title=final_title,
        content=state.get("synthesized_content", "") or submitted_title,
        description=sub.description,
        url="#",
        source=Source(
            id="journalist",
            name="Journalist Report",
            url="",
            category=SourceCategory.GENERAL,
            credibility_score=JOURNALIST_DEFAULT_CREDIBILITY,
        ),
        published_at=datetime.now(timezone.utc),
        content_type=ContentType.FACTUAL,
        provenance_source=ArticleProvenance.JOURNALIST_REPORT,
        decision_type=DecisionType.AUTO_APPROVED,
        evidence_urls=sub.evidence_urls or [],
    )
    db.save_article(article)
    return state


def run_approval_agent(db: ArticleDB, embeddings: EmbeddingSystem, submission_id: str, 
                      config: Optional[ApprovalConfig] = None, llm_config: Optional[LLMConfig] = None) -> ApprovalOutcome:
    """
    Main approval agent workflow using LangGraph.
    
    Workflow:
    1. Schema validation -> 2. Fetch evidence -> 3. Synthesize content
    4. Compute signals -> 5. LLM analysis -> 6. Adaptive thresholds
    7. Evidence gate (conditional) -> 8. Deduplicate -> 9. Score & decide
    10. Promote (if auto-approved) -> END
    """
    cfg = config or ApprovalConfig()
    sub = db.get_submission(submission_id)
    if not sub:
        raise ValueError(f"Submission not found: {submission_id}")

    # Initialize state (TypedDict allows direct dict usage)
    state: AgentState = {
        "submission": sub,
        "signals": {},
        "blockers": [],
        "reasons": [],
        "synthesized_content": "",
        "validity_score": 0.0,
        "status": None,
        "decision_type": None,
    }

    # Build workflow graph
    g = StateGraph(AgentState)
    
    # Add nodes with dependencies bound via closures
    g.add_node("schema", lambda s: n_schema_validate(s, cfg))
    g.add_node("fetch", lambda s: n_fetch_evidence(s, cfg))
    g.add_node("synth", lambda s: n_synthesize(s, cfg))
    g.add_node("signals", lambda s: n_signals(s, cfg, embeddings))
    g.add_node("llm_analysis", lambda s: n_llm_analysis(s, cfg, embeddings, llm_config))
    g.add_node("adaptive_thresholds", lambda s: n_adaptive_thresholds(s, cfg, db))
    g.add_node("evidence_gate", lambda s: n_evidence_gate(s, cfg))
    g.add_node("dedup", lambda s: n_deduplicate(s, cfg, embeddings))
    g.add_node("score", lambda s: n_score_and_decide(s, cfg))
    g.add_node("promote", lambda s: n_promote_if_auto(s, db))
    g.add_node("human_review", lambda s: s)  # Placeholder

    # Define workflow edges
    g.set_entry_point("schema")
    g.add_edge("schema", "fetch")
    g.add_edge("fetch", "synth")
    g.add_edge("synth", "signals")
    g.add_edge("signals", "llm_analysis")
    g.add_edge("llm_analysis", "adaptive_thresholds")
    g.add_edge("adaptive_thresholds", "evidence_gate")
    
    # Conditional branching based on evidence quality
    g.add_conditional_edges(
        "evidence_gate",
        lambda s: s.get("signals", {}).get("evidence_gate", "continue_automated"),
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

    # Execute workflow
    app = g.compile()
    try:
        state = app.invoke(state)
    except Exception as e:
        print(f"Error during agent execution: {e}")
        state["status"] = SubmissionStatus.REJECTED
        state["decision_type"] = None
        state.setdefault("blockers", []).append("execution_error")
        state.setdefault("reasons", []).append(f"Agent execution failed: {str(e)}")
        state["validity_score"] = 0.0

    # Persist decision
    db.update_submission_decision(
        submission_id,
        status=(state.get("status") or SubmissionStatus.REJECTED).value,
        decision_type=(state.get("decision_type").value if state.get("decision_type") else None),
        validity_score=state.get("validity_score", 0.0),
        reasons=state.get("reasons", []),
        blockers=state.get("blockers", []),
        synthesized_content=state.get("synthesized_content", ""),
        extracted_title=None,
        signals=state.get("signals", {}),
        decided_at_iso=now_iso(),
    )

    if state.get("status") == SubmissionStatus.NEEDS_REVIEW:
        db.set_submission_review_token(submission_id, str(uuid.uuid4()))

    submitted_title: str = (state["submission"].submitted_title or "").strip()
    return ApprovalOutcome(
        status=(state.get("status") or SubmissionStatus.REJECTED),
        decision_type=state.get("decision_type"),
        validity_score=state.get("validity_score", 0.0),
        reasons=state.get("reasons", []),
        blockers=state.get("blockers", []),
        final_title=submitted_title or "",
        synthesized_content=state.get("synthesized_content", ""),
        signals=state.get("signals", {}),
    )


# Backward-compatible wrapper name
def evaluate_submission(db: ArticleDB, embeddings: EmbeddingSystem, submission_id: str, config: Optional[ApprovalConfig] = None) -> ApprovalOutcome:
    return run_approval_agent(db, embeddings, submission_id, config)


