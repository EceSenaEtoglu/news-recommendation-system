from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

from ..storage import ArticleDB
from ..data_models import Article, Source, SourceCategory, ContentType, SubmissionStatus, DecisionType, ArticleProvenance, SubmissionModel
from ..embeddings import EmbeddingSystem
from ..config import ApprovalConfig, JOURNALIST_DEFAULT_CREDIBILITY

from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor
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


# TODO? do a semantic check
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


def _compute_evidence_quality(evidence_fetch_outcomes: List[Dict]) -> float:
    if not evidence_fetch_outcomes:
        return 0.0
    usable = [e for e in evidence_fetch_outcomes if (e.get("extracted_length", 0) or 0) >= 300]
    if not usable:
        return 0.0
    score = 0.5 if len(usable) >= 1 else 0.0
    if len(usable) >= 2:
        score = 0.8
    if len(usable) >= 3:
        score = 1.0
    return score


def _compute_coherence_stub(evidence_fetch_outcomes: List[Dict]) -> float:
    usable = [e for e in evidence_fetch_outcomes if (e.get("extracted_length", 0) or 0) >= 300]
    if len(usable) >= 2:
        return 0.6
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


# ---------------- Agent Nodes ----------------
def n_schema_validate(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    sub = state.submission
    submitted_title = (sub.get("submitted_title") or "").strip()
    evidence_urls = []
    try:
        evidence_urls = json.loads(sub.get("evidence_urls") or "[]")
    except Exception:
        pass
    if not submitted_title:
        state.blockers.append("missing_title")
    if not evidence_urls and not (sub.get("content_raw") or "").strip():
        state.blockers.append("no_evidence_and_no_content")
    return state


def n_fetch_evidence(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    sub = state.submission
    evidence_urls: List[str] = sub.evidence_urls or []
    outcomes: List[Dict] = []
    if evidence_urls:
        with ThreadPoolExecutor(max_workers=min(4, len(evidence_urls))) as ex:
            futs = [ex.submit(fetch_and_extract, u, timeout=cfg.fetch_timeout_s, retries=cfg.fetch_retries, max_redirects=cfg.fetch_max_redirects) for u in evidence_urls]
            for f in futs:
                try:
                    outcomes.append(f.result())
                except Exception as e:
                    outcomes.append({"url": "", "status_code": None, "extracted_text": "", "error": str(e), "extracted_length": 0, "boilerplate_ratio": 1.0, "num_links": 0.0})
    state.signals["evidence_fetch"] = outcomes
    return state


def n_synthesize(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    sub = state.submission
    content_raw: Optional[str] = sub.get("content_raw")
    outcomes: List[Dict] = state.signals.get("evidence_fetch", [])
    synthesized, _ = _synthesize_content_stub(content_raw, outcomes)
    if not synthesized:
        synthesized = (sub.get("description") or "").strip()
    state.synthesized_content = synthesized
    return state


def n_signals(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    sub = state.submission
    submitted_title: str = (sub.submitted_title or "").strip()
    outcomes: List[Dict] = state.signals.get("evidence_fetch", [])
    state.signals["evidence_quality"] = _compute_evidence_quality(outcomes)
    state.signals["cross_evidence_coherence"] = _compute_coherence_stub(outcomes)
    state.signals["title_consistency"] = _basic_title_consistency(submitted_title, state.synthesized_content)
    return state


def n_deduplicate(state: AgentState, cfg: ApprovalConfig, embeddings: EmbeddingSystem) -> AgentState:
    dup_cos = _dup_max_cosine(embeddings, state.synthesized_content)
    state.signals["dup_max_cosine_vs_corpus"] = dup_cos
    if dup_cos >= cfg.tau_dup:
        state.blockers.append("exact_duplicate")
    return state


def n_score_and_decide(state: AgentState, cfg: ApprovalConfig) -> AgentState:
    positives = (
        cfg.w_evidence_quality * state.signals.get("evidence_quality", 0.0) +
        cfg.w_coherence * state.signals.get("cross_evidence_coherence", 0.0) +
        cfg.w_title_consistency * state.signals.get("title_consistency", 0.0) +
        cfg.w_content_quality * (1.0 if len(state.synthesized_content) >= 800 else 0.5 if len(state.synthesized_content) >= 300 else 0.0)
    )
    dup_cos = state.signals.get("dup_max_cosine_vs_corpus", 0.0)
    dup_pen = min(1.0, max(0.0, (dup_cos - 0.80) / 0.15)) if dup_cos > 0.80 else 0.0
    penalty = cfg.w_penalties_dup_safety * max(dup_pen, 0.0)
    state.validity_score = max(0.0, min(1.0, positives + penalty))

    if state.signals.get("evidence_quality", 0.0) >= 0.8:
        state.reasons.append("+evidence_quality")
    if state.signals.get("cross_evidence_coherence", 0.0) >= 0.5:
        state.reasons.append("+coherence")
    if state.signals.get("title_consistency", 0.0) >= cfg.tau_title_align:
        state.reasons.append("+title_consistency")
    if dup_pen > 0.0:
        state.reasons.append("-near_duplicate")

    if state.blockers:
        state.status = SubmissionStatus.REJECTED
        state.decision_type = None
    elif state.validity_score >= cfg.tau_auto:
        state.status = SubmissionStatus.AUTO_APPROVED
        state.decision_type = DecisionType.AUTO_APPROVED
    elif state.validity_score >= cfg.tau_review:
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
    submitted_title: str = (sub.get("submitted_title") or "").strip()
    evidence_urls: List[str] = []
    try:
        evidence_urls = json.loads(sub.get("evidence_urls") or "[]")
    except Exception:
        pass
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
        title=submitted_title or ("Journalist Report " + article_id[:8]),
        content=state.synthesized_content or (submitted_title or ""),
        description=sub.get("description"),
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


def run_approval_agent(db: ArticleDB, embeddings: EmbeddingSystem, submission_id: str, config: Optional[ApprovalConfig] = None) -> ApprovalOutcome:
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
        }

    def from_dict(d: dict) -> AgentState:
        return AgentState(
            submission=d.get("submission", {}),
            signals=d.get("signals", {}),
            blockers=d.get("blockers", []),
            reasons=d.get("reasons", []),
            synthesized_content=d.get("synthesized_content", ""),
            validity_score=d.get("validity_score", 0.0),
            status=d.get("status"),
            decision_type=d.get("decision_type"),
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
    g.add_node("signals", wrap(lambda s: n_signals(s, cfg)))
    g.add_node("dedup", wrap(lambda s: n_deduplicate(s, cfg, embeddings)))
    g.add_node("score", wrap(lambda s: n_score_and_decide(s, cfg)))
    g.add_node("promote", wrap(lambda s: n_promote_if_auto(s, db)))

    g.set_entry_point("schema")
    g.add_edge("schema", "fetch")
    g.add_edge("fetch", "synth")
    g.add_edge("synth", "signals")
    g.add_edge("signals", "dedup")
    g.add_edge("dedup", "score")
    g.add_edge("score", "promote")
    g.add_edge("promote", END)

    app = g.compile()
    out_dict = app.invoke(to_dict(state))
    state = from_dict(out_dict)

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


