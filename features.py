"""
features.py — Step 1: turn one messy candidate JSON record into clean,
structured features we can score on.

WHY THIS FILE EXISTS:
The JD warns that the "wrong" answer is to just count AI-sounding keywords
in a candidate's skills list. The "right" answer is to look at what they
actually DID (career_history descriptions) and weigh that more heavily than
what they SAY they know (skills list).

So this file builds two separate signals and keeps them separate:
  1. skills_signal   -> how many JD-relevant skills they list, and how strong
  2. experience_signal -> how much their actual job history shows JD-relevant work

We deliberately do NOT blend these into one number yet. We score them
separately so a "skills_signal high, experience_signal low" candidate
(keyword-stuffer) can be told apart from a "skills_signal low,
experience_signal high" candidate (the plain-language Tier-5 the JD
explicitly wants us to catch).
"""

import json
import re
from datetime import date
from template_relevance import compute_template_score

# ----------------------------------------------------------------------------
# 1. JD VOCABULARY — the words/phrases that signal real relevance.
#    Grouped by theme so we can tell *which kind* of relevance a candidate has.
#    This list comes directly from reading job_description.md closely.
# ----------------------------------------------------------------------------

RETRIEVAL_TERMS = [
    "embedding", "embeddings", "sentence-transformers", "sentence transformers",
    "bge", "e5 embedding", "vector database", "vector db", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "faiss", "hybrid search",
    "hybrid retrieval", "dense retrieval", "semantic search", "vector search",
    "retrieval", "indexing", "ann search", "approximate nearest neighbor",
]

RANKING_EVAL_TERMS = [
    "ndcg", "mrr", "map", "mean average precision", "learning to rank",
    "learning-to-rank", "ltr", "xgboost", "lightgbm", "ranking model",
    "ranking system", "a/b test", "ab test", "offline evaluation",
    "offline-online correlation", "click-through", "ctr", "recommendation system",
    "recommender system", "recsys", "search relevance", "relevance labeling",
]

LLM_TERMS = [
    "llm", "large language model", "fine-tun", "lora", "qlora", "peft",
    "prompt engineering", "rag", "retrieval-augmented", "re-ranking", "reranking",
    "transformer", "bert", "gpt", "hugging face", "huggingface",
]

PRE_LLM_ML_PRODUCTION_TERMS = [
    "production ml", "deployed model", "shipped a model", "model in production",
    "feature pipeline", "feature engineering", "model serving", "mlops",
    "model monitoring", "shipped to real users", "shipped to users",
    "deployed to real users", "scale", "real users",
]

# Disqualifier-flavoured vocabulary (used for red flags, not scoring up)
RESEARCH_ONLY_TERMS = ["research lab", "academic lab", "phd thesis", "published paper",
                         "research scientist", "research-only"]
CONSULTING_FIRMS = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
                     "tech mahindra", "mindtree"]
CV_SPEECH_ROBOTICS_TERMS = ["computer vision", "image classification", "object detection",
                             "speech recognition", "robotics", "autonomous vehicle"]
FRAMEWORK_ENTHUSIAST_TERMS = ["langchain tutorial", "built a demo", "weekend project",
                               "side project fine-tuning", "kaggle competition"]


def _text_blob(candidate: dict) -> str:
    """Concatenate every free-text field we want to keyword-search, lowercased."""
    parts = [
        candidate["profile"].get("headline", ""),
        candidate["profile"].get("summary", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(parts).lower()


def _count_hits(text: str, terms: list[str]) -> int:
    """Count how many distinct terms from `terms` appear in `text` (substring match)."""
    return sum(1 for t in terms if t in text)


def _career_history_blob(candidate: dict) -> str:
    """Same as _text_blob but ONLY career_history (no headline/summary).
    Used for the experience_signal so skills/summary fluff can't inflate it."""
    parts = []
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    return " ".join(parts).lower()


def compute_skills_signal(candidate: dict) -> dict:
    """How strong is this candidate's SKILLS LIST (not career history) on JD-relevant tech?
    This is intentionally the 'weak' signal the JD warns us not to over-trust."""
    skills = candidate.get("skills", [])
    relevant_vocab = set(RETRIEVAL_TERMS + RANKING_EVAL_TERMS + LLM_TERMS)

    matched = []
    for s in skills:
        name = s.get("name", "").lower()
        if any(term in name or name in term for term in relevant_vocab):
            matched.append(s)

    # Weight by proficiency so "expert" counts more than "beginner"
    prof_weight = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    weighted_score = sum(prof_weight.get(s.get("proficiency", "beginner"), 1) for s in matched)

    return {
        "skills_signal_matched_count": len(matched),
        "skills_signal_weighted_score": weighted_score,
        "skills_signal_matched_names": [s["name"] for s in matched],
    }


def compute_experience_signal(candidate: dict) -> dict:
    """How strong is this candidate's ACTUAL CAREER HISTORY on JD-relevant work?
    This is the signal the JD says should dominate."""
    blob = _career_history_blob(candidate)

    retrieval_hits = _count_hits(blob, RETRIEVAL_TERMS)
    ranking_hits = _count_hits(blob, RANKING_EVAL_TERMS)
    llm_hits = _count_hits(blob, LLM_TERMS)
    prod_ml_hits = _count_hits(blob, PRE_LLM_ML_PRODUCTION_TERMS)

    return {
        "exp_retrieval_hits": retrieval_hits,
        "exp_ranking_eval_hits": ranking_hits,
        "exp_llm_hits": llm_hits,
        "exp_prod_ml_hits": prod_ml_hits,
        "exp_total_relevance_hits": retrieval_hits + ranking_hits + llm_hits + prod_ml_hits,
    }


SENIORITY_LADDER = ["junior", "associate", "senior", "staff", "principal",
                     "lead", "director", "vp", "head"]


def _seniority_rank(title: str) -> int:
    """Return the highest seniority-ladder index found in a title string, or -1
    if the title has no seniority modifier at all (e.g. plain 'Engineer',
    'Analyst', 'Manager' with no Junior/Senior/Staff/etc. prefix)."""
    title = title.lower()
    best = -1
    for i, word in enumerate(SENIORITY_LADDER):
        if word in title:
            best = max(best, i)
    return best


def compute_red_flags(candidate: dict) -> dict:
    """Explicit disqualifier checks straight from the JD's 'do NOT want' section."""
    blob = _text_blob(candidate)
    career = candidate.get("career_history", [])
    companies = [job.get("company", "").lower() for job in career]

    # Title-chasing (per JD): jumping SENIORITY LABELS upward (Senior -> Staff -> Principal)
    # company to company every ~1.5 yrs, WITHOUT changing functional role.
    # This is different from someone moving laterally into a more specialized role
    # (e.g. NLP Engineer -> Search Engineer -> Recommendation Systems Engineer), which
    # is a sign of growth, not title-chasing, even if tenures are short.
    short_stints = sum(1 for job in career if job.get("duration_months", 999) < 20)
    titles = [job.get("title", "") for job in career]
    seniority_ranks = [_seniority_rank(t) for t in titles]
    # Only consider it escalation if EVERY job actually carries an explicit seniority
    # word (no -1s mixed in — a missing label isn't evidence of low seniority) and
    # the sequence is strictly increasing (true climbing, not just present-and-flat).
    has_explicit_seniority_throughout = len(seniority_ranks) >= 3 and all(r >= 0 for r in seniority_ranks)
    strictly_escalating = has_explicit_seniority_throughout and seniority_ranks == sorted(set(seniority_ranks)) \
        and len(set(seniority_ranks)) == len(seniority_ranks)
    title_chasing_flag = short_stints >= 3 and strictly_escalating

    # 100% consulting career: every employer is a known consulting firm
    consulting_only = bool(companies) and all(
        any(firm in c for firm in CONSULTING_FIRMS) for c in companies
    )

    research_only_flag = _count_hits(blob, RESEARCH_ONLY_TERMS) > 0

    # JD's actual wording: "primary expertise is CV/speech/robotics WITHOUT significant
    # NLP/IR exposure." So this should only fire if CV/speech/robotics terms appear
    # AND retrieval/ranking/NLP terms are essentially absent from career history.
    cv_speech_robotics_hits = _count_hits(blob, CV_SPEECH_ROBOTICS_TERMS)
    nlp_ir_hits = _count_hits(_career_history_blob(candidate), RETRIEVAL_TERMS + RANKING_EVAL_TERMS)
    cv_speech_robotics_flag = cv_speech_robotics_hits > 0 and nlp_ir_hits == 0

    framework_enthusiast_flag = _count_hits(blob, FRAMEWORK_ENTHUSIAST_TERMS) > 0

    return {
        "flag_title_chasing": title_chasing_flag,
        "flag_consulting_only_career": consulting_only,
        "flag_research_only": research_only_flag,
        "flag_cv_speech_robotics_background": cv_speech_robotics_flag,
        "flag_framework_enthusiast_language": framework_enthusiast_flag,
    }


def compute_honeypot_flags(candidate: dict) -> dict:
    """Internal-consistency checks for 'subtly impossible' profiles.
    Confirmed pattern from direct dataset inspection: expert proficiency
    with 0 months duration. We also check a couple of other plausible
    impossibility patterns described in the README."""
    skills = candidate.get("skills", [])
    expert_zero_duration = [
        s["name"] for s in skills
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0
    ]

    # Years of experience should roughly reconcile with sum of career_history durations
    yoe = candidate["profile"].get("years_of_experience", 0)
    total_months_in_history = sum(j.get("duration_months", 0) for j in candidate.get("career_history", []))
    # convert candidate's stated YOE to months for comparison
    yoe_months = yoe * 12
    # Flag a large mismatch (more than 3 years / 36 months difference either way)
    yoe_mismatch = abs(yoe_months - total_months_in_history) > 36

    # Many "expert" skills relative to total relevant experience hits — classic keyword-stuffer
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")

    return {
        "honeypot_expert_zero_duration_skills": expert_zero_duration,
        "honeypot_expert_zero_duration_count": len(expert_zero_duration),
        "honeypot_yoe_history_mismatch": yoe_mismatch,
        "honeypot_yoe_months_stated": round(yoe_months),
        "honeypot_months_in_history": total_months_in_history,
        "expert_skill_count": expert_count,
    }


def compute_behavioral_signal(candidate: dict) -> dict:
    """Distill the 23 redrob_signals fields into a few summary scores."""
    sig = candidate.get("redrob_signals", {})

    last_active = sig.get("last_active_date")
    days_inactive = None
    if last_active:
        try:
            y, m, d = map(int, last_active.split("-"))
            days_inactive = (date(2026, 6, 21) - date(y, m, d)).days
        except Exception:
            pass

    is_reachable = bool(
        sig.get("open_to_work_flag", False)
        and (days_inactive is not None and days_inactive <= 60)
        and sig.get("recruiter_response_rate", 0) >= 0.3
    )

    return {
        "days_since_active": days_inactive,
        "recruiter_response_rate": sig.get("recruiter_response_rate"),
        "interview_completion_rate": sig.get("interview_completion_rate"),
        "notice_period_days": sig.get("notice_period_days"),
        "willing_to_relocate": sig.get("willing_to_relocate"),
        "open_to_work_flag": sig.get("open_to_work_flag"),
        "is_reachable_estimate": is_reachable,
    }


def extract_features(candidate: dict) -> dict:
    """Top-level entry point: run all feature extractors on one candidate record."""
    out = {
        "candidate_id": candidate["candidate_id"],
        "current_title": candidate["profile"].get("current_title"),
        "current_company": candidate["profile"].get("current_company"),
        "years_of_experience": candidate["profile"].get("years_of_experience"),
        "location": candidate["profile"].get("location"),
        "country": candidate["profile"].get("country"),
    }
    out.update(compute_skills_signal(candidate))
    out.update(compute_experience_signal(candidate))
    out.update(compute_red_flags(candidate))
    out.update(compute_honeypot_flags(candidate))
    out.update(compute_behavioral_signal(candidate))
    template_result = compute_template_score(candidate.get("career_history", []))
    out["template_max_score"] = template_result["template_max_score"]
    out["template_avg_score"] = template_result["template_avg_score"]
    out["template_current_role_score"] = template_result["template_current_role_score"]
    return out


if __name__ == "__main__":
    # Quick smoke test against the 50 sample candidates
    with open("/mnt/user-data/uploads/sample_candidates.json") as f:
        samples = json.load(f)

    rows = [extract_features(c) for c in samples]

    # Sort by total relevance hits (experience signal) descending, to eyeball the top
    rows_sorted = sorted(rows, key=lambda r: -r["exp_total_relevance_hits"])
    print(f"{'cand_id':<14}{'title':<32}{'exp_hits':<9}{'skills_w':<9}{'flags'}")
    for r in rows_sorted[:15]:
        flags = [k.replace("flag_", "") for k, v in r.items() if k.startswith("flag_") and v]
        print(f"{r['candidate_id']:<14}{r['current_title'][:30]:<32}"
              f"{r['exp_total_relevance_hits']:<9}{r['skills_signal_weighted_score']:<9}{flags}")
