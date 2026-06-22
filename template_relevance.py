"""
template_relevance.py — Precision upgrade over keyword counting.

KEY DISCOVERY: this dataset's 300,171 career_history descriptions are not
unique free text -- they're drawn from a fixed pool of only 44 templates,
reused across candidates (confirmed by direct frequency count: the most
common template appears 25,515 times). This means we can classify each of
the 44 templates EXACTLY ONCE, by reading it carefully, and then score any
candidate by exact lookup -- far more precise than keyword/substring
matching, which can be fooled by partial matches or miss real signal
phrased differently than our keyword list expected.

Each template below was read in full and scored 0-10 on JD relevance,
with a short note on WHY, based on direct comparison to job_description.md's
must-haves, nice-to-haves, and explicit disqualifiers.

Scores reflect a deliberate tiering that matches the actual count
distribution we observed:
  0   = completely unrelated function (sales, support, marketing, accounting,
        HR, ops, design, mechanical -- templates 1-9)
  1   = tech-adjacent but non-ML (DevOps, mobile, frontend, generic backend,
        QA -- templates 10-15)
  2   = data engineering / analytics, explicitly NOT ML modeling
        (templates 16-21)
  3-4 = genuine ML work but with an EXPLICIT JD disqualifier stated in the
        template's own text (CV-only, forecasting-only, "not the model
        itself", sentiment/classification without retrieval -- templates
        22-27). These are the trap zone: real ML keywords appear, but the
        template itself discloses exactly the limitation the JD warns
        against.
  5   = ranking/recsys work, present but framed as secondary/easy
        (template 28)
  6-7 = solid, clearly relevant production ranking/recsys/ML-ops work
        (templates 29, 32, 33)
  7-8 = semantic search / embeddings work, core to the JD's must-haves
        (templates 30, 38, 39)
  8-9 = RAG / LLM fine-tuning / hybrid retrieval with explicit evaluation
        framework design -- exactly the JD's ideal-candidate sketch
        (templates 31, 34, 35, 36, 37)
  9-10 = the clearest, most senior matches: explicit NDCG/MRR-style eval
        framework ownership combined with retrieval+ranking ownership
        end-to-end (templates 40-44, which use deliberately generic
        language -- "systems that understand what users are looking for" --
        that a naive keyword matcher would likely UNDER-score, which is
        probably intentional: these read like the "plain-language Tier 5"
        candidates the JD explicitly tells us not to miss)
"""

# Maps the first ~60 chars of a description (a stable enough prefix given
# templates are reused verbatim) to (score_0_to_10, short_reason).
# Built from manually reading all 44 templates in all_44_templates.txt.

TEMPLATE_SCORES = {
    "Enterprise sales of cloud software solutions into": (0, "sales, unrelated"),
    "Customer support team lead at a SaaS product": (0, "support/people management, unrelated"),
    "Marketing leadership role at a B2B SaaS company": (0, "marketing, unrelated"),
    "Business analyst at a consulting firm, working pri": (0, "consulting/BA, self-discloses limited AI depth"),
    "Brand design and creative direction at a consumer-": (0, "design, unrelated"),
    "Mechanical engineering design role at a hardware-p": (0, "mechanical engineering, unrelated"),
    "Senior accounting role at a mid-sized company": (0, "accounting, unrelated"),
    "Content writing and SEO strategy for a tech-focuse": (0, "content/SEO, unrelated despite AI topic coverage"),
    "Operations management role at a logistics company.": (0, "ops management, unrelated"),
    "Cloud infrastructure and DevOps work at an enterpr": (1, "DevOps/infra, no ML"),
    "Android mobile development using Java and (more re": (1, "mobile dev, no ML"),
    "Frontend engineering at a media company. React, Ty": (1, "frontend, no ML"),
    "Java backend development at a large enterprise": (1, "generic backend, no ML"),
    "Full-stack web application development at a SaaS c": (1, "full-stack, no ML"),
    "Test automation and QA engineering for a fintech p": (1, "QA, no ML"),
    "Designed and maintained the analytical data wareho": (2, "data warehousing/BI, not ML modeling"),
    "Built and maintained data pipelines on Apache Airf": (2, "data engineering, ML-adjacent only"),
    "Backend + data hybrid role at a growth-stage start": (2, "data infra, explicitly minor ML"),
    "Implemented streaming data pipelines on Kafka and ": (2, "data engineering, adjacent ML exposure only"),
    "Mixed data science and analytics-engineering role ": (2, "mostly analytics infra, light ML"),
    "Backend development with Python (FastAPI), Postgre": (2, "backend eng integrating model API, not modeling"),
    "Contributed to ML feature engineering and model de": (3, "JD disqualifier: 'production-side engineer, not modeling'"),
    "Built recommendation-style features at a mid-stage": (4, "real recsys work, but lightweight, no eval framework mentioned"),
    "Built computer vision models for our product's ima": (3, "JD disqualifier: CV-only, explicitly no NLP/IR exposure"),
    "Worked on time-series forecasting models for suppl": (3, "forecasting/RL, not retrieval or ranking"),
    "Worked on customer-facing predictive modeling for ": (3, "classic tabular ML, not retrieval/ranking"),
    "Built NLP pipelines for sentiment analysis and doc": (4, "real NLP but classification, not retrieval/ranking; self-discloses fine-tuning-only depth"),
    "Owned the ranking layer for an e-commerce search p": (5, "real learning-to-rank ownership, but framed as 'easy bit'"),
    "Trained and shipped multiple ranking models for ou": (7, "solid production ranking + offline/online eval correlation, strong match"),
    "Developed a semantic search feature for an interna": (8, "embeddings + FAISS + measured relevance improvement, core JD match"),
    "Implemented a RAG-based customer support chatbot i": (8, "RAG + embeddings + Pinecone + eval framework, strong JD match"),
    "Built a content recommendation system serving 10M+": (7, "production recsys with A/B-measured impact, strong match"),
    "Built and operated production ML pipelines using M": (6, "production MLOps, relevant infra but not retrieval/ranking-specific"),
    "Fine-tuned LLaMA-2-7B and Mistral-7B variants usin": (9, "LLM fine-tuning (LoRA/QLoRA) + eval harness + production deploy, ideal-candidate match"),
    "Built a RAG-based ranking pipeline serving 50M+ qu": (9, "hybrid retrieval + LLM re-ranker + NDCG/MRR eval framework, ideal-candidate match"),
    "Built and shipped a production recommendation syst": (8, "recsys + explicit cold-start ML technique + measured impact"),
    "Owned the end-to-end ranking pipeline at a recomme": (9, "embeddings + retrieval + LTR + evaluation methodology ownership, ideal match"),
    "Owned the design and rollout of a large-scale sema": (9, "hybrid search migration + measured NDCG improvement + team lead, ideal match"),
    "Led the migration from keyword-based to embedding-": (9, "exact JD scenario: keyword-to-embedding migration with measured results"),
    "Built systems that understand what users are looki": (8, "plain-language but clearly retrieval+ranking+eval ownership -- a 'Tier 5' case"),
    "Shipped the personalization infrastructure: the sy": (8, "plain-language personalization/ranking + eval framework ownership"),
    "Designed the ranking layer for the company's flags": (9, "plain-language but explicit ranking+data+eval ownership at flagship scale"),
    "Owned the search and discovery experience end-to-e": (9, "plain-language but full search/ranking/eval ownership, ideal-candidate sketch"),
    "Led the engineering team building infrastructure t": (8, "plain-language large-scale search infra + ranking calibration + leadership"),
}


def score_description(description: str) -> tuple[float, str]:
    """Look up a job description's relevance score via its stable prefix.
    Falls back to a neutral low score if a description doesn't match any
    known template (shouldn't happen given our exhaustive scan, but kept
    safe in case the held-out evaluation uses different/additional data)."""
    prefix = description[:50] if description else ""
    for known_prefix, (score, reason) in TEMPLATE_SCORES.items():
        if description.startswith(known_prefix[:50]):
            return score, reason
    return 0.0, "unrecognized template (not in our 44-template scan)"


def compute_template_score(career_history: list) -> dict:
    """Score a candidate's full career history using exact template lookup.
    Weighted toward MOST RECENT relevant role, since the JD cares about
    current capability, but a strong past role still counts."""
    if not career_history:
        return {"template_max_score": 0.0, "template_avg_score": 0.0,
                "template_current_role_score": 0.0, "template_scored_roles": []}

    scored_roles = []
    for job in career_history:
        score, reason = score_description(job.get("description", ""))
        scored_roles.append({
            "company": job.get("company"),
            "title": job.get("title"),
            "is_current": job.get("is_current"),
            "score": score,
            "reason": reason,
        })

    max_score = max(r["score"] for r in scored_roles)
    avg_score = sum(r["score"] for r in scored_roles) / len(scored_roles)
    current_role = next((r for r in scored_roles if r["is_current"]), scored_roles[0])

    return {
        "template_max_score": max_score,
        "template_avg_score": round(avg_score, 2),
        "template_current_role_score": current_role["score"],
        "template_scored_roles": scored_roles,
    }


if __name__ == "__main__":
    # Sanity check: every template prefix we hand-classified should match
    # at least one real description in the dataset.
    import json
    from collections import Counter

    desc_counter = Counter()
    with open("candidates.jsonl") as f:
        for line in f:
            c = json.loads(line)
            for job in c.get("career_history", []):
                desc_counter[job.get("description", "")] += 1

    matched_prefixes = set()
    unrecognized = 0
    for desc in desc_counter:
        score, reason = score_description(desc)
        if "unrecognized" in reason:
            unrecognized += 1
        else:
            for prefix in TEMPLATE_SCORES:
                if desc.startswith(prefix[:50]):
                    matched_prefixes.add(prefix)
                    break

    print(f"Unique descriptions in dataset: {len(desc_counter)}")
    print(f"Our hand-classified templates: {len(TEMPLATE_SCORES)}")
    print(f"Templates that matched >=1 real description: {len(matched_prefixes)}")
    print(f"Unrecognized descriptions: {unrecognized}")
    unmatched = set(TEMPLATE_SCORES) - matched_prefixes
    if unmatched:
        print("WARNING -- these hand-classified templates never matched anything:")
        for u in unmatched:
            print(" -", u)
