# Redrob Hackathon — Intelligent Candidate Discovery & Ranking

Track 1 submission: ranks 100,000 candidates against the Senior AI Engineer
job description, producing a top-100 ranked CSV with grounded, per-candidate
reasoning.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+. No GPU, no network access, no API keys needed to run
the ranking step.

## Reproduce the submission (single command)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

- `--candidates`: path to the 100,000-row `candidates.jsonl` file from the
  hackathon bundle (not included in this repo — see "Getting the data" below).
- `--out`: where to write the ranked CSV.

Measured runtime: **~40 seconds** on a CPU-only machine with 100,000 input
rows (well under the 5-minute budget). Peak memory stays under 1 GB — the
whole candidate pool is loaded once into a pandas DataFrame, no batching
needed at this data size.

## Getting the data

`candidates.jsonl` (~465MB uncompressed) is not committed to this repo —
it's the hackathon-provided dataset. Place it at the repo root (or pass its
path via `--candidates`) before running `rank.py`.

## How it works

The pipeline has no LLM calls, no GPU, no network access during ranking —
required by submission_spec.md Section 3. It's a transparent, rule-based
pipeline in three stages, all in `rank.py` (which imports `features.py` and
`scoring.py`):

### 1. Feature extraction (`features.py`, `template_relevance.py`)

**Key discovery, made while hand-validating our scores against real candidate
records**: this dataset's 300,171 `career_history` descriptions are not unique
free text — they're drawn from a fixed pool of only **44 templates**, reused
across candidates (verified: the most common template alone appears 25,515
times; zero descriptions fall outside this set of 44, confirmed by exhaustive
scan). This is a fundamentally different problem than free-text NLP matching:
it's a classification problem with a known, finite label set.

So instead of keyword/substring matching against career-history text (which is
fuzzy and can be fooled by partial overlaps or miss real signal phrased
differently than expected), `template_relevance.py` manually classifies **all
44 templates exactly once**, scored 0–10 against `job_description.md`'s
must-haves, nice-to-haves, and explicit disqualifiers, with a one-line reason
per template. Any candidate's relevance is then an exact lookup, not a guess.

The 44 templates form a clear tiered structure once read in full:
- **Templates 1–9** (~25K each): completely unrelated functions (sales,
  support, marketing, accounting, design, mechanical engineering) — score 0.
- **Templates 10–15** (~10K each): tech-adjacent, no ML (DevOps, mobile,
  frontend, generic backend, QA) — score 1.
- **Templates 16–21** (~1.8K each): data engineering/analytics, explicitly
  *not* modeling work — score 2.
- **Templates 22–27** (330–390 each): **the trap zone** — genuine ML keywords
  appear, but each template's own text discloses the exact limitation the JD
  warns against (CV-only with no NLP, "production-side engineer, not
  modeling," forecasting/tabular ML rather than retrieval/ranking) — score 3–4.
- **Template 28** (78): real ranking work, framed as secondary — score 5.
- **Templates 29–33** (57–65 each): solid production ranking/recsys/MLOps —
  score 6–7.
- **Templates 34–39** (8–64 each): RAG, LLM fine-tuning (LoRA/QLoRA), hybrid
  retrieval, explicit NDCG/MRR evaluation-framework ownership — score 8–9,
  matching the JD's stated ideal-candidate profile almost exactly.
- **Templates 40–44** (2–6 each): deliberately plain-language descriptions of
  the same senior ranking/retrieval/eval ownership — score 8–9. We treat these
  as the JD's explicitly-flagged "Tier 5" case: real, senior, relevant work
  described without buzzwords, which a naive keyword matcher would likely
  under-score.

This also separately surfaces (still computed, kept for cross-validation):
skills-list keyword matches (kept deliberately weak relative to the
template-based experience score, since the JD explicitly warns against
rewarding keyword-stuffed skill lists over real career-history evidence),
red flags matching the JD's explicit disqualifier list, honeypot checks
(internal profile-consistency, e.g. "expert" skill proficiency claimed with
0 months of use), and a behavioral signal summary from the 23
`redrob_signals` fields.

### 2. Scoring (`scoring.py`)

Combines the features into one explainable 0–1 score using a weighted sum
of named, interpretable components (not a trained black-box model):

| Component | Max points | Rationale |
|---|---|---|
| `experience_score` | 40 | Exact template-lookup score (see above), 70% current role / 30% best-ever |
| `skills_score` | 15 | Deliberately capped below experience — supportive only |
| `yoe_fit_score` | 10 | Smooth falloff outside the JD's 5–9yr band (JD states this is flexible) |
| `location_score` | 5 | Pune/Noida preferred, other Tier-1 India cities okay, outside India lowest (no visa sponsorship) |
| `availability_score` | 10 | Active, responsive, reasonable notice period |
| `red_flag_penalty` | subtracted | JD disqualifiers, weighted by how strongly the JD states them |

Honeypot-flagged candidates are hard-excluded (score forced to 0) rather
than just penalized, since Stage 3 disqualifies on honeypot rate in the
top 100 regardless of overall composite score — a false negative here is
much more costly than a false positive.

We chose a transparent weighted-sum approach over a trained ranking model
deliberately: it fits the compute budget trivially, and — more importantly —
it's something we can fully explain and defend line-by-line in the Stage 5
interview, which an opaque model would make much harder.

### 3. Ranking + reasoning (`rank.py`)

Selects the top 100 by `total_score`, breaking ties first by
`experience_score` (secondary signal) and then by `candidate_id` ascending
(deterministic, as required). For each of the top 100, generates a 1–2
sentence reasoning string built from real profile fields only — every
fact cited (years of experience, employer, named skills, behavioral
signals) is read directly from that candidate's record, not invented.

The reasoning generator scores several candidate-specific "lead fact"
options (deepest past tenure, named matching skills, total relevant-
experience signal strength, years of experience) and picks the most
distinctive one *per candidate*, specifically to avoid templated,
identical-sounding output across the 100 rows — this was an actual bug we
caught and fixed during development (see git history) by checking the
distribution of opening phrases across all 100 reasoning strings.

## Validating the output

```bash
python self_check.py
```

This checks the hard rules from `submission_spec.md` Sections 2, 3, and 6
(exactly 100 rows, ranks 1–100 each used once, no duplicate candidate_ids,
all candidate_ids exist in the source file, scores non-increasing with
rank, etc). **This is a self-written check** based on reading the spec —
run the official `validate_submission.py` as well once available; it is
authoritative, this script is a development aid.

## Known limitations / honest notes

- `flag_title_chasing` (career-history seniority-label escalation across
  short stints) almost never fires on this dataset — we verified this
  isn't a bug in our detector; only ~58 of 100,000 candidates even have
  explicit seniority words (Senior/Staff/Principal/etc.) across 3+
  consecutive jobs, and none of those show genuine upward escalation
  without a functional role change. The rule is kept in case it matters
  on held-out data, but it's not doing meaningful work on what we can see.
- **Honeypot/trap investigation, done systematically rather than guessed:**
  We hunted for each archetype the README names, checking actual prevalence
  in the data before deciding whether to build a rule for it:
  - *"Subtly impossible profiles"*: confirmed via two signatures — "expert"
    skill proficiency claimed with 0 months duration, and years-of-experience
    vs. career-history-duration mismatches >36 months. Combined catch: ~68
    candidates, roughly matching the README's "~80 honeypots."
  - *Education-ends-after-first-job and salary-min>max anomalies*: found at
    huge scale (~19% and ~19% of the full 100K respectively) — far too
    common to be rare honeypots. Concluded these are synthetic-data
    generation noise (e.g. min/max salary fields randomly swapped), not
    deliberate traps, and did **not** build exclusion rules for them, since
    doing so would have wrongly penalized thousands of normal candidates.
  - *"Keyword stuffers"*: checked specifically for candidates with multiple
    "expert"-rated AI/retrieval-specific skills paired with career history
    that scores 0–2 on our template lookup (i.e. skills claimed with zero
    real backing). Found **zero** candidates matching this dangerous
    pattern — the handful of candidates with mismatched generic-tech
    "expert" skills (e.g. SAP, Hadoop) on irrelevant careers don't actually
    threaten the ranking, since those skills aren't in our JD-relevant
    vocabulary anyway and score near-zero on `skills_score` regardless.
  - *"Behavioral twins"*: found 7 closely-matched pairs within our
    high-relevance candidate pool (same title/years-of-experience/template
    score) that differ mainly on `redrob_signals` fields. Spot-checked one
    pair directly: our `availability_score` and template-based
    `experience_score` together correctly rank the more reachable, more
    currently-active candidate higher — this is an emergent property of
    the scoring design (using *current*-role template score plus
    behavioral signals), not a rule we had to add specifically.
  - We have **not** exhaustively proven there are no honeypot archetypes
    left uncaught — only that the ones we hypothesized and checked for are
    either handled correctly or genuinely absent at meaningful scale.
- The location/relocation scoring and salary-fit are intentionally simple
  (the JD doesn't give us salary band data to compare against, so expected
  salary isn't currently used as a scoring input at all).

## AI tools used

Declared per spec Section 10.2/10.4: built with assistance from Claude
(Anthropic) for code generation, debugging, and design discussion. All
architectural decisions, rule corrections (see git history for the
seniority-ranking bug fix and the reasoning-deduplication fix), and final
review were done by the submitting participant.
