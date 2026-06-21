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

### 1. Feature extraction (`features.py`)

Turns each candidate's messy JSON profile into structured signals. The key
design decision: **skills-list keyword matches and career-history keyword
matches are computed and scored separately**, never blended into one number
at this stage. This directly addresses the JD's own warning that the
"wrong" answer to this challenge is rewarding candidates whose *skills
list* contains the most AI buzzwords — the JD explicitly wants candidates
whose *career history* (what they actually did) shows real retrieval/
ranking/ML-production work, even if their skills list doesn't list the
trendy terms (the "plain-language Tier 5" case).

Also computed here:
- **Red flags** matching the JD's explicit disqualifier list: pure-research
  background, 100%-consulting-firm career with no product-company
  experience, CV/speech/robotics background *without* NLP/IR exposure,
  framework-tutorial-level language, seniority-label title-chasing.
- **Honeypot checks**: internal profile-consistency checks (e.g. "expert"
  skill proficiency claimed with 0 months of use) that flag the dataset's
  ~80 deliberately-impossible profiles.
- **Behavioral signal summary** from the 23 `redrob_signals` fields:
  recency of activity, recruiter response rate, notice period, etc.

### 2. Scoring (`scoring.py`)

Combines the features into one explainable 0–1 score using a weighted sum
of named, interpretable components (not a trained black-box model):

| Component | Max points | Rationale |
|---|---|---|
| `experience_score` | 40 | Dominant weight — real career-history evidence |
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
- Honeypot detection currently relies on two confirmed signature checks
  (expert-proficiency-with-zero-duration skills; years-of-experience vs.
  career-history-duration mismatch). We verified these catch a combined
  ~68 candidates, in the right ballpark versus the ~80 honeypots the
  README describes, but we have not exhaustively confirmed there's no
  double-counting or that every honeypot archetype is covered.
- The location/relocation scoring and salary-fit are intentionally simple
  (the JD doesn't give us salary band data to compare against, so expected
  salary isn't currently used as a scoring input at all).

## AI tools used

Declared per spec Section 10.2/10.4: built with assistance from Claude
(Anthropic) for code generation, debugging, and design discussion. All
architectural decisions, rule corrections (see git history for the
seniority-ranking bug fix and the reasoning-deduplication fix), and final
review were done by the submitting participant.
