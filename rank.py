"""
rank.py — SINGLE ENTRYPOINT for the submission.

Per submission_spec.md Section 10.3: "your README must indicate a single
command that produces the submission CSV from the candidates file."

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

This script runs the full pipeline end to end:
  1. feature extraction (features.py)       -- ~30s on 100K rows
  2. scoring (scoring.py)                    -- ~6s on 100K rows
  3. top-100 selection + reasoning generation (logic from rank_and_submit.py)
  4. writes the final CSV

No network calls. No GPU. Designed to comfortably clear the 5-minute /
16GB / CPU-only compute budget in submission_spec.md Section 3 -- see the
timing printed at the end of the run, and README.md for a measured result
on a real machine.
"""

import argparse
import json
import csv
import time
import sys

import pandas as pd

from features import extract_features
from scoring import compute_total_score


def load_candidates(path: str) -> list[dict]:
    candidates = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def build_reasoning(candidate_row: pd.Series, full_record: dict, rank: int) -> str:
    """See rank_and_submit.py for full design notes on why this is built
    as scored, candidate-specific variants rather than one fixed template."""
    profile = full_record["profile"]
    title = profile["current_title"]
    company = profile["current_company"]
    yoe = profile["years_of_experience"]
    location = profile["location"]

    career = full_record.get("career_history", [])
    longest_past_job = max(
        (j for j in career if not j.get("is_current")),
        key=lambda j: j.get("duration_months", 0),
        default=None,
    )

    matched_names = str(candidate_row.get("skills_signal_matched_names", "") or "")
    matched_list = [s for s in matched_names.split("|") if s]

    exp_hits = candidate_row["exp_total_relevance_hits"]
    notice = candidate_row.get("notice_period_days")
    resp_rate = candidate_row.get("recruiter_response_rate")
    days_inactive = candidate_row.get("days_since_active")

    concerns = []
    if candidate_row.get("flag_consulting_only_career"):
        concerns.append("their entire visible career has been at IT-services/consulting firms")
    if candidate_row.get("flag_cv_speech_robotics_background"):
        concerns.append("their background leans computer-vision/speech without clear NLP or retrieval exposure")
    if candidate_row.get("flag_framework_enthusiast_language"):
        concerns.append("some of their language reads more tutorial-level than systems-depth")
    if pd.notna(notice) and notice > 60:
        concerns.append(f"a {int(notice)}-day notice period, longer than the JD's sub-30-day preference")
    if pd.notna(days_inactive) and days_inactive > 60:
        concerns.append(f"hasn't been active on the platform in {int(days_inactive)} days")
    if pd.notna(resp_rate) and resp_rate < 0.3:
        concerns.append("a low recruiter response rate, so reachability is uncertain")
    out_of_band = not (5 <= yoe <= 9)

    lead_candidates = []
    num_roles = len(career)
    if exp_hits >= 10 and num_roles >= 3:
        lead_candidates.append((
            exp_hits,
            f"Across {num_roles} roles including {company}, their career history reads as "
            f"deep, consistent applied-ML/retrieval work rather than a single recent pivot."
        ))
    if longest_past_job and longest_past_job.get("duration_months", 0) >= 24:
        lead_candidates.append((
            longest_past_job["duration_months"] / 4,
            f"Spent {longest_past_job['duration_months']} months at {longest_past_job['company']} "
            f"as {longest_past_job['title']} before moving to {company} as {title} -- "
            f"real tenure, not job-hopping."
        ))
    if matched_list:
        named = ", ".join(matched_list[:2])
        lead_candidates.append((
            len(matched_list) * 2,
            f"{title} at {company} ({yoe:.1f} yrs total) with {named} explicitly on their "
            f"skill list, backed by matching language in their actual job descriptions."
        ))
    if exp_hits >= 6:
        lead_candidates.append((
            exp_hits * 1.5,
            f"{yoe:.1f} years in, currently {title} at {company} -- their job descriptions "
            f"name specific retrieval/ranking work, not just adjacent buzzwords."
        ))
    if company and yoe >= 7:
        lead_candidates.append((
            yoe,
            f"Seven-plus years in, now {title} at {company}; tenure alone puts them past the "
            f"point where this would read as an opportunistic pivot into AI."
        ))
    if not lead_candidates:
        lead_candidates.append((0, (
            f"{title} at {company}, {yoe:.1f} years of experience, with some JD-relevant "
            f"signal in their career history."
        )))

    lead_candidates.sort(key=lambda x: -x[0])
    lead = lead_candidates[0][1]

    pieces = [lead]
    if out_of_band:
        direction = "above" if yoe > 9 else "below"
        pieces.append(f"At {yoe:.1f} years they sit {direction} the JD's 5-9yr band, "
                       f"but the JD explicitly treats that as flexible given strong signals elsewhere.")
    if "pune" not in location.lower() and "noida" not in location.lower():
        relocate = full_record.get("redrob_signals", {}).get("willing_to_relocate")
        if relocate:
            pieces.append(f"Based in {location}, not Pune/Noida, but marked willing to relocate.")
        else:
            pieces.append(f"Based in {location}, outside the preferred Pune/Noida hubs.")
    if concerns:
        if len(concerns) == 1:
            pieces.append(f"One thing to weigh: {concerns[0]}.")
        else:
            pieces.append(f"Worth weighing: {'; '.join(concerns)}.")
    elif rank <= 10:
        pieces.append("No red flags surfaced in our checks.")

    return " ".join(pieces)


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against the Redrob JD.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="Path to write the output submission CSV")
    args = parser.parse_args()

    t0 = time.time()

    print(f"[1/4] Loading candidates from {args.candidates} ...", file=sys.stderr)
    candidates = load_candidates(args.candidates)
    print(f"      Loaded {len(candidates)} candidates ({time.time()-t0:.1f}s elapsed)", file=sys.stderr)

    print("[2/4] Extracting features ...", file=sys.stderr)
    feature_rows = [extract_features(c) for c in candidates]
    df = pd.DataFrame(feature_rows)
    print(f"      Done ({time.time()-t0:.1f}s elapsed)", file=sys.stderr)

    print("[3/4] Scoring ...", file=sys.stderr)
    results = df.apply(lambda row: compute_total_score(row), axis=1, result_type="expand")
    df = pd.concat([df, results], axis=1)
    df_sorted = df.sort_values(
        by=["total_score", "experience_score", "candidate_id"],
        ascending=[False, False, True]
    ).reset_index(drop=True)
    top100 = df_sorted.head(100).copy()
    print(f"      Done ({time.time()-t0:.1f}s elapsed)", file=sys.stderr)

    print("[4/4] Generating reasoning + writing CSV ...", file=sys.stderr)
    top_ids = set(top100["candidate_id"])
    full_records = {c["candidate_id"]: c for c in candidates if c["candidate_id"] in top_ids}

    rows_out = []
    for i, row in top100.iterrows():
        rank = i + 1
        cid = row["candidate_id"]
        reasoning = build_reasoning(row, full_records[cid], rank)
        rows_out.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(float(row["total_score"]), 4),
            "reasoning": reasoning,
        })

    scores_seq = [r["score"] for r in rows_out]
    assert all(scores_seq[i] >= scores_seq[i + 1] for i in range(len(scores_seq) - 1)), \
        "Score is not non-increasing with rank!"

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        w.writeheader()
        w.writerows(rows_out)

    elapsed = time.time() - t0
    print(f"      Wrote {args.out} ({len(rows_out)} rows)", file=sys.stderr)
    print(f"\nTOTAL TIME: {elapsed:.1f}s (budget: 300s)", file=sys.stderr)
    if elapsed > 300:
        print("WARNING: exceeded the 5-minute compute budget!", file=sys.stderr)


if __name__ == "__main__":
    main()
