"""
self_check.py — Validates submission.csv against the rules in
submission_spec.md Sections 2, 3, and 6, since the official
validate_submission.py hasn't been provided yet. Run the real validator
too, once you have it -- this is a stand-in, not a replacement.
"""

import csv
import json

ERRORS = []
WARNINGS = []


def check(condition, message, is_error=True):
    if not condition:
        (ERRORS if is_error else WARNINGS).append(message)


def main():
    with open("submission.csv") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # --- Section 2: required columns, in order ---
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    check(reader.fieldnames == expected_cols,
          f"Columns must be exactly {expected_cols} in order, got {reader.fieldnames}")

    # --- Section 3: exactly 100 rows ---
    check(len(rows) == 100, f"Must have exactly 100 data rows, got {len(rows)}")

    # --- Ranks: 1-100, each exactly once ---
    ranks = [int(r["rank"]) for r in rows]
    check(sorted(ranks) == list(range(1, 101)),
          f"Ranks must be exactly 1-100 each used once. Got min={min(ranks)}, max={max(ranks)}, "
          f"duplicates={len(ranks) - len(set(ranks))}")

    # --- candidate_id: unique, and must exist in candidates.jsonl ---
    cids = [r["candidate_id"] for r in rows]
    check(len(cids) == len(set(cids)), "Duplicate candidate_id found in submission")

    valid_ids = set()
    with open("candidates.jsonl") as f:
        for line in f:
            valid_ids.add(json.loads(line)["candidate_id"])
    missing = [c for c in cids if c not in valid_ids]
    check(len(missing) == 0, f"candidate_ids not found in candidates.jsonl: {missing[:5]}")

    # --- score: monotonically non-increasing with rank ---
    rows_sorted_by_rank = sorted(rows, key=lambda r: int(r["rank"]))
    scores = [float(r["score"]) for r in rows_sorted_by_rank]
    non_increasing = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    check(non_increasing, "score is not monotonically non-increasing as rank increases")

    # --- Section 6 common rejections ---
    check(min(ranks) == 1, "Ranks must start at 1, not 0")
    check(len(set(scores)) > 1, "All scores are identical -- model isn't differentiating "
          "(this is an explicit common-rejection example in the spec)", is_error=False)

    # --- Reasoning checks (soft, since optional, but let's verify quality basics) ---
    empty_reasoning = sum(1 for r in rows if not r["reasoning"].strip())
    check(empty_reasoning == 0, f"{empty_reasoning} rows have empty reasoning", is_error=False)

    identical_reasoning = len(rows) - len(set(r["reasoning"] for r in rows))
    check(identical_reasoning == 0,
          f"{identical_reasoning} duplicate reasoning strings found", is_error=False)

    # --- Print results ---
    print(f"Checked {len(rows)} rows.\n")
    if ERRORS:
        print(f"❌ {len(ERRORS)} ERROR(S):")
        for e in ERRORS:
            print("  -", e)
    else:
        print("✅ No hard errors found against the rules we could check.")

    if WARNINGS:
        print(f"\n⚠️  {len(WARNINGS)} WARNING(S):")
        for w in WARNINGS:
            print("  -", w)

    print("\nNOTE: this is a self-written check based on reading submission_spec.md.")
    print("Run the OFFICIAL validate_submission.py once you have it -- do not rely on this alone.")


if __name__ == "__main__":
    main()
