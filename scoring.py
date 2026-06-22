"""
scoring.py — Step 2: combine the features from features.py into ONE score
per candidate, used to produce the final top-100 ranking.

DESIGN PRINCIPLE: every component below is a simple, interpretable number.
No black-box model. This is deliberate — Stage 5 of the hackathon is a
live interview where we have to defend exactly why each candidate is
ranked where they are. A pile of weighted, named components is something
a human can walk through; an opaque trained model is much harder to defend
in 30 minutes, and the compute constraints (no GPU, no network, 5 min CPU
budget for 100K rows) make a heavy model impractical anyway.

The final score is a weighted sum of:
  1. experience_score   (0-40 pts) -- by far the most important: does their
                                       ACTUAL career history show JD-relevant work?
  2. skills_score        (0-15 pts) -- supportive signal from skills list,
                                       intentionally capped low relative to
                                       experience_score per the JD's explicit
                                       warning against keyword-stuffing wins.
  3. yoe_fit_score       (0-10 pts) -- how well years_of_experience matches
                                       the JD's flexible 5-9 yr band.
  4. location_score      (0-5 pts)  -- Pune/Noida preferred, other listed
                                       Tier-1 cities acceptable, elsewhere
                                       in India lower, outside India lowest
                                       (JD: no visa sponsorship).
  5. availability_score  (0-10 pts) -- behavioral signals: active, responsive,
                                       open to work, reasonable notice period.
  6. red_flag_penalty    (subtracted) -- explicit JD disqualifiers reduce
                                       score rather than zero it outright,
                                       since the JD's own language is
                                       "probably not move forward" (strong
                                       signal, not an absolute rule) for some
                                       of these.
  7. honeypot_exclusion  -- if any honeypot signal fires, score is forced
                                       to effectively zero so these can never
                                       land in the top 100.

Total before honeypot override: 0-80 points, normalized to 0-1 at the end.
"""

import pandas as pd

PREFERRED_CITIES = {"pune", "noida"}
OTHER_TIER1_CITIES = {"hyderabad", "mumbai", "delhi", "delhi ncr", "bengaluru",
                       "bangalore", "gurugram", "gurgaon"}


def experience_score(row) -> float:
    """0-40 points. NOW USES EXACT TEMPLATE LOOKUP (template_relevance.py)
    rather than fuzzy keyword counting. We discovered this dataset's career
    descriptions are drawn from only 44 fixed templates (confirmed: 300,171
    descriptions, 44 unique strings), so we could read and score every single
    one by hand instead of guessing relevance from keyword overlap.

    Uses template_current_role_score most heavily (what they do NOW matters
    most) blended with template_max_score (credit for having done relevant
    work even if their current role is a step in a different direction).
    """
    current = row.get("template_current_role_score", 0) or 0
    best_ever = row.get("template_max_score", 0) or 0
    # 70% weight on current role, 30% on best-ever -- rewards relevant
    # current work most, but still credits someone who did great ranking
    # work two jobs ago and is now in a slightly more generic role.
    blended = 0.7 * current + 0.3 * best_ever
    score = 40 * (blended / 10.0)  # template scores are 0-10, scale to 0-40
    return round(score, 2)


def skills_score(row) -> float:
    """0-15 points. Deliberately capped lower than experience_score so a
    candidate can't out-score real career experience purely by listing
    buzzwords. Uses weighted skill score (proficiency-weighted) from features.py."""
    weighted = row["skills_signal_weighted_score"]
    # cap contribution; weighted scores above ~30 get full marks
    score = min(15, 15 * (weighted / 30))
    return round(score, 2)


def yoe_fit_score(row) -> float:
    """0-10 points. JD wants 5-9 years but is explicit that this is a
    flexible band, not a hard requirement -- 'we'll seriously consider
    candidates outside the band if other signals are strong.' So we use
    a smooth falloff rather than a hard cutoff."""
    yoe = row["years_of_experience"]
    if 5 <= yoe <= 9:
        return 10.0
    # smooth falloff outside the band, gentler than a cliff
    distance = (5 - yoe) if yoe < 5 else (yoe - 9)
    score = max(0.0, 10 - distance * 1.5)
    return round(score, 2)


def location_score(row) -> float:
    """0-5 points. Pune/Noida preferred; other major Indian tech hubs okay;
    elsewhere in India lower; outside India lowest since JD states no visa
    sponsorship (not a hard disqualifier -- 'case-by-case' per JD -- so we
    don't zero it, just rank it lowest)."""
    loc = str(row["location"]).lower()
    country = str(row["country"]).lower()

    if country != "india":
        return 0.5  # JD: "case-by-case" for outside India, no sponsorship
    if any(city in loc for city in PREFERRED_CITIES):
        return 5.0
    if any(city in loc for city in OTHER_TIER1_CITIES):
        return 3.5
    return 2.0  # elsewhere in India -- JD doesn't rule this out, willing_to_relocate matters more


def availability_score(row) -> float:
    """0-10 points. Behavioral signals from redrob_signals. A perfect-on-paper
    candidate who is inactive / unresponsive is, per the JD's own instructions,
    'not actually available' for hiring purposes and should be down-weighted."""
    score = 0.0

    if row.get("open_to_work_flag") is True or row.get("open_to_work_flag") == "True":
        score += 3.0

    days_inactive = row.get("days_since_active")
    if pd.notna(days_inactive):
        if days_inactive <= 30:
            score += 3.0
        elif days_inactive <= 90:
            score += 1.5
        # else 0 -- inactive 90+ days contributes nothing

    resp_rate = row.get("recruiter_response_rate")
    if pd.notna(resp_rate):
        score += 2.0 * float(resp_rate)  # 0-2 pts scaled by response rate

    notice = row.get("notice_period_days")
    if pd.notna(notice):
        if notice <= 30:
            score += 2.0
        elif notice <= 60:
            score += 1.0
        # 60+ day notice: JD says "bar gets higher", contributes 0 here

    return round(min(score, 10.0), 2)


def red_flag_penalty(row) -> float:
    """Points subtracted for explicit JD disqualifiers. Not an automatic zero
    for most of these -- JD's own language is 'probably not move forward',
    i.e. strongly negative, not an absolute bar (except pure-research, which
    the JD states with no hedging: 'we will not move forward')."""
    penalty = 0.0
    if row.get("flag_research_only"):
        penalty += 35.0  # JD states this with no hedge -- effectively disqualifying
    if row.get("flag_consulting_only_career"):
        penalty += 18.0  # "we've had bad fit experiences in both directions"
    if row.get("flag_cv_speech_robotics_background"):
        penalty += 15.0  # "you'd be re-learning fundamentals here"
    if row.get("flag_framework_enthusiast_language"):
        penalty += 6.0   # softer signal, JD calls this "fine but not what we need"
    if row.get("flag_title_chasing"):
        penalty += 12.0  # rare in this dataset, but keep the check
    return penalty


def is_honeypot(row) -> bool:
    """Hard exclusion check. If ANY of these fire, this candidate must not
    appear in the top 100 -- the honeypot rate check at Stage 3 is a
    disqualifying gate (>10% honeypot rate in top 100 = disqualified),
    so false negatives here are much costlier than false positives."""
    if row.get("honeypot_expert_zero_duration_count", 0) > 0:
        return True
    if row.get("honeypot_yoe_history_mismatch") in (True, "True"):
        return True
    return False


def compute_total_score(row) -> dict:
    """Combine all components into one final score (0-1 normalized) plus
    a breakdown dict, so we can show our work in the reasoning column later."""
    if is_honeypot(row):
        return {
            "total_score": 0.0,
            "is_honeypot": True,
            "experience_score": 0.0, "skills_score": 0.0, "yoe_fit_score": 0.0,
            "location_score": 0.0, "availability_score": 0.0, "red_flag_penalty": 0.0,
        }

    exp = experience_score(row)
    skl = skills_score(row)
    yoe = yoe_fit_score(row)
    loc = location_score(row)
    avail = availability_score(row)
    penalty = red_flag_penalty(row)

    raw_total = exp + skl + yoe + loc + avail - penalty
    raw_total = max(raw_total, 0.0)  # don't go negative
    normalized = round(raw_total / 80.0, 4)  # 80 = max possible before penalty

    return {
        "total_score": normalized,
        "is_honeypot": False,
        "experience_score": exp,
        "skills_score": skl,
        "yoe_fit_score": yoe,
        "location_score": loc,
        "availability_score": avail,
        "red_flag_penalty": penalty,
    }


if __name__ == "__main__":
    df = pd.read_csv("features_full.csv", low_memory=False)

    results = df.apply(lambda row: compute_total_score(row), axis=1, result_type="expand")
    df = pd.concat([df, results], axis=1)

    df_sorted = df.sort_values("total_score", ascending=False)
    top100 = df_sorted.head(100)

    print("Top 15 by total_score:")
    print(top100[["candidate_id", "current_title", "current_company",
                   "total_score", "experience_score", "skills_score",
                   "yoe_fit_score", "location_score", "availability_score",
                   "red_flag_penalty"]].head(15).to_string(index=False))

    print()
    print("Honeypots excluded:", df["is_honeypot"].sum())
    print("Score range in top 100:", top100["total_score"].min(), "-", top100["total_score"].max())

    df_sorted.to_csv("scored_full.csv", index=False)
    print("\nSaved scored_full.csv")
