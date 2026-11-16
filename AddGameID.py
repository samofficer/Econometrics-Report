import pandas as pd
from pathlib import Path


# 1. File locations
folder = Path(
    r"C:\Users\Admin\OneDrive - University of Toronto\School\MetricOdds"
)

odds_path = folder / "NFL_prop_analysis.xlsx"
pbp_path = folder / "nfl_playbyplay_full_teams_updated(RAW).csv"

# Separate output file for checking the matches.
output_path = folder / "PairedOdds_with_Game_ID.xlsx"


# 2. Read the paired odds and play-by-play data
paired = pd.read_excel(
    odds_path,
    sheet_name="PairedOdds",
    engine="openpyxl"
)

pbp = pd.read_csv(
    pbp_path,
    usecols=["Game ID", "Home Team", "Away Team", "Wall Clock"]
)


# 3. Convert the play-by-play timestamps to UTC datetimes
pbp["Wall Clock"] = pd.to_datetime(
    pbp["Wall Clock"],
    utc=True,
    format="mixed"
)


# 4. Create one record per game
# Group by Game ID and both teams.
# min() finds the first recorded play time for each game.
games = (
    pbp.groupby(
        ["Game ID", "Home Team", "Away Team"],
        as_index=False
    )["Wall Clock"]
    .min()
    .rename(columns={
        "Game ID": "Game_ID",
        "Wall Clock": "First_play"
    })
)

# Stop if a game lacks a usable time or has conflicting team assignments.
assert games["First_play"].notna().all(), "A game has no recorded time."
assert games["Game_ID"].is_unique, "A Game ID has multiple team combinations."


# 5. Prepare the odds timestamps for matching
# Keep these helper columns separate from the original paired data.
odds_times = paired[["Home Team", "Away Team"]].copy()

odds_times["Odds_time"] = pd.to_datetime(
    paired["Timestamp"],
    utc=True
)

# Remember the original row positions before sorting.
odds_times["Row_order"] = range(len(paired))


# 6. Match each odds observation to a game
matched = pd.merge_asof(
    # Both inputs must be sorted by their matching time.
    odds_times.sort_values("Odds_time"),
    games.sort_values("First_play"),

    # Time columns used for the comparison.
    left_on="Odds_time",
    right_on="First_play",

    # Both team names must match exactly.
    by=["Home Team", "Away Team"],

    # Find the closest first-play time, before or after the odds time.
    direction="nearest",

    # Reject matches farther than 12 hours away.
    tolerance=pd.Timedelta(hours=12)
)


# 7. Restore the original row order and check the matches
matched = matched.sort_values("Row_order")

unmatched = matched["Game_ID"].isna().sum()

if unmatched > 0:
    raise ValueError(
        f"{unmatched} rows could not be matched. No output file was saved."
    )

assert len(matched) == len(paired), "The row count changed."


# 8. Add Game_ID to the original paired data
# Use the values in their restored order, rather than matching index labels.
paired["Game_ID"] = matched["Game_ID"].to_numpy()


# 9. Save a NEW workbook containing the PairedOdds sheet
paired.to_excel(
    output_path,
    sheet_name="PairedOdds",
    index=False,
    engine="openpyxl"
)


# 10. Display the results
print("Rows saved:", len(paired))
print("Unique games:", paired["Game_ID"].nunique())
print("Unmatched rows:", unmatched)
print("Saved to:", output_path)

print(
    paired[
        ["Home Team", "Away Team", "Timestamp", "Game_ID"]
    ].head(10).to_string(index=False)
)