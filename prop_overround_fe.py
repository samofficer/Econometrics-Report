"""
Sportsbook pricing of NFL passing props: pre-kickoff panel, fixed-effects
regression, and the play-by-play match.

Usage:
    python prop_overround_fe.py PairedOdds_with_Game_ID.xlsx nfl_playbyplay_full_teams_updatedRAW.csv

Requires: pandas, numpy, statsmodels, openpyxl
"""
import re
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ODDS_PATH, PBP_PATH = sys.argv[1], sys.argv[2]
REF_BOOK, REF_MARKET = "Caesars", "yds"


# ---------------------------------------------------------------------------
# 1. Load odds and separate pre-kickoff quotes from in-play quotes
# ---------------------------------------------------------------------------
odds = pd.read_excel(ODDS_PATH).dropna(subset=["Overround"])
odds["mk"] = odds["Market"].str.replace("player_pass_", "")

# Timestamp is the snapshot time in UTC. Market Update is in Eastern time,
# so convert it to UTC before comparing the two.
odds["upd_utc"] = (
    odds["Market Update"]
    .dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    .dt.tz_convert("UTC")
    .dt.tz_localize(None)
)

# Snapshots start at kickoff and run every 5 minutes, so the first snapshot
# of each game is the kickoff time.
odds["kickoff"] = odds.groupby("Game_ID")["Timestamp"].transform("min")
odds["mins_from_kickoff"] = (odds["upd_utc"] - odds["kickoff"]).dt.total_seconds() / 60
odds["pregame"] = odds["mins_from_kickoff"] <= 0

print("Share of rows quoted before kickoff, by book")
print(odds.groupby("Bookmaker")["pregame"].mean().round(3), "\n")


# ---------------------------------------------------------------------------
# 2. Build the closing main-line panel: one row per book x player x market x game
# ---------------------------------------------------------------------------
key = ["Game_ID", "Player", "Market", "Bookmaker"]
pre = odds[odds["pregame"]].copy()

# Keep each book's last update before kickoff for that player-market-game.
pre = pre[pre["upd_utc"] == pre.groupby(key)["upd_utc"].transform("max")]
pre = pre.drop_duplicates(key + ["Point"])

# Some books post alternate lines. The main line is the one whose de-vigged
# over probability is closest to 50%.
pre["p_over_fair"] = pre["BreakEvenOver"] / (pre["BreakEvenOver"] + pre["BreakEvenUnder"])
pre["dist_from_even"] = (pre["p_over_fair"] - 0.5).abs()
panel = pre.sort_values("dist_from_even").drop_duplicates(key).copy()

# William Hill (US) is the Caesars feed under another name.
panel = panel[panel["Bookmaker"] != "William Hill (US)"].copy()

# A product is one quarterback-market-game. Every book quoting it is pricing
# the same bet.
panel["product"] = panel["Game_ID"] + "|" + panel["Player"] + "|" + panel["mk"]

print("Mean closing overround, main line")
print(panel.groupby(["mk", "Bookmaker"])["Overround"].mean().unstack().round(4), "\n")


# ---------------------------------------------------------------------------
# 3. Fixed-effects regression
#    Overround = book + book x market + product FE + error, clustered by game
# ---------------------------------------------------------------------------
# Products quoted by a single book carry no within-product variation.
reg = panel[panel.groupby("product")["Bookmaker"].transform("size") >= 2].copy()

# Book dummies (reference book omitted).
book_d = pd.get_dummies(reg["Bookmaker"]).astype(float).drop(columns=REF_BOOK)

# Book x market dummies (reference book and reference market omitted).
inter = pd.get_dummies(reg["Bookmaker"] + "|" + reg["mk"]).astype(float)
inter = inter[[c for c in inter.columns
               if not c.startswith(REF_BOOK + "|") and not c.endswith("|" + REF_MARKET)]]

X = pd.concat([book_d, inter], axis=1)

# Within transformation: subtracting each product's mean from y and from every
# regressor is equivalent to including one dummy per product.
def demean(df, groups):
    return df - df.groupby(groups).transform("mean")

y_w = reg["Overround"] - reg.groupby("product")["Overround"].transform("mean")
X_w = demean(X, reg["product"])

fe = sm.OLS(y_w, X_w).fit(cov_type="cluster", cov_kwds={"groups": reg["Game_ID"]})

print(f"Observations: {len(reg)}, products: {reg['product'].nunique()}, "
      f"games (clusters): {reg['Game_ID'].nunique()}")
print("Coefficients in percentage points of overround")
print(pd.DataFrame({"coef": fe.params * 100, "se": fe.bse * 100, "t": fe.tvalues}).round(3), "\n")

# Joint test that every book x market term is zero.
def wald(columns):
    R = np.zeros((len(columns), X.shape[1]))
    for i, c in enumerate(columns):
        R[i, X.columns.get_loc(c)] = 1
    w = fe.wald_test(R, scalar=True, use_f=True)
    return float(w.statistic), float(w.pvalue)

F, p = wald(list(inter.columns))
print(f"All book x market terms = 0:  F = {F:.1f}, p = {p:.2e}")
for b in book_d.columns:
    F, p = wald([c for c in inter.columns if c.startswith(b + "|")])
    print(f"  {b:<11} F = {F:7.1f}, p = {p:.2e}")

within_r2 = 1 - fe.ssr / (y_w ** 2).sum()
book_only = sm.OLS(y_w, demean(book_d, reg["product"])).fit()
print(f"Within R2 with interactions: {within_r2:.3f}; "
      f"book effects only: {1 - book_only.ssr / (y_w ** 2).sum():.3f}\n")


# ---------------------------------------------------------------------------
# 4. Play-by-play: quarterback stat lines from the play descriptions
# ---------------------------------------------------------------------------
pbp = pd.read_csv(PBP_PATH, low_memory=False)
passes = pbp[pbp["Play Type"] == "pass"]

PASSER = re.compile(r"([A-Z][\w']*\.[\w'\-]+) (pass|spikes)")

def parse_pass(desc):
    """Return the passer and stat contributions of one pass play, or None."""
    # If a review overturned the call, the text after it is the final ruling.
    if "overturned." in desc:
        desc = desc.split("overturned.", 1)[-1]
    if "No Play" in desc:                       # nullified by penalty
        return None
    m = PASSER.search(desc)
    if not m:                                   # sacks and scrambles have no "X pass"
        return None
    qb = m.group(1)
    if m.group(2) == "spikes":                  # a spike is an incomplete attempt
        return dict(qb=qb, att=1, cmp=0, yds=0, td=0, ints=0)

    incomplete = "incomplete" in desc
    intercepted = "INTERCEPTED" in desc
    complete = (" complete" in desc) and not incomplete
    if not (incomplete or intercepted or complete):
        return None

    yards = 0
    if complete:
        g = re.search(r"Gain of (-?\d+) yards", desc)
        yards = int(g.group(1)) if g else 0
    td = int(complete and "TOUCHDOWN" in desc and "FUMBLES" not in desc)
    return dict(qb=qb, att=1, cmp=int(complete), yds=yards, td=td, ints=int(intercepted))

rows = []
for game_id, desc in zip(passes["Game ID"], passes["Description"]):
    r = parse_pass(desc)
    if r:
        r["Game_ID"] = game_id
        rows.append(r)

qb_game = (
    pd.DataFrame(rows)
    .groupby(["Game_ID", "qb"])
    .agg(att=("att", "sum"), cmp=("cmp", "sum"), yds=("yds", "sum"),
         td=("td", "sum"), ints=("ints", "sum"), lng=("yds", "max"))
    .reset_index()
)


# ---------------------------------------------------------------------------
# 5. Match stat lines to the odds panel
# ---------------------------------------------------------------------------
# Odds use full names ("Patrick Mahomes"); play descriptions use "P.Mahomes".
def short_name(full):
    full = re.sub(r"\(.*?\)", "", full).replace(" II", "").strip()   # "(BAL)", "II"
    parts = full.split()
    return parts[0][0].upper() + "." + parts[-1].capitalize() if len(parts) > 1 else full

def normalise(qb):
    first, last = qb.split(".", 1)
    return first + "." + last.capitalize()

panel["qb"] = panel["Player"].map(short_name)
qb_game["qb"] = qb_game["qb"].map(normalise)

matched = panel.merge(qb_game, on=["Game_ID", "qb"], how="left")

stat_for_market = {"yds": "yds", "tds": "td", "completions": "cmp",
                   "attempts": "att", "interceptions": "ints",
                   "longest_completion": "lng"}
matched["realized"] = np.nan
for mk, col in stat_for_market.items():
    matched.loc[matched["mk"] == mk, "realized"] = matched.loc[matched["mk"] == mk, col]
matched["over_hit"] = np.where(matched["realized"].isna(), np.nan,
                               (matched["realized"] > matched["Point"]).astype(float))
print(f"Share of panel matched to a stat line: {matched['realized'].notna().mean():.4f}")

# Calibration: implied over probability against how often the over hit,
# one observation per product so each outcome is counted once.
prod = (matched.dropna(subset=["over_hit"])
        .groupby(["product", "mk", "Game_ID"])
        .agg(implied=("p_over_fair", "mean"), actual=("over_hit", "mean"))
        .reset_index())
cal = prod.groupby("mk").agg(n=("actual", "size"), implied=("implied", "mean"),
                             actual=("actual", "mean"))
cal["z"] = (cal["actual"] - cal["implied"]) / np.sqrt(cal["actual"] * (1 - cal["actual"]) / cal["n"])
print(cal.round(3), "\n")


# ---------------------------------------------------------------------------
# 6. Game-level conditions from the play-by-play, interacted with book
# ---------------------------------------------------------------------------
# Parse the timestamps first: the column has some blanks, and taking the
# minimum of mixed text and missing values fails on some pandas versions.
pbp["wall_clock"] = pd.to_datetime(pbp["Wall Clock"], utc=True, errors="coerce")

game = (pbp.groupby("Game ID")
        .agg(temp=("Temperature", "first"), wind=("Wind Speed", "first"),
             cond=("Weather Condition", "first"), venue=("Venue", "first"),
             game_type=("Game Type", "first"), first_play=("wall_clock", "min"))
        .reset_index().rename(columns={"Game ID": "Game_ID"}))

# Fixed and retractable roofs, coded by hand from the venue name.
ROOFED = ["AT&T", "Allegiant", "Caesars Superdome", "Ford Field", "Lucas Oil",
          "Mercedes-Benz", "NRG", "SoFi", "State Farm", "U.S. Bank"]
game["dome"] = game["venue"].apply(lambda v: any(x in v for x in ROOFED)).astype(int)
game["windy"] = ((game["wind"] >= 15) & (game["dome"] == 0)).astype(int)
game["cold"] = ((game["temp"] <= 35) & (game["dome"] == 0)).astype(int)
game["precip"] = (game["cond"].fillna("")
                  .str.contains("rain|snow|drizzle|storm", case=False).astype(int))
et = game["first_play"].dt.tz_convert("America/New_York")
game["prime"] = ((et.dt.hour >= 19) | et.dt.dayofweek.isin([0, 3])).astype(int)
game["playoff"] = (game["game_type"] == "playoff").astype(int)

# These are constant within a game, so the product fixed effect would absorb
# them. They can only enter interacted with book.
pw = panel.merge(game, on="Game_ID")
for v in ["windy", "cold", "precip", "prime", "playoff", "dome"]:
    m = smf.ols(f"Overround ~ C(Bookmaker)*C(mk) + C(Bookmaker):{v}", pw).fit(
        cov_type="cluster", cov_kwds={"groups": pw["Game_ID"]})
    out = {k.split("[")[1].split("]")[0]: f"{m.params[k]*100:+.3f} (t={m.tvalues[k]:.1f})"
           for k in m.params.index if ":" + v in k}
    print(v, out)
