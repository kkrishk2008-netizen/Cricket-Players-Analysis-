#!/usr/bin/env python3
"""
SupeRR Selector S2 - Round 1: Reliability-under-pressure pipeline  (v2.0)
==========================================================================
Implements PRESSURE.pdf (Pressure Index Framework v1.0) and upgrades it with a
higher-order Markov model of the innings.

What changed vs v1
------------------
1. DATA FIX     `extras1_runs` is an extras *type code* (wide=2, legbyes=3 ...), not a run count.
                Team runs are now  runs + 1 per wide/no-ball/bye/leg-bye + extras_2_runs, and the
                chase target is reconciled with the official result when the ball data disagree.
2. MARKOV MODEL A second-order (outcome-memory), state-dependent Markov chain of every delivery:
                    state  = (format, innings, balls left, wickets, runs needed | score, previous outcome)
                    p(o|s) = hierarchical-Dirichlet-smoothed outcome distribution over
                             {0,1,2,3,4,6,W, wide/nb +1,+2,+5,+7}
                conditioned on over, wickets bucket, required-rate bucket (chase) and previous
                ball class (dot / run / boundary / wicket).  Solved exactly by backward induction:
                    V2[b,w,r,m]  P(chasing side wins)
                    V1[b,w,s,m]  P(side batting first wins)   (terminal = 1 - V2 at target s+1)
                    E [b,w,m]    expected runs still to come (Markov resource table)
                (Bandyopadhyay & Bhattacharjee 2025/26 higher-order Markov PI; Mishra 2026
                 exactly-solvable WP; Tango-style leverage; Misra & Rajaraman 2026 "Delta").
3. PRESSURE v2  SPI is anchored on LEVERAGE: the exact expected |change in win probability| on
                this ball, i.e. how much is at stake.  It is blended with jeopardy (how far the side is
                behind in win probability), resource-adjusted chase difficulty / projection deficit,
                Markov resource loss, dot-ball squeeze and entry status (bat), or phase, boundary
                momentum and set-batter threat (bowl).  Dead-rubber balls are discounted.
4. METRICS v2   - NPV uses a STATE-DEPENDENT wicket cost (Markov runs lost by a dismissal now),
                  not one league-average constant.
                - Level = opposition-adjusted, empirical-Bayes ridge estimate (batter + bowler
                  effects fitted jointly), i.e. shrinkage and opposition adjustment in one step.
                - Impact = WPA (win probability added) per 100 high-pressure balls, ridge-shrunk.
                - Consistency = beta-binomial posterior share of HP innings with NPV >= 0.
                - Breadth = expected number of pressure contexts with above-average level (soft).
                - Optional recency weighting (half-life in seasons).
                - Rank-stability bootstrap (P(top-10), 90% rank interval).
5. VALIDATION   adds Markov-order test, out-of-fold WP calibration, martingale check, and
                old-vs-new split-half stability.

Usage
-----
  python pressure_pipeline.py --bbb "data/ball_by_ball/*" --data-dir data --out out
  python pressure_pipeline.py --bbb data/ball_by_ball/ball_by_ball.csv --data-dir data --out out --fast
"""
import argparse, glob, json, sys, time, warnings, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
CFG = dict(
    # ---- SPI v2 weights (each block sums to 1; components scaled to 0-1)
    w_bat_2nd=dict(L=.35, J=.15, C=.20, R=.10, D=.10, E=.10),
    w_bat_1st=dict(L=.35, J=.15, P=.15, R=.15, D=.10, E=.10),
    w_bowl=dict(L=.35, J=.15, Q=.10, F=.05, B=.15, S=.20),
    garbage_discount=.5,         # SPI x (1 - g + g*live); live = 0 when the game is effectively decided
    band_q=(.50, .80, .95),      # Low / Medium / High / Extreme cut-offs (percentiles within dataset)
    # ---- Markov chain
    markov_alpha=(200., 800., 3000., 800.),  # Dirichlet strengths: wickets, req-rate, memory, format (auto-tuned)
    R_max=320, S_max=330,        # runs-needed / score grid
    rb_edges=(0.75, 1.25, 1.6, 2.0),   # required runs PER BALL buckets (4.5 / 7.5 / 9.6 / 12 rpo)
    # ---- Reliability
    rel_w=dict(level=.40, cons=.25, breadth=.15, impact=.10, conf=.10),
    k_min=30, k_max=5000,        # bounds for empirical-Bayes prior strengths
    k_delta_max=400,             # cap on the pressure-delta prior (EB usually finds ~no clutch signal; the cap keeps
                                 # pressure-specific evidence able to move a player with ~400 HP balls by half)
    pressure_weight_floor=.25,   # ball weight in the ability model = floor + (1-floor) * SPI percentile
    cons_prior=8.,               # fallback beta-binomial strength (normally estimated by EB)
    half_life=3.0,               # recency weighting in seasons (None/0 = off)
    min_hp_balls=60, min_innings=8, min_inn_balls=5, elig_mode="or",
    ctx_min_bat=20, ctx_min_bowl=18,
    exclude_revised=True, exclude_no_result=True,
    n_sens=30, n_boot_rank=60, seed=7,
)

# outcome classes of the Markov chain
OUT_LABEL = ["0", "1", "2", "3", "4", "6", "W", "X1", "X2", "X5", "X7"]
OUT_RUNS = np.array([0, 1, 2, 3, 4, 6, 0, 1, 2, 5, 7])
LEGAL_RUN_K = range(6)           # legal, non-wicket
K_W = 6
X_K = (7, 8, 9, 10)              # wides / no-balls (ball not consumed)
M_NEXT = np.array([1, 0, 0, 0, 2, 2, 3, -1, -1, -1, -1])   # previous-ball class: 0 run/start, 1 dot, 2 boundary, 3 wicket
NK, NM = 11, 4
WB = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])            # wickets fallen 0..9 -> bucket
NWB, NRB = 5, 5


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def _read_one(fobj):
    df = pd.read_csv(fobj, dtype=str, on_bad_lines="skip")
    return df[df["match_id"].astype(str).str.fullmatch(r"\d+", na=False)]   # drops "Sandbox export..." junk rows


def load_bbb(patterns):
    files = []
    for p in patterns:
        files += sorted(glob.glob(p, recursive=True)) or [p]
    parts = []
    for f in files:
        fp = Path(f)
        if not fp.is_file():
            continue
        if fp.suffix.lower() == ".zip":
            with zipfile.ZipFile(fp) as z:
                for n in sorted(z.namelist()):
                    if n.lower().endswith(".csv"):
                        parts.append(_read_one(z.open(n)))
        elif fp.suffix.lower() == ".csv":
            parts.append(_read_one(fp))
    if not parts:
        sys.exit("No ball-by-ball rows found in: %s" % patterns)
    d = pd.concat(parts, ignore_index=True)
    num = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id",
           "bowling_team_id", "striker_id", "non_striker_id", "bowler_id", "runs", "extras1_runs",
           "extras_2_runs", "dismissed_player_id", "batting_order"]
    for c in num:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["is_four", "is_six", "is_wicket"]:
        d[c] = d[c].astype(str).str.lower().eq("true")
    d = d.drop_duplicates(["match_id", "innings", "innings_ball_num"])
    log(f"[load] {len(d):,} deliveries from {d.match_id.nunique()} matches")
    return d


# ----------------------------------------------------------------------------
# 2. CLEAN + MATCH STATE
# ----------------------------------------------------------------------------
def _margin_check(d, team_runs, margins):
    t = team_runs.groupby([d.match_id, d.innings]).sum().unstack()
    t = t.loc[t.index.intersection(margins.index)]
    diff = (t[1] - t[2]) - margins.loc[t.index]
    return dict(exact=float((diff == 0).mean()), within3=float((diff.abs() <= 3).mean()), n=int(len(t)))


def build_state(bbb, data_dir):
    m = pd.read_csv(Path(data_dir) / "matches.csv")
    comp = pd.read_csv(Path(data_dir) / "competitions.csv")
    m = m.merge(comp[["comp_id", "comp_name", "match_format"]], on="comp_id", how="left")
    m["match_id"] = m.match_id.astype(int)
    m["year"] = pd.to_datetime(m.match_date).dt.year
    keep = ["match_id", "comp_id", "comp_name", "match_date", "year", "match_format", "venue_id", "win_team_id",
            "result", "first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]
    d = bbb.merge(m[keep], on="match_id", how="inner")
    n0 = d.match_id.nunique()
    d = d[d.innings.isin([1, 2])]                                   # drop super overs
    if CFG["exclude_revised"]:
        rev = d.first_inn_revised_overs.notna() | d.second_inn_revised_overs.notna() | d.revised_target.notna()
        d = d[~rev]
    if CFG["exclude_no_result"]:
        d = d[~d.result.fillna("").str.contains("No Result|Abandon", case=False)]
    d = d[d.groupby("match_id").innings.transform("nunique") == 2]  # need both innings
    log(f"[clean] matches kept {d.match_id.nunique()} of {n0} (dropped revised / no-result / incomplete)")
    d = d.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    # ---- extras & run accounting ------------------------------------------------
    et = d.extras_type.fillna("").str.lower()
    d["is_wide"] = et.str.contains("wide")
    d["is_nb"] = et.isin(["overstep", "other"]) | et.str.contains(r"no.?ball")
    d["is_bye"] = et.str.fullmatch(r"byes?")
    d["is_lb"] = et.str.contains("leg")
    d["is_pen"] = et.str.contains("penalty")
    d["faced"] = ~d.is_wide & ~d.is_pen
    d["legal"] = ~d.is_wide & ~d.is_nb & ~d.is_pen
    runs = d.runs.fillna(0)
    e2 = d.extras_2_runs.fillna(0).clip(0, 6)
    bat_runs = runs.where(~(d.is_bye | d.is_lb), 0)
    margins = d.drop_duplicates("match_id").set_index("match_id").result.str.extract(
        r"(?i)won by (\d+) runs?", expand=False).astype(float).dropna()
    one = (d.is_wide | d.is_nb | d.is_bye | d.is_lb).astype(int)
    cands = {"runs_only": runs, "runs+1/wide&nb (v1)": runs + (d.is_wide | d.is_nb),
             "runs+1/extra+extras_2 (v2)": runs + one + e2 + 5 * d.is_pen}
    checks = {k: _margin_check(d, v, margins) for k, v in cands.items()}
    log("[clean] run-accounting check vs official 'won by N runs' margins:")
    for k, v in checks.items():
        log(f"         {k:28s} exact={v['exact']:.2f}  |diff|<=3={v['within3']:.2f}  (n={v['n']})")
    d["team_runs"] = cands["runs+1/extra+extras_2 (v2)"]
    d["bat_runs"] = bat_runs
    d["bowl_runs"] = bat_runs + (one * (d.is_wide | d.is_nb)) + e2 * (d.is_wide | d.is_nb)
    d["dot"] = d.legal & (d.team_runs == 0)
    d["bnd_evt"] = d.is_four | d.is_six | d.is_wide | d.is_nb

    dt = d.dismissal_type.fillna("").str.lower()
    d["batter_out"] = d.is_wicket & ((d.dismissed_player_id == d.striker_id) | d.dismissed_player_id.isna()) \
        & ~dt.str.contains("retire")
    d["bowler_wkt"] = d.is_wicket & ~dt.str.contains(r"run.?out|retire|obst|timed")

    # ---- cumulative innings state -------------------------------------------------
    g = d.groupby(["match_id", "innings"], sort=False)
    d["legal_before"] = g.legal.cumsum() - d.legal
    d["score_before"] = g.team_runs.cumsum() - d.team_runs
    d["wkts_before"] = g.is_wicket.cumsum() - d.is_wicket
    d["balls_rem"] = (120 - d.legal_before).clip(lower=1)
    d["over"] = (d.legal_before // 6 + 1).clip(upper=20)
    d["over_idx"] = d.over - 1
    d["w"] = d.wkts_before.clip(0, 9)
    d["inn_total"] = g.team_runs.transform("sum")

    # ---- target, reconciled with the official result -----------------------------
    tot = d.groupby(["match_id", "innings"]).team_runs.sum().unstack()
    bt = d.groupby(["match_id", "innings"]).batting_team_id.first().unstack()
    win = d.groupby("match_id").win_team_id.first()
    target = tot[1] + 1
    chase_won = win == bt[2]
    first_won = win == bt[1]
    fix_a = chase_won & (tot[2] < target)              # chaser won but ball data lost runs
    fix_b = first_won & (tot[2] >= target)             # defended but ball data say target reached
    mg = margins.reindex(target.index)
    target = target.where(~fix_a, tot[2])
    target = target.where(~fix_b, tot[2] + mg.fillna(0) + 1)
    log(f"[clean] target reconciled with result in {int(fix_a.sum() + fix_b.sum())} matches")
    d["target"] = np.where(d.innings == 2, d.match_id.map(target), np.nan)
    d["bat_first_won"] = d.match_id.map(first_won.astype(float).where(first_won | chase_won))
    d["bat_side_won"] = np.where(d.innings == 1, d.bat_first_won, 1 - d.bat_first_won)
    d["runs_needed"] = np.where(d.innings == 2, d.target - d.score_before, np.nan)
    d = d[~((d.innings == 2) & (d.runs_needed <= 0))].reset_index(drop=True)

    # ---- short-term context --------------------------------------------------------
    def lastn(col, win=6):
        c = d[col].astype(int).groupby([d.match_id, d.innings], sort=False).cumsum()
        before = c - d[col].astype(int)
        lag = before.groupby([d.match_id, d.innings], sort=False).shift(win).fillna(0)
        return before - lag
    d["dots_last6"] = lastn("dot")
    d["bnd_last6"] = lastn("bnd_evt")
    grp = (~d["dot"]).groupby([d.match_id, d.innings], sort=False).cumsum()
    d["dot_streak"] = d["dot"].groupby([d.match_id, d.innings, grp], sort=False).cumsum() - d["dot"].astype(int)
    d["wk_last14"] = lastn("is_wicket", 14)
    gb = d.groupby(["match_id", "innings", "striker_id"], sort=False)
    d["bf_before"] = gb.faced.cumsum() - d.faced
    d["bruns_before"] = gb.bat_runs.cumsum() - d.bat_runs
    d["entry_wk"] = gb.wk_last14.transform("first") > 0
    d["rrr"] = d.runs_needed * 6 / d.balls_rem
    d["fmt"] = d.match_format.fillna("T20")
    d["fmt_code"] = d.fmt.astype("category").cat.codes
    d["venue"] = d.venue_id.where(d.venue_id.map(d.venue_id.value_counts()) >= 300, -1)
    d["venue_code"] = d.venue.astype("category").cat.codes
    d["season"] = d.year
    d["spin"] = (d.pace_or_spin == "Spin").astype(int)
    d["lhb"] = (d.batting_type == "LHB").astype(int)

    # ---- Markov outcome class & previous-ball memory ------------------------------
    tr = d.team_runs.values
    cls = np.select([tr <= 0, tr == 1, tr == 2, tr == 3, tr <= 5], [0, 1, 2, 3, 4], 5)
    cls = np.where(d.is_wicket.values, K_W, cls)
    xcls = np.select([tr <= 1, tr <= 3, tr <= 5], [7, 8, 9], 10)
    d["cls"] = np.where(d.legal.values, cls, xcls)
    mc = pd.Series(np.where(d.legal, M_NEXT[d.cls.values], np.nan), index=d.index)
    gg = [d.match_id, d.innings]
    prev = mc.groupby(gg, sort=False).shift(1)
    d["m"] = prev.groupby(gg, sort=False).ffill().fillna(0).astype(int)
    d["m_next"] = np.where(d.legal, M_NEXT[d.cls.values], d.m)
    rb = np.digitize((d.runs_needed / d.balls_rem).fillna(0).values, CFG["rb_edges"])
    d["rb"] = np.where(d.innings == 2, rb, 0)
    d["wb"] = WB[d.w.values]
    log(f"[state] deliveries={len(d):,}  matches={d.match_id.nunique()}  formats={dict(d.groupby('fmt').match_id.nunique())}")
    return d, dict(run_accounting_check=checks, target_reconciled=int(fix_a.sum() + fix_b.sum()))


# ----------------------------------------------------------------------------
# 3. HIGHER-ORDER MARKOV CHAIN  (transition model + exact backward induction)
# ----------------------------------------------------------------------------
def fit_transitions(d, n_fmt, use_m=True, alpha=None):
    """p[f, inn, over, wb, rb, m, outcome] with hierarchical Dirichlet smoothing, one strength per level:
       (inn,over) -a_wb-> +wickets -a_rb-> +required-rate -a_m-> +previous-ball memory -a_fmt-> +format
       A very large strength switches a level off (child = parent)."""
    a_wb, a_rb, a_m, a_f = CFG["markov_alpha"] if alpha is None else alpha
    if not use_m:
        a_m = 1e12
    shape = (n_fmt, 2, 20, NWB, NRB, NM, NK)
    idx = np.ravel_multi_index((d.fmt_code.values, d.innings.values - 1, d.over_idx.values.astype(int),
                                d.wb.values, d.rb.values, d.m.values, d.cls.values), shape)
    N = np.bincount(idx, minlength=int(np.prod(shape))).reshape(shape).astype(float)

    def blend(cnt, prior, a):
        return (cnt + a * prior) / (cnt.sum(-1, keepdims=True) + a)
    N0 = N.sum((0, 3, 4, 5))                                    # (2,20,K)
    p0 = (N0 + 1) / (N0 + 1).sum(-1, keepdims=True)
    p1 = blend(N.sum((0, 4, 5)), p0[:, :, None, :], a_wb)       # (2,20,WB,K)
    p2 = blend(N.sum((0, 5)), p1[:, :, :, None, :], a_rb)       # (2,20,WB,RB,K)
    p3 = blend(N.sum(0), p2[..., None, :], a_m)                 # (2,20,WB,RB,M,K)
    p4 = blend(N, p3[None], a_f)                                # (F,2,20,WB,RB,M,K)
    return p4 / p4.sum(-1, keepdims=True)


def _oof_logloss(d, alpha, use_m=True):
    half = (d.match_id % 2).values
    F = int(d.fmt_code.max()) + 1
    r = []
    for hv in (0, 1):
        tr, te = d[half != hv], d[half == hv]
        p = fit_transitions(tr, F, use_m, alpha)
        pp = p[te.fmt_code.values, te.innings.values - 1, te.over_idx.values.astype(int), te.wb.values,
               te.rb.values, te.m.values, te.cls.values]
        r.append(-np.mean(np.log(np.clip(pp, 1e-12, 1))))
    return float(np.mean(r))


def tune_markov(d):
    """Coordinate search of the four smoothing strengths by out-of-fold (match-parity) log-loss."""
    grid = [10, 50, 200, 800, 3000, 1e5]
    al = list(CFG["markov_alpha"])
    best = _oof_logloss(d, al)
    for _ in range(2):
        for lvl in range(4):
            for g in grid:
                trial = al.copy(); trial[lvl] = g
                v = _oof_logloss(d, trial)
                if v < best - 1e-6:
                    best, al = v, trial
    CFG["markov_alpha"] = tuple(float(x) for x in al)
    return dict(alpha_wb_rb_mem_fmt=CFG["markov_alpha"], oof_logloss=best)


def solve_chain(p, f):
    """Backward induction for one format. Returns dict(V1, V2, E)."""
    R, S = CFG["R_max"], CFG["S_max"]
    runs = OUT_RUNS
    # --- runs expectancy (unconstrained / 1st-innings dynamics)
    E = np.zeros((121, 11, NM))
    for b in range(1, 121):
        o = min((120 - b) // 6, 19)
        P = p[f, 0, o][WB][:, 0]                                 # (10, M, K)
        num = sum(P[..., k] * (runs[k] + E[b - 1, :10, M_NEXT[k]][:, None]) for k in LEGAL_RUN_K)
        num = num + P[..., K_W] * E[b - 1, 1:11, 3][:, None]
        num = num + sum(P[..., x] * runs[x] for x in X_K)
        px = P[..., list(X_K)].sum(-1)
        E[b, :10] = num / (1 - px)
    # --- chase win probability V2[b, w, r, m]
    r = np.arange(R + 2)
    tend = np.where(r <= 0, 1., np.where(r == 1, .5, 0.))       # tie -> 0.5
    V2 = np.zeros((121, 11, R + 2, NM), np.float32)
    V2[0] = tend[None, :, None]
    V2[:, 10] = tend[None, :, None]
    for b in range(1, 121):
        o = min((120 - b) // 6, 19)
        rbi = np.digitize(r / b, CFG["rb_edges"])
        P = p[f, 1, o][WB][:, rbi]                               # (10, R+2, M, K)
        prev = V2[b - 1].astype(float)
        L = np.zeros((10, R + 2, NM))
        for k in LEGAL_RUN_K:
            L += P[..., k] * prev[:10, np.clip(r - runs[k], 0, None), M_NEXT[k]][:, :, None]
        L += P[..., K_W] * prev[1:11, :, 3][:, :, None]
        V = L / P[..., :7].sum(-1)
        for _ in range(4):                                       # extras: same ball count, fixed point
            Vn = L.copy()
            for x in X_K:
                Vn += P[..., x] * V[:, np.clip(r - runs[x], 0, None), :]
            Vn[:, 0, :] = 1.
            V = Vn
        V2[b, :10] = V
    # --- first-innings win probability V1[b, w, s, m]
    s = np.arange(S + 1)
    U = 1. - V2[120, 0, np.clip(s + 1, 0, R + 1), 0].astype(float)
    V1 = np.zeros((121, 11, S + 1, NM), np.float32)
    V1[0] = U[None, :, None]
    V1[:, 10] = U[None, :, None]
    for b in range(1, 121):
        o = min((120 - b) // 6, 19)
        P = p[f, 0, o][WB][:, 0][:, None]                        # (10, 1, M, K)
        prev = V1[b - 1].astype(float)
        L = np.zeros((10, S + 1, NM))
        for k in LEGAL_RUN_K:
            L += P[..., k] * prev[:10, np.clip(s + runs[k], None, S), M_NEXT[k]][:, :, None]
        L += P[..., K_W] * prev[1:11, :, 3][:, :, None]
        V = L / P[..., :7].sum(-1)
        for _ in range(4):
            Vn = L.copy()
            for x in X_K:
                Vn += P[..., x] * V[:, np.clip(s + runs[x], None, S), :]
            V = Vn
        V1[b, :10] = V
    return dict(V1=V1, V2=V2, E=E)


class Markov:
    def __init__(self, d, use_m=True):
        self.F = int(d.fmt_code.max()) + 1
        self.p = fit_transitions(d, self.F, use_m)
        self.ch = [solve_chain(self.p, f) for f in range(self.F)]
        self.E = np.stack([c["E"] for c in self.ch])             # (F,121,11,M)
        self.cal = (0.0, 0.0)                                     # (c, bias): time-varying temperature

    def calibrate(self, raw, inn, b):
        """WP_cal = sigmoid((1 - c*tau) * logit(WP) + bias*tau),  tau = share of the match still to play.
           Exact at the end of the match (tau -> 0); corrects the chain's over-confidence early on, which
           comes from treating both teams as league-average."""
        c, bias = self.cal
        if c == 0 and bias == 0:
            return raw
        tau = (np.where(np.asarray(inn) == 1, 120, 0) + np.clip(np.asarray(b), 0, 120)) / 240.
        p = np.clip(raw, 1e-6, 1 - 1e-6)
        z = (1 - c * tau) * np.log(p / (1 - p)) + bias * tau * np.where(np.asarray(inn) == 1, 1, -1)
        out = 1 / (1 + np.exp(-z))
        return np.where((raw <= 0) | (raw >= 1), raw, out)

    def wp(self, f, inn, b, w, x, m, raw=False):
        """P(batting side wins). x = runs needed (inn 2) or score (inn 1)."""
        R, S = CFG["R_max"], CFG["S_max"]
        f, inn, m = (np.asarray(v).astype(int) for v in (f, inn, m))
        b = np.clip(np.asarray(b).astype(int), 0, 120)
        w = np.clip(np.asarray(w).astype(int), 0, 10)
        x = np.nan_to_num(np.asarray(x, float)).astype(int)
        out = np.empty(len(b))
        for ff in np.unique(f):
            for ii in (1, 2):
                mk = (f == ff) & (inn == ii)
                if not mk.any():
                    continue
                if ii == 2:
                    out[mk] = self.ch[ff]["V2"][b[mk], w[mk], np.clip(x[mk], 0, R + 1), m[mk]]
                else:
                    out[mk] = self.ch[ff]["V1"][b[mk], w[mk], np.clip(x[mk], 0, S), m[mk]]
        return out if raw else self.calibrate(out, inn, b)

    def probs(self, d):
        return self.p[d.fmt_code.values, d.innings.values - 1, d.over_idx.values.astype(int),
                      d.wb.values, d.rb.values, d.m.values]

    def state_x(self, d):
        return np.where(d.innings == 2, d.runs_needed, d.score_before)

    def leverage(self, d, wp_pre):
        """exact E_o |WP(next) - WP(now)| and E_o [WP(next) - WP(now)] (martingale check)"""
        P = self.probs(d)
        sg = np.where(d.innings == 2, -1, 1)
        x = self.state_x(d)
        b, w, f, inn, m = d.balls_rem.values, d.w.values, d.fmt_code.values, d.innings.values, d.m.values
        absw = np.zeros(len(d)); drift = np.zeros(len(d))
        for k in range(NK):
            if k == K_W:
                v = self.wp(f, inn, b - 1, w + 1, x, np.full(len(d), 3))
            elif k in X_K:
                v = self.wp(f, inn, b, w, x + sg * OUT_RUNS[k], m)
            else:
                v = self.wp(f, inn, b - 1, w, x + sg * OUT_RUNS[k], np.full(len(d), M_NEXT[k]))
            absw += P[:, k] * np.abs(v - wp_pre)
            drift += P[:, k] * (v - wp_pre)
        return absw, drift


def markov_order_test(d):
    """Out-of-fold log-loss of the next-ball outcome: marginal-by-over vs 1st-order (state) vs 2nd-order (state+memory)."""
    al = CFG["markov_alpha"]
    res = {"marginal_by_over": _oof_logloss(d, (1e12, 1e12, 1e12, 1e12)),
           "first_order(state, no memory)": _oof_logloss(d, al, use_m=False),
           "second_order(state+memory)": _oof_logloss(d, al)}
    res["memory_gain_nats_per_ball"] = res["first_order(state, no memory)"] - res["second_order(state+memory)"]
    return res


def wp_calibration(d, nbins=10):
    """Out-of-fold (by match parity) win probability vs actual results; fits the time-varying temperature
       (c, bias) on the out-of-fold predictions. Returns (stats, calibration table, (c, bias))."""
    from scipy.optimize import minimize
    half = d.match_id % 2
    pred = np.full(len(d), np.nan)
    for hv in (0, 1):
        mk = (half == hv).values
        mm = Markov(d[~mk])
        te = d[mk]
        pred[mk] = mm.wp(te.fmt_code, te.innings, te.balls_rem, te.w, mm.state_x(te), te.m, raw=True)
    y = d.bat_side_won.values
    ok = ~np.isnan(y) & ~np.isnan(pred)
    y, p = y[ok], np.clip(pred[ok], 1e-6, 1 - 1e-6)
    inn, bb = d.innings.values[ok], d.balls_rem.values[ok]
    tmp = Markov.__new__(Markov)

    def cal_p(theta):
        tmp.cal = tuple(theta)
        return np.clip(tmp.calibrate(p, inn, bb), 1e-6, 1 - 1e-6)
    ll = lambda pp: float(-np.mean(y * np.log(pp) + (1 - y) * np.log(1 - pp)))
    theta = minimize(lambda t: ll(cal_p(t)), x0=[0.3, 0.0], method="Nelder-Mead").x
    pc = cal_p(theta)

    def summary(pp):
        q = pd.cut(pp, np.linspace(0, 1, nbins + 1), include_lowest=True)
        tab = pd.DataFrame({"p": pp, "y": y, "bin": q}).groupby("bin", observed=True).agg(
            mean_pred=("p", "mean"), obs_rate=("y", "mean"), n=("y", "size")).reset_index()
        tab["bin"] = tab["bin"].astype(str)
        st = dict(brier=float(np.mean((pp - y) ** 2)), logloss=ll(pp),
                  ece=float(np.sum(tab.n * np.abs(tab.mean_pred - tab.obs_rate)) / tab.n.sum()))
        return st, tab
    st_raw, tab_raw = summary(p)
    st_cal, tab_cal = summary(pc)
    base = np.full_like(p, y.mean())
    stats = dict(raw_markov=st_raw, calibrated=st_cal, temperature_c=float(theta[0]), bias=float(theta[1]),
                 brier_baseline=float(np.mean((base - y) ** 2)), logloss_baseline=ll(base), n_balls=int(len(y)),
                 note="calibration parameters (2) fitted on the same out-of-fold predictions")
    tab = tab_raw.add_suffix("_raw").rename(columns={"bin_raw": "bin"}).merge(
        tab_cal.add_suffix("_cal").rename(columns={"bin_cal": "bin"}), on="bin", how="outer")
    return stats, tab, tuple(theta)


# ----------------------------------------------------------------------------
# 4. SPI v2 COMPONENTS
# ----------------------------------------------------------------------------
def _pct_rank(x):
    return pd.Series(x).rank(pct=True).values


def components(d, mk):
    f, b, w, m = d.fmt_code.values, d.balls_rem.values, d.w.values, d.m.values
    E = mk.E
    E0 = E[f, 120, 0, 0]
    d["erem"] = E[f, b, w, m]
    d["res_pct"] = np.clip(E[f, b, w, 0] / E0, 0, 1)
    d["R"] = 1 - d.res_pct
    bm1 = np.clip(b - 1, 0, 120)
    d["wkt_cost"] = np.clip(E[f, bm1, w, 0] - E[f, bm1, np.clip(w + 1, 0, 10), 3], 0, None)
    # ---- win probability & leverage (Markov)
    x = mk.state_x(d)
    d["wp_bat"] = mk.wp(f, d.innings, b, w, x, m)
    absw, drift = mk.leverage(d, d.wp_bat.values)
    d["swing_exp"] = absw
    d["LI"] = absw / absw.mean()
    d["mart_drift"] = drift
    d["L"] = _pct_rank(d.LI.values)
    d["live"] = np.clip(4 * d.wp_bat * (1 - d.wp_bat) / .36, 0, 1)
    d["J_bat"] = np.clip(1 - 2 * d.wp_bat, 0, 1)
    d["J_bowl"] = np.clip(2 * d.wp_bat - 1, 0, 1)
    # ---- resource-adjusted chase difficulty (2nd) / projection deficit (1st)
    c2 = (d.innings == 2).values
    ratio = np.where(c2, d.runs_needed / np.maximum(d.erem, 1), np.nan)
    cap = np.nanquantile(ratio[c2], .95)
    d["need_vs_par"] = ratio
    d["C"] = np.where(c2, np.clip((ratio - .8) / max(cap - .8, 1e-6), 0, 1), 0)
    d["proj_final"] = d.score_before + d.erem
    deficit = np.where(~c2, np.maximum(E0 - d.proj_final, 0), 0)
    d["P"] = np.where(~c2, np.clip(deficit / max(np.quantile(deficit[~c2], .95), 1e-6), 0, 1), 0)
    # ---- micro-pressure (framework)
    d["D"] = np.clip(d.dots_last6 / 6 + .15 * (d.dot_streak >= 4), 0, 1)
    bf = d.bf_before
    taper = np.where(bf < 10, 1.0, np.clip((25 - bf) / 15, 0, 1))
    d["E"] = np.where(d.entry_wk, taper, 0.3 * (bf < 10))
    d["Q"] = np.select([d.over <= 6, d.over >= 17], [0.8, 1.0], 0.3)
    d["F"] = (10 - d.wkts_before.clip(0, 10)) / 10
    d["B"] = np.clip(d.bnd_last6 / 3, 0, 1)
    sr = np.where(d.bf_before > 0, d.bruns_before / d.bf_before.clip(lower=1) * 100, 0)
    d["S"] = np.maximum(np.minimum(d.bf_before / 20, 1), np.clip(sr / 150, 0, 1) * (d.bf_before >= 10))
    return d


def _comb(d, w, jcol):
    return sum(w[c] * (d[jcol] if c == "J" else d[c]) for c in w) / sum(w.values())


def _discount(d):
    g = CFG["garbage_discount"]
    return 1 - g + g * d.live


def spi_bat(d, w1=None, w2=None):
    w1 = w1 or CFG["w_bat_1st"]; w2 = w2 or CFG["w_bat_2nd"]
    raw = np.where(d.innings == 2, _comb(d, w2, "J_bat"), _comb(d, w1, "J_bat"))
    return pd.Series(10 * raw * _discount(d), index=d.index)


def spi_bowl(d, w=None):
    w = w or CFG["w_bowl"]
    return pd.Series(10 * _comb(d, w, "J_bowl") * _discount(d), index=d.index)


def bands(spi, q=None):
    th = np.quantile(spi, q or CFG["band_q"])
    b = np.select([spi < th[0], spi < th[1], spi < th[2]], ["Low", "Medium", "High"], "Extreme")
    return pd.Series(b, index=spi.index), th


# ----------------------------------------------------------------------------
# 5. EXPECTED OUTCOME MODEL + STATE-DEPENDENT NPV + WPA
# ----------------------------------------------------------------------------
def expected_models(d, report):
    from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import log_loss
    feats = ["over_idx", "w", "balls_rem", "innings", "rrr", "runs_needed", "dot_streak", "dots_last6",
             "score_before", "fmt_code", "venue_code", "spin", "lhb", "m", "wp_bat"]
    X = d[feats].astype(float)
    allr = pd.Series(True, index=d.index)
    tasks = {"eb_runs": ("bat_runs", d.faced), "eb_wkt": ("batter_out", d.faced),
             "ew_runs": ("bowl_runs", allr), "ew_wkt": ("bowler_wkt", allr)}
    gkf = GroupKFold(n_splits=5)
    for out, (tgt, mask) in tasks.items():
        y = d[tgt].astype(float); oof = np.full(len(d), np.nan)
        idx = np.where(mask.values)[0]
        reg = out.endswith("runs")
        for tr, te in gkf.split(idx, groups=d.match_id.values[idx]):
            mdl = (HR(max_iter=200, learning_rate=.06, max_depth=4, l2_regularization=1.0, random_state=0) if reg else
                   HC(max_iter=200, learning_rate=.06, max_depth=3, l2_regularization=1.0, random_state=0))
            mdl.fit(X.iloc[idx[tr]], y.iloc[idx[tr]] if reg else y.iloc[idx[tr]].astype(int))
            oof[idx[te]] = mdl.predict(X.iloc[idx[te]]) if reg else mdl.predict_proba(X.iloc[idx[te]])[:, 1]
        d[out] = oof
        yy, pp = y.values[idx], oof[idx]
        if reg:
            report[f"model_{out}"] = dict(rmse=float(np.sqrt(np.mean((yy - pp) ** 2))), rmse_baseline=float(yy.std()))
        else:
            report[f"model_{out}"] = dict(logloss=float(log_loss(yy, np.clip(pp, 1e-6, 1 - 1e-6))),
                                          logloss_baseline=float(log_loss(yy, np.full_like(pp, yy.mean()))))
        log(f"[model] {out}: {report[f'model_{out}']}")
    return d


def attribute(d, mk):
    """NPV (runs currency, state-dependent wicket cost) and WPA (win currency) for batter and bowler."""
    f, inn, b, w = d.fmt_code.values, d.innings.values, d.balls_rem.values, d.w.values
    sg = np.where(inn == 2, -1, 1)
    x = mk.state_x(d)
    b1 = b - d.legal.values.astype(int)
    m1 = d.m_next.values
    wp_bat_post = mk.wp(f, inn, b1, w + d.batter_out.values, x + sg * d.bat_runs.values, m1)
    wp_bowl_post = mk.wp(f, inn, b1, w + d.bowler_wkt.values, x + sg * d.bowl_runs.values, m1)
    wp_all_post = mk.wp(f, inn, b1, w + d.is_wicket.values, x + sg * d.team_runs.values, m1)
    d["wpa_bat"] = np.where(d.faced, wp_bat_post - d.wp_bat, 0.)
    d["wpa_bowl"] = -(wp_bowl_post - d.wp_bat)
    d["wpa_team"] = wp_all_post - d.wp_bat
    wc = d.wkt_cost
    d["npv_bat"] = np.where(d.faced, (d.bat_runs - d.eb_runs) - wc * (d.batter_out - d.eb_wkt), 0.)
    d["npv_bowl"] = (d.ew_runs - d.bowl_runs) + wc * (d.bowler_wkt - d.ew_wkt)
    return d


# ----------------------------------------------------------------------------
# 6. PLAYER METRICS -> RELIABILITY
# ----------------------------------------------------------------------------
def _recency(season):
    hl = CFG["half_life"]
    if not hl:
        return np.ones(len(season))
    return 0.5 ** ((season.max() - season) / hl)


def prep_role(d, role, spi, band):
    """All deliveries for a role (faced balls for batters, all deliveries for bowlers) with pressure flags
       and model weights  wt = recency x pressure weight  (pressure weight = floor + (1-floor) x SPI percentile)."""
    if role == "bat":
        h = d[d.faced]
        out = pd.DataFrame({"pid": h.striker_id, "opp": h.bowler_id, "pname": h.striker_name,
                            "y": h.npv_bat, "wpa": h.wpa_bat, "e": h.eb_runs, "p": h.eb_wkt,
                            "runs": h.bat_runs, "o": h.batter_out.astype(float)})
    else:
        h = d
        out = pd.DataFrame({"pid": h.bowler_id, "opp": h.striker_id, "pname": h.bowler_name,
                            "y": h.npv_bowl, "wpa": h.wpa_bowl, "e": h.ew_runs, "p": h.ew_wkt,
                            "runs": h.bowl_runs, "o": h.bowler_wkt.astype(float)})
    for c in ["match_id", "innings", "season", "fmt", "over", "entry_wk", "bf_before", "pace_or_spin", "S",
              "LI", "wkt_cost", "legal"]:
        out[c] = h[c].values
    out["bnd"] = (h.is_four | h.is_six).astype(float).values
    out["nd"] = (out.runs > 0).astype(float)
    out["band"] = band.loc[h.index].values
    out["spi"] = spi.loc[h.index].values
    out["hp"] = out.band.isin(["High", "Extreme"]).values
    pw = CFG["pressure_weight_floor"]
    out["wt"] = _recency(d.season).loc[h.index].values * (pw + (1 - pw) * out.spi.rank(pct=True).values)
    return out.reset_index(drop=True)


def eb_k(y, g, wt, nmin=30):
    """Empirical-Bayes prior strength k = noise variance / between-player variance (per ball),
       method of moments on weighted player means. Returns (k, noise var, between var)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(g), "w": np.asarray(wt, float)})
    mu = np.average(df.y, weights=df.w)
    s2 = float(np.average((df.y - mu) ** 2, weights=df.w))
    a = df.assign(wy=df.w * df.y, w2=df.w ** 2).groupby("g").agg(sw=("w", "sum"), swy=("wy", "sum"),
                                                                  sw2=("w2", "sum"), n=("y", "size"))
    a = a[a.n >= nmin]
    if len(a) < 10:
        return float(CFG["k_max"]), s2, 0.0
    mean_g = a.swy / a.sw
    neff = a.sw ** 2 / a.sw2
    tau2 = float(mean_g.var() - (s2 / neff).mean())
    k = s2 / tau2 if tau2 > 0 else CFG["k_max"]
    return float(np.clip(k, CFG["k_min"], CFG["k_max"])), s2, max(tau2, 0.0)


def fit_level(h, ycol, k_main, k_delta, k_opp=None):
    """Hierarchical ridge (RAPM-style):
         y_ball = mu + HP_shift*hp + a_player + d_player*hp + o_opponent + noise
       a = all-situation ability (prior strength k_main), d = extra under pressure ("clutch delta",
       prior strength k_delta), o = opposition quality. Pressure level = a + d (per ball, vs league)."""
    import scipy.sparse as sp
    k_opp = k_opp or k_main
    y = h[ycol].values.astype(float); wt = h.wt.values; hp = h.hp.values
    pi, pidx = np.unique(h.pid.values, return_inverse=True)
    oi, oidx = np.unique(h.opp.values, return_inverse=True)
    n, npl, no = len(h), len(pi), len(oi)
    rows = np.arange(n)
    r = np.r_[rows, rows[hp], rows, rows[hp]]
    c = np.r_[pidx, npl + pidx[hp], 2 * npl + oidx, np.full(hp.sum(), 2 * npl + no)]
    X = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, 2 * npl + no + 1))
    mu = np.average(y, weights=wt)
    XtW = X.T.multiply(wt).tocsr()
    A = (XtW @ X).toarray()
    A[np.diag_indices_from(A)] += np.r_[np.full(npl, k_main), np.full(npl, k_delta), np.full(no, k_opp), 1e-6]
    beta = np.linalg.solve(A, XtW @ (y - mu))
    resid = y - mu - X @ beta
    s2 = float(np.average(resid ** 2, weights=wt))
    w_all = np.bincount(pidx, weights=wt, minlength=npl)
    w_hp = np.bincount(pidx[hp], weights=wt[hp], minlength=npl)
    main, delta = beta[:npl], beta[npl:2 * npl]
    sd_main = np.sqrt(s2 / (w_all + k_main))
    df = pd.DataFrame(dict(main=main, delta=delta, hp_level=main + delta, sd_main=sd_main,
                           sd=np.sqrt(sd_main ** 2 + s2 / (w_hp + k_delta)), w_all=w_all, w_hp=w_hp), index=pi)
    part = y - mu - beta[2 * npl:2 * npl + no][oidx] - main[pidx] - beta[-1] * hp   # opp/ability-adjusted residual
    return df, part, s2


def estimate_ks(h, fixed_delta=None):
    """EB prior strengths for ability (all balls) and pressure delta (HP residuals), for NPV and WPA."""
    out = {}
    for ycol, tag in (("y", ""), ("wpa", "w")):
        k_main, s2, tau2 = eb_k(h[ycol], h.pid, h.wt)
        _, part, _ = fit_level(h, ycol, k_main, 1e9)
        hp = h.hp.values
        rr = part[hp] - np.average(part[hp], weights=h.wt.values[hp])
        k_d_eb, _, tau2_d = eb_k(rr, h.pid.values[hp], h.wt.values[hp])
        k_delta = fixed_delta if (fixed_delta and tag == "") else float(np.clip(k_d_eb, CFG["k_min"], CFG["k_delta_max"]))
        out.update({k: float(v) for k, v in {f"k{tag}_main": k_main, f"k{tag}_delta": k_delta, f"k{tag}_delta_eb": k_d_eb,
                    f"tau{tag}_main_per100": 100 * np.sqrt(tau2), f"tau{tag}_delta_per100": 100 * np.sqrt(tau2_d)}.items()})
    return out


def _ctx_masks(h, role):
    if role == "bat":
        return {"chasing": h.innings == 2, "setting": h.innings == 1,
                "collapse_entry": h.entry_wk & (h.bf_before <= 25), "death": h.over >= 17,
                "vs_spin": h.pace_or_spin == "Spin", "vs_pace": h.pace_or_spin == "Pace"}
    return {"defending": h.innings == 2, "bowling_first": h.innings == 1, "powerplay": h.over <= 6,
            "middle": h.over.between(7, 16), "death": h.over >= 17, "vs_set_batter": h.S >= .75}


def _sz(x, post_var):
    """Evidence-weighted z: centre, then divide by the estimated TRUE between-player SD
       (Var(true) = Var(posterior means) + E[posterior var]). Heavily-shrunk (low-signal) terms get a
       small spread and therefore little influence, instead of being re-inflated to unit variance."""
    x = pd.Series(x).astype(float)
    tau2 = x.var(ddof=0) + float(np.nanmean(post_var))
    return (x - x.mean()) / np.sqrt(tau2) if tau2 > 0 else x * 0


def _z(x):
    x = pd.Series(x).astype(float)
    return (x - x.mean()) / (x.std(ddof=0) or 1)


def score_players(h, role, ks, rel_w=None, light=False):
    """h = all role rows from prep_role. Returns (eligible ranked table, all-HP-players table, innings table, meta)."""
    from scipy.stats import norm
    rel_w = rel_w or CFG["rel_w"]
    hp = h[h.hp]
    lv, part, s2 = fit_level(h, "y", ks["k_main"], ks["k_delta"])
    im, _, _ = fit_level(h, "wpa", ks["kw_main"], ks["kw_delta"])
    g = hp.groupby("pid")
    a = pd.DataFrame({"n": g.size(), "name": g.pname.first()})
    lv, im = lv.reindex(a.index), im.reindex(a.index)
    # ---- level: NPV per 100 HP balls, opposition-adjusted, hierarchical EB
    a["level"] = 100 * lv.hp_level; a["level_sd"] = 100 * lv.sd
    a["level_base"] = 100 * lv.main; a["clutch_delta"] = 100 * lv.delta
    a["level_lo90"], a["level_hi90"] = a.level - 1.645 * a.level_sd, a.level + 1.645 * a.level_sd
    a["p_above_avg"] = norm.cdf(a.level / a.level_sd)
    # ---- impact: WPA per 100 HP balls in win-% points
    a["impact"] = 1e4 * im.hp_level; a["impact_sd"] = 1e4 * im.sd
    a["neff_hp"] = lv.w_hp
    a["conf"] = lv.w_hp / (lv.w_hp + ks["k_delta"])
    # ---- innings / spells and consistency (beta-binomial, EB prior strength)
    ig = hp.groupby(["pid", "match_id", "innings"])
    inn = pd.DataFrame({"npv": ig.y.sum(), "n": ig.size(), "season": ig.season.first(), "wpa": ig.wpa.sum()})
    inn = inn[inn.n >= CFG["min_inn_balls"]]
    pi = inn.groupby(level=0)
    a["n_inn"] = pi.size().reindex(a.index).fillna(0)
    hits = pi.npv.apply(lambda s: (s >= 0).sum()).reindex(a.index).fillna(0)
    p0 = float(hits.sum() / max(a.n_inn.sum(), 1))
    sel = a.n_inn >= CFG["min_innings"]
    if sel.sum() >= 10:
        rate = hits[sel] / a.n_inn[sel]
        exp_var = float((p0 * (1 - p0) / a.n_inn[sel]).mean())
        rho = (float(rate.var()) - exp_var) / (p0 * (1 - p0))
        strength = float(np.clip((1 - rho) / rho, 4, 400)) if rho > 0 else 400.
    else:
        strength = CFG["cons_prior"]
    al, be = hits + strength * p0, a.n_inn - hits + strength * (1 - p0)
    a["cons_raw"] = hits / a.n_inn.replace(0, np.nan)
    a["consistency"] = al / (al + be)
    a["consistency_var"] = al * be / ((al + be) ** 2 * (al + be + 1))
    a["eligible"] = ((a.n >= CFG["min_hp_balls"]) & (a.n_inn >= CFG["min_innings"])) if CFG["elig_mode"] == "and" \
        else ((a.n >= CFG["min_hp_balls"]) | (a.n_inn >= CFG["min_innings"]))
    # ---- breadth: expected number of pressure contexts where the player is above average
    cmin = CFG["ctx_min_bat"] if role == "bat" else CFG["ctx_min_bowl"]
    hp_part = part[h.hp.values]
    sum_p = pd.Series(0., index=a.index); elig = pd.Series(0., index=a.index); ctx_lv = {}
    for cname, msk in _ctx_masks(hp, role).items():
        msk = msk.values
        s = hp[msk]
        if not len(s):
            continue
        rr = hp_part[msk] - np.average(hp_part[msk], weights=s.wt)
        c = pd.DataFrame({"pid": s.pid.values, "w": s.wt.values, "wr": s.wt.values * rr}).groupby("pid").agg(
            n=("w", "size"), sw=("w", "sum"), swr=("wr", "sum"))
        c = c.join(lv[["main", "sd_main"]], how="left")
        c["lvl"] = c.main + c.swr / (c.sw + ks["k_delta"])
        c["sd"] = np.sqrt(c.sd_main ** 2 + s2 / (c.sw + ks["k_delta"]))
        ok = c[c.n >= cmin]
        ctx_lv[cname] = (100 * c.lvl).where(c.n >= cmin)
        elig = elig.add(pd.Series(1., index=ok.index), fill_value=0)
        sum_p = sum_p.add(pd.Series(norm.cdf(ok.lvl / ok.sd), index=ok.index), fill_value=0)
    a["ctx_eligible"] = elig.reindex(a.index).fillna(0)
    a["ctx_expected_hits"] = sum_p.reindex(a.index).fillna(0)
    a["breadth"] = (a.ctx_expected_hits + 1) / (a.ctx_eligible + 2)
    if not light:
        for cname, s in ctx_lv.items():
            a[f"level_{cname}"] = s
        a["median_npv_inn"] = pi.npv.median()
        a["inn_npv_p25"] = pi.npv.quantile(.25)
        a["best_inn_share"] = pi.npv.apply(lambda s: s.max() / s[s > 0].sum() if (s > 0).any() else np.nan)
        sums = g[["y", "e", "p", "runs", "o", "bnd", "nd", "wpa", "LI"]].sum()
        avg_lost = hp.runs.sum() / max(hp.o.sum(), 1)
        sgn = 1 if role == "bat" else -1
        a["npv100_raw_v1"] = 100 * (sgn * (sums.runs - sums.e) - sgn * avg_lost * (sums.o - sums.p)) / a.n
        a["npv100_raw_v2"] = 100 * sums.y / a.n
        a["wpa100_raw"] = 1e4 * sums.wpa / a.n
        a["rae100"] = sgn * 100 * (sums.runs - sums.e) / a.n
        a["wk_delta100"] = 100 * (sums.o - sums.p) / a.n
        a["runs_per100"] = 100 * sums.runs / a.n
        a["bnd_pct"] = 100 * sums.bnd / a.n
        a["control_rate"] = 100 * (sums.nd + sums.bnd) / a.n if role == "bat" else np.nan
        a["avg_LI"] = sums.LI / a.n
        a["avg_spi"] = g.spi.mean()
        a["seasons"] = g.season.nunique()
    e = a[a.eligible].copy()
    e["reliability"] = (rel_w["level"] * _sz(e.level, e.level_sd ** 2) + rel_w["cons"] * _sz(e.consistency, e.consistency_var)
                        + rel_w["breadth"] * _z(e.breadth) + rel_w["impact"] * _sz(e.impact, e.impact_sd ** 2)
                        + rel_w["conf"] * _z(e.conf))
    e = e.sort_values("reliability", ascending=False)
    e["rank"] = np.arange(1, len(e) + 1)
    meta = dict(consistency_prior_strength=strength, consistency_league_rate=p0,
                level_spread_ratio=float(e.level.std() / np.sqrt(e.level.var() + (e.level_sd ** 2).mean())) if len(e) else None)
    return e, a, inn, meta


def evaluate(d, spi, band, role, ks=None, rel_w=None, ret_parts=False, light=False, fixed_delta=None):
    h = prep_role(d, role, spi, band)
    if ks is None:
        ks = estimate_ks(h, fixed_delta)
    e, a, inn, m2 = score_players(h, role, ks, rel_w, light)
    if not ret_parts:
        return e
    hp = h[h.hp]
    meta = dict(ks=ks, **m2, n_balls=int(len(h)), n_hp_balls=int(len(hp)), n_players=int(len(a)), n_eligible=int(len(e)),
                hp_share_innings2=float((hp.innings == 2).mean()),
                hp_shift_npv100=float(100 * (np.average(hp.y, weights=hp.wt) - np.average(h.y, weights=h.wt))))
    return e, a, inn, h, meta


def rank_stability(h, role, ks, top_ids, B):
    """Match-level bootstrap of the whole reliability score. -> P(top 10), 90% rank interval."""
    rng = np.random.default_rng(CFG["seed"])
    groups = h.groupby("match_id").indices
    mids = np.array(list(groups.keys()))
    ranks = {p: [] for p in top_ids}
    for _ in range(B):
        pick = rng.choice(mids, len(mids), replace=True)
        idx = np.concatenate([groups[m] for m in pick])
        newm = np.repeat(np.arange(len(pick)), [len(groups[m]) for m in pick])
        hb = h.iloc[idx].reset_index(drop=True).assign(match_id=newm)
        e = score_players(hb, role, ks, light=True)[0]
        for p in top_ids:
            ranks[p].append(e["rank"].get(p, len(e) + 1))
    return pd.DataFrame({p: dict(p_top10=np.mean(np.array(r) <= 10), rank_p05=np.percentile(r, 5),
                                 rank_p50=np.median(r), rank_p95=np.percentile(r, 95)) for p, r in ranks.items()}).T


# ----------------------------------------------------------------------------
# 7. VALIDATION (framework section 8 + Markov checks)
# ----------------------------------------------------------------------------
def validate(d, spi_b, spi_w, band_b, band_w, res_bat, res_bowl, out, report, metas):
    v = {}
    g = [d.match_id, d.innings]
    prev_w = d.is_wicket.groupby(g, sort=False).shift(1).fillna(False).astype(bool)
    v["dir_after_wicket"] = dict(spi_bat_after=float(spi_b[prev_w].mean()), spi_bat_other=float(spi_b[~prev_w].mean()),
                                 LI_after=float(d.LI[prev_w].mean()), LI_other=float(d.LI[~prev_w].mean()),
                                 note="bowler SPI is expected to FALL after a wicket (margin grows)",
                                 spi_bowl_after=float(spi_w[prev_w].mean()), spi_bowl_other=float(spi_w[~prev_w].mean()))
    prev = d.groupby(["match_id", "innings", "over"]).team_runs.sum().rename("pr").reset_index()
    prev["over"] += 1
    t = d[(d.innings == 2) & (d.legal_before % 6 == 0) & (d.legal_before > 0)][["match_id", "innings", "over", "rrr"]]
    t = t.assign(spi=spi_b.loc[t.index].values).drop_duplicates(["match_id", "over"])
    t = t.merge(prev, on=["match_id", "innings", "over"], how="inner")
    t["below"] = t.pr < t.rrr
    v["dir_below_rrr_over"] = dict(spi_after_below=float(t[t.below].spi.mean()),
                                   spi_after_above=float(t[~t.below].spi.mean()), n=int(len(t)))
    v["spi_vs_leverage_spearman"] = dict(bat=float(pd.Series(spi_b).corr(d.LI, method="spearman")),
                                         bowl=float(pd.Series(spi_w).corr(d.LI, method="spearman")))
    hpb = band_b.isin(["High", "Extreme"])
    v["hp_composition_bat"] = dict(share_innings2=float((d.innings[hpb] == 2).mean()),
                                   share_death=float((d.over[hpb] >= 17).mean()),
                                   share_powerplay=float((d.over[hpb] <= 6).mean()))
    # face validity dump
    cols = ["match_id", "innings", "over", "legal_before", "score_before", "wkts_before", "target", "runs_needed", "rrr",
            "striker_name", "bowler_name", "wp_bat", "LI", "L", "J_bat", "J_bowl", "C", "P", "R", "D", "E", "S", "B"]
    x = d.assign(spi_bat=spi_b, spi_bowl=spi_w)
    pd.concat([x.nlargest(40, "spi_bat")[cols + ["spi_bat", "spi_bowl"]].assign(kind="top_spi_bat"),
               x.nlargest(40, "spi_bowl")[cols + ["spi_bat", "spi_bowl"]].assign(kind="top_spi_bowl"),
               x.nlargest(40, "LI")[cols + ["spi_bat", "spi_bowl"]].assign(kind="top_leverage"),
               x.sample(40, random_state=1)[cols + ["spi_bat", "spi_bowl"]].assign(kind="random")]
              ).to_csv(out / "face_validity_sample.csv", index=False)
    # split-half stability: new level vs v1-style raw NPV vs raw WPA
    half = d.match_id % 2
    for role, spi, band in (("bat", spi_b, band_b), ("bowl", spi_w, band_w)):
        r = []
        for hv in (0, 1):
            mk = (half == hv).values
            r.append(evaluate(d[mk], spi[mk], band[mk], role, ks=metas[role]["ks"], ret_parts=True)[1])
        j = r[0].join(r[1], lsuffix="0", rsuffix="1", how="inner")
        j = j[(j.n0 >= 20) & (j.n1 >= 20)]
        sh = {"n_players": int(len(j))}
        for col in ["level", "level_base", "clutch_delta", "impact", "consistency", "breadth",
                    "npv100_raw_v2", "npv100_raw_v1", "wpa100_raw"]:
            if len(j) > 2:
                sh[col] = dict(pearson=float(j[col + "0"].corr(j[col + "1"])),
                               spearman=float(j[col + "0"].corr(j[col + "1"], method="spearman")))
        v[f"split_half_{role}"] = sh
    # weight sensitivity (+-20%) and shrinkage sensitivity
    rng = np.random.default_rng(CFG["seed"])
    for role, res in (("bat", res_bat), ("bowl", res_bowl)):
        top15 = set(res.index[:15]); ov = []
        jit = lambda w: {c: x * rng.uniform(.8, 1.2) for c, x in w.items()}
        for _ in range(CFG["n_sens"]):
            s = spi_bat(d, jit(CFG["w_bat_1st"]), jit(CFG["w_bat_2nd"])) if role == "bat" else spi_bowl(d, jit(CFG["w_bowl"]))
            b, _ = bands(s)
            r = evaluate(d, s, b, role, ks=metas[role]["ks"], rel_w=jit(CFG["rel_w"]), light=True)
            ov.append(len(top15 & set(r.index[:15])))
        v[f"weight_sens_{role}_top15_overlap"] = dict(mean=float(np.mean(ov)), min=int(np.min(ov)), max=int(np.max(ov)))
        spi, band = (spi_b, band_b) if role == "bat" else (spi_w, band_w)
        out_k = {}
        base_ks = metas[role]["ks"]
        for k in (30, 60, 100, 300, 1000):            # pressure-delta prior strength (framework: 30 / 60 / 100)
            r = evaluate(d, spi, band, role, ks={**base_ks, "k_delta": k}, light=True)
            out_k[f"k_delta={k}"] = len(top15 & set(r.index[:15]))
        for mult in (.5, 2.):                         # ability prior strength
            r = evaluate(d, spi, band, role, ks={**base_ks, "k_main": base_ks["k_main"] * mult}, light=True)
            out_k[f"k_main x{mult}"] = len(top15 & set(r.index[:15]))
        v[f"shrink_sens_{role}_top15_overlap"] = out_k
    report["validation"] = v
    log("[validate]", json.dumps(v, indent=1, default=float))


def baseline_from_hp(res, hp, role):
    g = hp.groupby("pid")
    s = pd.DataFrame({"n": g.size(), "runs": g.runs.sum(), "outs": g.o.sum(), "legal": g.legal.sum()})
    r = res.copy()
    if role == "bat":
        r["hp_strike_rate"] = 100 * s.runs / s.n
        r["hp_average"] = s.runs / s.outs.replace(0, np.nan)
        r["rank_hp_sr"] = r.hp_strike_rate.rank(ascending=False)
        r["rank_hp_avg"] = r.hp_average.fillna(r.hp_average.max()).rank(ascending=False)
    else:
        r["hp_economy"] = 6 * s.runs / s.legal.replace(0, np.nan)
        r["hp_bowl_sr"] = s.legal / s.outs.replace(0, np.nan)
        r["rank_hp_econ"] = r.hp_economy.rank()
        r["rank_hp_bowl_sr"] = r.hp_bowl_sr.fillna(9999).rank()
    r["rank_raw_npv_v1"] = r.npv100_raw_v1.rank(ascending=False)
    r["rank_level_v2"] = r.level.rank(ascending=False)
    r["rank_impact_wpa"] = r.impact.rank(ascending=False)
    return r


# ----------------------------------------------------------------------------
# 8. SQUAD-FIT TEMPLATE, FIGURES
# ----------------------------------------------------------------------------
def annotate(res, inn, hp, role):
    cols = [c for c in res.columns if c.startswith("level_") and c not in ("level_sd", "level_lo90", "level_hi90", "level_base")]
    base = ["name", "rank", "reliability", "level", "level_lo90", "level_hi90", "level_base", "clutch_delta", "impact",
            "n", "n_inn", "consistency", "breadth", "p_top10", "rank_p05", "rank_p95"]
    top = res.head(15)[[c for c in base if c in res.columns] + cols].copy()
    fm = hp.groupby(["pid", "fmt"]).apply(lambda s: np.nan if len(s) < 15 else 100 * s.y.sum() / len(s)).unstack()
    top = top.join(fm.add_prefix("npv100_"), how="left")
    sl = {}
    for pid in top.index:
        if pid not in inn.index.get_level_values(0):
            sl[pid] = np.nan; continue
        s = inn.loc[pid].groupby("season").agg(n=("n", "sum"), v=("npv", "sum"))
        s = s[s.n >= 15]
        sl[pid] = np.polyfit(s.index, 100 * s.v / s.n, 1)[0] if len(s) >= 2 else np.nan
    top["trend_npv100_per_season"] = pd.Series(sl)
    for c in ["age", "nationality_overseas", "capped", "price_tier", "role_gap_RR", "phase_fit_RR",
              "why_this_player", "why_RR", "what_could_go_wrong"]:
        top[c] = ""
    return top


def figures(d, spi_b, th_b, bat_all, bowl_all, calib, out, sample_match=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception:
        log("[fig] matplotlib missing, skipped"); return
    C1, C2, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1d", "#6b6a64", "#e1e0d9"
    plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "font.size": 9})
    # 1 SPI distribution
    fig, ax = plt.subplots(figsize=(7, 3.5)); ax.hist(spi_b, bins=60, color=C1)
    for t, n in zip(th_b, ["p50", "p80", "p95"]):
        ax.axvline(t, color=INK, ls="--", lw=.8); ax.text(t, ax.get_ylim()[1] * .92, " " + n, color=MUTED)
    ax.set_xlabel("Batter SPI v2 (0-10)"); ax.set_ylabel("deliveries"); ax.set_title("Batter SPI distribution and pressure bands")
    fig.tight_layout(); fig.savefig(out / "fig_spi_distribution.png", dpi=150); plt.close(fig)
    # 2 sample match: win-probability worm + pressure (two panels, one axis each)
    if sample_match is None:
        c = d[d.innings == 2].groupby("match_id").agg(li=("LI", "mean"), n=("LI", "size"))
        sample_match = c[c.n >= 100].li.idxmax()
    s = d[d.match_id == sample_match].copy()
    s["x"] = np.where(s.innings == 2, 120 + s.legal_before, s.legal_before)
    s["wp_first"] = np.where(s.innings == 1, s.wp_bat, 1 - s.wp_bat)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True)
    a1.plot(s.x, s.wp_first, color=C1, lw=2); a1.set_ylim(0, 1); a1.axhline(.5, color=MUTED, lw=.6)
    a1.set_ylabel("P(team batting first wins)")
    for i, col in ((1, C1), (2, C2)):
        x = s[s.innings == i]
        a2.plot(x.x, spi_b.loc[x.index].rolling(6, min_periods=1).mean(), color=col, lw=2, label=f"innings {i}")
        for xx in x[x.is_wicket].x:
            a1.axvline(xx, color=MUTED, alpha=.25, lw=.8); a2.axvline(xx, color=MUTED, alpha=.25, lw=.8)
    a2.set_ylabel("Batter SPI (6-ball mean)"); a2.set_xlabel("legal balls (innings 2 starts at 120)"); a2.legend(frameon=False)
    a1.set_title(f"Match {sample_match}: Markov win probability and pressure (grey lines = wickets)")
    fig.tight_layout(); fig.savefig(out / "fig_pressure_curve_sample_match.png", dpi=150); plt.close(fig)
    # 3 shrinkage
    for role, allp in (("batters", bat_all), ("bowlers", bowl_all)):
        allp = allp[allp.n >= 15]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.scatter(allp.n, allp.npv100_raw_v2, s=12, alpha=.45, color=C2, label="raw NPV per 100", edgecolor="white", lw=.3)
        ax.scatter(allp.n, allp.level, s=12, alpha=.8, color=C1, label="level (ridge, opp-adjusted)", edgecolor="white", lw=.3)
        ax.axhline(0, color=INK, lw=.6); ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
        ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500, 1000]))
        ax.xaxis.set_major_formatter(ScalarFormatter()); ax.xaxis.set_minor_locator(NullLocator())
        lo, hi = np.nanpercentile(allp.npv100_raw_v2, [1, 99]); ax.set_ylim(min(lo, allp.level.min()) * 1.1, max(hi, allp.level.max()) * 1.1)
        ax.set_xlabel("high-pressure balls (log scale, players with 15+)"); ax.set_ylabel("NPV per 100 HP balls (runs)"); ax.legend(frameon=False)
        ax.set_title(f"{role}: hierarchical shrinkage of pressure NPV")
        fig.tight_layout(); fig.savefig(out / f"fig_shrinkage_{role}.png", dpi=150); plt.close(fig)
    # 4 WP calibration
    if calib is not None:
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect")
        ax.plot(calib.mean_pred_raw, calib.obs_rate_raw, color=C2, lw=2, marker="o", ms=6, label="raw Markov WP")
        ax.plot(calib.mean_pred_cal, calib.obs_rate_cal, color=C1, lw=2, marker="o", ms=6, label="calibrated WP")
        ax.set_xlabel("predicted win probability"); ax.set_ylabel("observed win rate"); ax.legend(frameon=False)
        ax.set_title("Win-probability calibration (out-of-fold)"); fig.tight_layout(); fig.savefig(out / "fig_wp_calibration.png", dpi=150); plt.close(fig)
    # 5 leverage heatmap (chases)
    gl = d[d.innings == 2].groupby(["wkts_before", "over"]).LI
    hm = gl.mean().where(gl.size() >= 30).unstack().reindex(index=range(10), columns=range(1, 21))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    im = ax.imshow(hm.values, aspect="auto", origin="lower", cmap="Blues", extent=[.5, 20.5, -.5, 9.5])
    ax.grid(False); ax.set_xlabel("over (cells with <30 balls blank)"); ax.set_ylabel("wickets down")
    ax.set_xticks(range(1, 21)); ax.set_yticks(range(10)); fig.colorbar(im, ax=ax, label="mean leverage index")
    ax.set_title("Where the stakes are: mean leverage in run chases")
    fig.tight_layout(); fig.savefig(out / "fig_leverage_heatmap.png", dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbb", nargs="+", required=True, help="ball-by-ball CSV/ZIP files or globs")
    ap.add_argument("--data-dir", required=True, help="folder with matches.csv, competitions.csv, ...")
    ap.add_argument("--out", default="out")
    ap.add_argument("--sample-match", type=int, default=None)
    ap.add_argument("--k-bat", type=float, help="fix batter pressure-delta prior strength (default: empirical Bayes)")
    ap.add_argument("--k-bowl", type=float, help="fix bowler pressure-delta prior strength (default: empirical Bayes)")
    ap.add_argument("--half-life", type=float, help="recency half-life in seasons (0 = off)")
    ap.add_argument("--min-hp-balls", type=int); ap.add_argument("--min-innings", type=int)
    ap.add_argument("--strict", action="store_true", help="require BOTH min balls AND min innings")
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--n-boot", type=int, help="rank-stability bootstrap replicates")
    ap.add_argument("--fast", action="store_true", help="fewer sensitivity / bootstrap runs")
    a = ap.parse_args()
    if a.half_life is not None: CFG["half_life"] = a.half_life
    if a.min_hp_balls: CFG["min_hp_balls"] = a.min_hp_balls
    if a.min_innings: CFG["min_innings"] = a.min_innings
    if a.strict: CFG["elig_mode"] = "and"
    if a.n_boot is not None: CFG["n_boot_rank"] = a.n_boot
    if a.fast: CFG.update(n_sens=6, n_boot_rank=min(CFG["n_boot_rank"], 15))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {"version": "2.0", "config": {k: v for k, v in CFG.items()}}

    d, info = build_state(load_bbb(a.bbb), a.data_dir); report["data"] = info

    # ---- Markov chain
    log("[markov] tuning smoothing by out-of-fold log-loss ...")
    report["markov_tuning"] = tune_markov(d)
    log(f"[markov] {report['markov_tuning']}")
    log("[markov] fitting higher-order transition model and solving backward induction ...")
    mk = Markov(d)
    log("[markov] out-of-fold win-probability calibration ...")
    st, calib, mk.cal = wp_calibration(d)
    report["wp_calibration_oof"] = st
    calib.to_csv(out / "wp_calibration.csv", index=False)
    log(f"[markov] calibration: raw={st['raw_markov']}  calibrated={st['calibrated']}  c={st['temperature_c']:.3f} bias={st['bias']:.3f}")
    d = components(d, mk)
    report["markov"] = dict(
        expected_first_innings_total={str(f): float(mk.E[i, 120, 0, 0]) for f, i in
                                      d.groupby("fmt").fmt_code.first().items()},
        martingale_mean_abs_drift_after_calibration=float(np.abs(d.mart_drift).mean()),
        mean_abs_wp_swing_per_ball=float(d.swing_exp.mean()),
        order_test_logloss=markov_order_test(d))
    log(f"[markov] {json.dumps(report['markov'], default=float)}")
    np.savez_compressed(out / "markov_chain.npz", p=mk.p, E=mk.E, outcome_labels=np.array(OUT_LABEL),
                        formats=np.array(d.groupby("fmt_code").fmt.first().values))

    spi_b, spi_w = spi_bat(d), spi_bowl(d)
    band_b, th_b = bands(spi_b); band_w, th_w = bands(spi_w)
    report["spi_thresholds_bat"] = list(map(float, th_b)); report["spi_thresholds_bowl"] = list(map(float, th_w))
    d = expected_models(d, report)
    d = attribute(d, mk)

    # ---- player evaluation
    metas, res, parts = {}, {}, {}
    for role, spi, band in (("bat", spi_b, band_b), ("bowl", spi_w, band_w)):
        fixed = a.k_bat if role == "bat" else a.k_bowl
        e, al, inn, h, meta = evaluate(d, spi, band, role, ret_parts=True, fixed_delta=fixed)
        log(f"[evaluate:{role}] {meta}")
        if CFG["n_boot_rank"]:
            top_ids = list(e.index[:30])
            rs = rank_stability(h, role, meta["ks"], top_ids, CFG["n_boot_rank"])
            e = e.join(rs)
        metas[role], res[role], parts[role] = meta, e, (al, inn, h[h.hp])
    report["batting_meta"], report["bowling_meta"] = metas["bat"], metas["bowl"]
    bat, bowl = res["bat"], res["bowl"]

    keep = ["player_id", "name", "rank", "reliability", "level", "level_lo90", "level_hi90", "p_above_avg",
            "level_base", "clutch_delta", "impact", "impact_sd",
            "consistency", "cons_raw", "breadth", "ctx_expected_hits", "ctx_eligible", "conf", "n", "neff_hp", "n_inn",
            "p_top10", "rank_p05", "rank_p50", "rank_p95", "median_npv_inn", "inn_npv_p25", "best_inn_share",
            "npv100_raw_v2", "npv100_raw_v1", "wpa100_raw", "rae100", "wk_delta100", "runs_per100", "bnd_pct",
            "control_rate", "avg_LI", "avg_spi", "seasons"]
    for role, r in (("batters", bat), ("bowlers", bowl)):
        rr = r.assign(player_id=r.index)
        cols = [c for c in keep if c in rr] + [c for c in rr if c.startswith("level_") and c not in keep and c != "level_sd"]
        rr[cols].to_csv(out / f"ranking_{role}_full.csv", index=False)
        rr[cols].head(10).to_csv(out / f"TOP10_{role}.csv", index=False)
    baseline_from_hp(bat.head(25), parts["bat"][2], "bat").to_csv(out / "baseline_comparison_batters.csv")
    baseline_from_hp(bowl.head(25), parts["bowl"][2], "bowl").to_csv(out / "baseline_comparison_bowlers.csv")
    annotate(bat, parts["bat"][1], parts["bat"][2], "bat").to_csv(out / "squad_fit_batters_TEMPLATE.csv")
    annotate(bowl, parts["bowl"][1], parts["bowl"][2], "bowl").to_csv(out / "squad_fit_bowlers_TEMPLATE.csv")
    d.assign(spi_bat=spi_b, spi_bowl=spi_w, band_bat=band_b, band_bowl=band_w)[
        ["match_id", "innings", "over", "legal_before", "score_before", "wkts_before", "target", "runs_needed", "rrr",
         "striker_id", "bowler_id", "m", "cls", "wp_bat", "LI", "L", "J_bat", "J_bowl", "C", "P", "R", "D", "E",
         "Q", "F", "B", "S", "live", "spi_bat", "spi_bowl", "band_bat", "band_bowl", "erem", "wkt_cost",
         "eb_runs", "eb_wkt", "ew_runs", "ew_wkt", "npv_bat", "npv_bowl", "wpa_bat", "wpa_bowl"]
    ].to_csv(out / "ball_level_spi.csv.gz", index=False)

    if not a.skip_validation:
        validate(d, spi_b, spi_w, band_b, band_w, bat, bowl, out, report, metas)
    figures(d, spi_b, th_b, parts["bat"][0], parts["bowl"][0], calib, out, a.sample_match)
    report["runtime_sec"] = round(time.time() - t0, 1)
    (out / "run_report.json").write_text(json.dumps(report, indent=2, default=float))

    show = ["name", "reliability", "level", "level_lo90", "level_hi90", "clutch_delta", "impact", "n", "n_inn",
            "consistency", "breadth", "p_top10"]
    for role in ("bat", "bowl"):
        kk = metas[role]["ks"]
        log(f"[k:{role}] ability k={kk['k_main']:.0f}  pressure-delta k={kk['k_delta']:.0f} (EB {kk['k_delta_eb']:.0f})  "
            f"true-ability SD={kk['tau_main_per100']:.1f}/100  clutch SD={kk['tau_delta_per100']:.1f}/100")
    log("\n=== TOP 10 BATTERS ===");  log(bat.head(10)[[c for c in show if c in bat]].round(3).to_string())
    log("\n=== TOP 10 BOWLERS ==="); log(bowl.head(10)[[c for c in show if c in bowl]].round(3).to_string())
    log(f"\nOutputs in {out.resolve()}  ({report['runtime_sec']} s)")


if __name__ == "__main__":
    main()
