# Pressure Pipeline v2: how it works

**SupeRR Selector S2, Round 1: reliability under pressure**
File: `pressure_pipeline.py` (v2.0). Builds on *Pressure Index Framework v1.0* (`PRESSURE.pdf`).

This document covers four things:

1. How pressure is defined, ball by ball (the **SPI v2**).
2. How the **Markov model** of a T20 innings works, and what it feeds into the rest of the pipeline.
3. How player **reliability under pressure** is measured.
4. **Every change** from the original `pressure_pipeline.py` (v1), and why it was made.

All numbers below come from the full run on the hackathon data pack: 819 matches (IPL "T20" plus T20Is), about 195,000 deliveries, run time about 3 minutes.

---

## Contents

1. [The short version](#1-the-short-version)
2. [Pipeline at a glance](#2-pipeline-at-a-glance)
3. [Data fixes](#3-data-fixes)
4. [The Markov model](#4-the-markov-model)
5. [The pressure definition: SPI v2](#5-the-pressure-definition-spi-v2)
6. [Measuring performance under pressure](#6-measuring-performance-under-pressure)
7. [The reliability score](#7-the-reliability-score)
8. [Validation results](#8-validation-results)
9. [Full change log, v1 to v2](#9-full-change-log-v1-to-v2)
10. [Configuration and command line](#10-configuration-and-command-line)
11. [Output files](#11-output-files)
12. [Judgement calls and limitations](#12-judgement-calls-and-limitations)
13. [References](#13-references)

---

## 1. The short version

- **Pressure = what's at stake on this ball.** A Markov model of the innings gives the exact win probability before every ball, and the expected size of the swing in that probability on the next ball (the *leverage*). Leverage is now the backbone of the pressure score, at 35%. The framework's situational factors (chase difficulty, wickets lost, dot-ball squeeze, new batter, boundary momentum, set batter) make up the rest.
- **Performance under pressure is measured in two currencies.**
  - *Runs* (NPV): a wicket costs whatever it is worth **at that moment** of the innings, not one league-average number.
  - *Wins* (WPA): how much the player moved his team's win probability.
- **Estimates are opposition-adjusted and shrunk honestly.** Batter and bowler strengths are fitted together, so nobody gets credit for scoring against weak attacks. The data cannot separate a "clutch" skill from general ability (Section 6.4). So a player's pressure rating is his all-situation ability, with high-pressure balls weighted more, plus a heavily shrunk "clutch delta".
- **Three data problems were fixed** that distorted v1: wrong extras accounting, a mis-spelt dismissal type slipping through a filter, and chase targets that contradicted the official results.

---

## 2. Pipeline at a glance

```
ball_by_ball.csv + matches.csv + competitions.csv
        │
        ▼
[1] Load & clean ── run accounting fix, target reconciliation, drop super overs / revised / no-result
        │
        ▼
[2] Match state per ball ── balls left, wickets, score, runs needed, dots, streaks, previous-ball class
        │
        ▼
[3] MARKOV MODEL
     ├─ estimate the odds of each ball outcome given the state (smoothing levels tuned out-of-fold)
     ├─ backward induction ─► win probability tables V1 (1st innings), V2 (chase)
     │                     └► expected-runs table E (resources, wicket cost)
     └─ out-of-fold calibration of win probability
        │
        ▼
[4] Per-ball pressure ── leverage, jeopardy, chase difficulty, resources, squeeze, entry …
        │                 ─► SPI_bat, SPI_bowl (0-10) ─► bands Low/Medium/High/Extreme
        ▼
[5] Expected-outcome model (gradient boosting, out-of-fold) ─► E[runs], P(wicket) per ball
        │
        ▼
[6] Per-ball value ── NPV (runs, state-dependent wicket cost) and WPA (win probability added)
        │
        ▼
[7] Player model ── hierarchical ridge: ability + clutch delta + opposition (empirical Bayes)
        │            consistency, breadth, confidence ─► RELIABILITY score, rank intervals
        ▼
[8] Validation ── direction tests, split-half stability, sensitivity, calibration ─► run_report.json
```

---

## 3. Data fixes

### 3.1 Run accounting (the biggest bug in v1)

`extras1_runs` is a **code for the type of extra** (wide = 2, leg-bye = 3, …), not a number of runs. Bye and leg-bye rows carry `runs = 0`, and the actual number of runs for each extra is not recorded anywhere. v1 added +1 for wides and no-balls only, so byes and leg-byes were lost from team totals.

v2 counts team runs per ball as:

```
team_runs = runs + 1·[wide | no-ball | bye | leg-bye] + extras_2_runs   (+5 for a penalty)
bat_runs  = runs  (0 on byes / leg-byes)
bowl_runs = bat_runs + 1·[wide | no-ball] + extras_2_runs on wides / no-balls
```

The pipeline checks each rule against the official "won by N runs" margins (396 matches):

| Rule | Margin reproduced exactly | Within 3 runs |
|---|---|---|
| bat runs only | 6% | 44% |
| v1: + 1 per wide/no-ball | 12% | 63% |
| **v2: + 1 per extra + extras_2_runs** | **32%** | **75%** |

The match isn't perfect because multi-run byes and wides aren't recorded. That noise is unavoidable with this data.

### 3.2 Target reconciliation

Because totals are approximate, a chase can "fall short" in the ball data even though the official result says the chasing side won (or the reverse). Any target that disagrees with the result is corrected:

- **Chasing side won, but its computed total is below the target:** target = chase total.
- **Side batting first won by N runs, but the ball data says the target was reached:** target = chase total + N + 1.

This happened in 115 matches. Without it, the win probability at the final ball would point to the wrong winner and corrupt every win-probability-added (WPA) number in those matches.

### 3.3 Other cleaning

- The dismissal filter used the regex `obstruct`, but the data spells it **"Obstracting the field"**. It now matches `obst`. These dismissals and run-outs are not credited to the bowler.
- Retired-out is not counted as the batter's dismissal in his NPV, but it does count as a wicket in the innings state.
- Super overs (innings 3/4), revised (DLS) matches, no-results and matches without both innings are removed.
- Each ball gets a **Markov outcome class** and the **previous legal ball's class** (`m`). Both are explained in Section 4.

---

## 4. The Markov model

### 4.1 The idea in one paragraph

A T20 innings is a sequence of balls. If we know the situation (balls left, wickets down, runs needed), the next ball's outcome follows a probability distribution (dot, single, four, six, wicket, wide…) that depends **only on that situation**. That is a Markov chain. Once those probabilities are estimated from data, we can work backwards from the end of the innings and compute, **exactly**, the probability of winning from *every* possible situation. No simulation is needed. Win probability, leverage, the value of a wicket and remaining resources all come out of this one model.

### 4.2 The state

| Symbol | Meaning | Range |
|---|---|---|
| `f` | format (IPL "T20" or T20I) | 2 values |
| `inn` | innings | 1 or 2 |
| `b` | legal balls remaining | 0 to 120 |
| `w` | wickets fallen | 0 to 10 |
| `r` | runs needed (2nd innings) | 0 to 321 |
| `s` | score so far (1st innings) | 0 to 330 |
| `m` | class of the previous legal ball: 0 = run or start, 1 = dot, 2 = boundary, 3 = wicket | 4 values |

Adding `m` makes this a **second-order** chain. The next ball can depend on what just happened (momentum, or the batter being "stuck"). This follows the higher-order Markov approach of Bandyopadhyay & Bhattacharjee. As Section 4.4 shows, the data then gets to decide whether that memory actually helps.

### 4.3 Outcomes

There are 11 outcome classes per delivery:

| Class | `0` | `1` | `2` | `3` | `4` | `6` | `W` | `X1` | `X2` | `X5` | `X7` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Runs | 0 | 1 | 2 | 3 | 4 | 6 | 0 | 1 | 2 | 5 | 7 |
| Uses a legal ball? | yes | yes | yes | yes | yes | yes | yes | **no** | **no** | **no** | **no** |

- `X*` are wides and no-balls, grouped by total runs conceded (1, 2–3, 4–5, 6+). They add runs **without** using up a ball.
- A legal 5 is grouped with `4`.
- A wicket on a wide or no-ball is treated as the extra. This is rare.

### 4.4 Estimating the outcome probabilities

The model needs `p(outcome | state)` for every cell of

```
format × innings × over (20) × wickets bucket (5) × required-rate bucket (5) × previous-ball class (4)
```

That is 8,000 cells and 11 outcomes each. Most cells are rare, so raw frequencies would be noise. The pipeline uses **hierarchical Dirichlet smoothing**: each finer level is pulled toward the level above it.

```
p0 = (innings, over)                         ← raw frequency, plenty of data
p1 = + wickets bucket        = (N1 + α_wb · p0) / (n1 + α_wb)
p2 = + required-rate bucket  = (N2 + α_rb · p1) / (n2 + α_rb)
p3 = + previous-ball class   = (N3 + α_m  · p2) / (n3 + α_m)
p4 = + format                = (N4 + α_f  · p3) / (n4 + α_f)
```

- **Wickets buckets:** 0–1, 2–3, 4–5, 6–7, 8–9.
- **Required-rate buckets** (runs needed per ball, chase only): below 0.75, 0.75–1.25, 1.25–1.6, 1.6–2.0, above 2.0. That is roughly below 4.5, 4.5–7.5, 7.5–9.6, 9.6–12 and above 12 runs per over. Conditioning on required rate matters because batters change their aggression to match the chase.
- **Each α is a "pseudo-count"**, i.e. how many balls of evidence a cell needs before it's trusted over its parent level. A very large α switches that level off.

**The α values are tuned automatically.** A coordinate search over {10, 50, 200, 800, 3000, 10⁵} for each level minimises the **out-of-fold log-loss** of next-ball outcomes. The model is trained on odd-numbered matches and tested on even ones, then the other way round.

Results:

| Model (out-of-fold log-loss per ball, lower is better) | Log-loss |
|---|---|
| Outcome rates by over only | 1.6133 |
| First-order chain (state, no memory) | 1.6092 |
| Second-order chain (state + previous ball) | 1.6092 |

- Chosen smoothing: α_wb = 800, α_rb = 800, α_m = 3000, α_f = 3000.
- **Finding:** the game state (wickets, required rate, format) improves prediction. The previous-ball memory adds **essentially nothing** (gain ≈ 0.00002 nats per ball). The model keeps the memory term in its structure, but the tuned smoothing collapses it almost to its parent level. For example, in the 19th over of a chase with 4–5 wickets down and more than 12 runs per over needed, the outcome distribution is almost identical whatever the previous ball was:

  | Previous ball | dot | 1 | 4 | 6 | W |
  |---|---|---|---|---|---|
  | run | .159 | .363 | .123 | .121 | .077 |
  | dot | .161 | .361 | .124 | .119 | .079 |
  | boundary | .159 | .364 | .123 | .119 | .078 |
  | wicket | .162 | .362 | .123 | .116 | .080 |

  This is a real result worth stating: **in this data there is no measurable ball-to-ball "momentum" beyond what the scoreboard already explains.**

### 4.5 Solving the chain: backward induction

Let `P_k` be the probability of outcome `k` in the current state, `ρ_k` its runs, and `m'(k)` the new previous-ball class. The pipeline computes three tables per format.

#### (a) Chase win probability `V2[b, w, r, m]`

This is the probability that the chasing side wins with `b` balls left, `w` wickets down and `r` runs needed.

**Boundary conditions:**

```
r ≤ 0                    → 1     (target reached)
b = 0 or w = 10, r = 1   → 0.5   (scores level: tie, treated as a coin-flip super over)
b = 0 or w = 10, r ≥ 2   → 0     (chase failed)
```

**Recursion**, for b = 1 … 120:

```
V2[b,w,r,m] =  Σ_{k ∈ 0,1,2,3,4,6}  P_k · V2[b-1, w,   r-ρ_k, m'(k)]      legal run outcomes
             + P_W                  · V2[b-1, w+1, r,     3    ]      wicket
             + Σ_{x ∈ X1,X2,X5,X7}  P_x · V2[b,   w,   r-ρ_x, m   ]      wide / no-ball: same ball count
```

The last line refers to the *same* ball count `b`, so it is solved by a short fixed-point iteration. Extras are about 5% of deliveries, so 4 iterations converge to better than 10⁻⁵. The whole table (121 × 11 × 322 × 4) is filled with vectorised numpy in well under a second.

#### (b) First-innings win probability `V1[b, w, s, m]`

The terminal value of the first innings is the chance that the side batting first **defends** the resulting score:

```
U(s) = 1 − V2[120, 0, s+1, 0]      (the chase starts from target s + 1)
```

`V1` then uses the same recursion, with the score `s` going **up** instead of runs needed going down. The two innings are linked exactly: the end of innings 1 equals the start of innings 2.

#### (c) Expected remaining runs `E[b, w, m]` (the resource table)

This uses first-innings dynamics, i.e. no chase constraint:

```
E[b,w,m] = ( Σ_legal P_k (ρ_k + E[b-1, w', m'_k])  +  Σ_x P_x ρ_x ) / (1 − Σ_x P_x)
```

It answers "how many more runs would an average side make from here?" and gives two things used throughout:

| Quantity | Formula | Example (IPL table) |
|---|---|---|
| **Expected first-innings total** | `E[120,0,0]` | 177.5 (IPL) · 175.8 (T20I) |
| **Resources remaining** | `E[b,w,0] / E[120,0,0]` | 60 balls, 2 down: 0.56 · 60 balls, 5 down: 0.44 · 12 balls, 6 down: 0.12 |
| **Wicket cost (runs)** | `E[b-1,w,0] − E[b-1,w+1,3]` | 60 balls, 2 down: 5.5 · 60 balls, 5 down: 9.3 · 30 balls, 3 down: 4.0 · 12 balls, 6 down: 1.3 |

The wicket cost is a key improvement. v1 charged **one constant** for every dismissal (17.3 runs for batters, 30.7 for bowlers, taken from the league average). In reality, losing a wicket in the 19th over costs almost nothing, while losing your 5th wicket at halfway costs a lot. The Markov table puts a price on each situation.

### 4.6 Calibration of win probability

The raw chain treats both teams as league-average and ignores team strength, venue and toss. That makes it **over-confident**: situations the chain rates below 10% (3% on average) actually end in a win about 10% of the time, and situations it rates above 90% are won only about 89% of the time. This is a known property of exactly-solved win probability models (Mishra 2026).

v2 fits a **time-varying temperature** out of fold:

```
WP_cal = sigmoid( (1 − c·τ) · logit(WP_raw)  +  bias · τ · side )

τ    = share of the match still to play  (1 at the first ball, 0 at the last)
side = +1 for the side batting first, −1 for the chaser
```

The correction is strongest at the start and **vanishes at the end**, so the final win/loss/tie values stay exact. The two parameters are fitted on out-of-fold predictions (model trained on half the matches, scored on the other half). Fitted values: `c = 0.85`, `bias = −0.08`.

| Win probability quality, out of fold | Brier score | Log-loss | Calibration error (ECE) |
|---|---|---|---|
| Constant (no model) | 0.250 | 0.693 | – |
| Raw Markov chain | 0.188 | 0.567 | 0.062 |
| **Calibrated Markov chain** | **0.183** | **0.535** | **0.023** |

See `fig_wp_calibration.png` and `wp_calibration.csv`.

### 4.7 What the Markov model provides to the rest of the pipeline

| Output | Used for |
|---|---|
| `wp_bat`: win probability of the batting side before each ball | jeopardy (J), dead-rubber discount, WPA |
| **Leverage** `LI`: expected absolute win-probability swing on this ball | the backbone of pressure (L) |
| `E`: expected remaining runs | resources (R), chase difficulty (C), projection deficit (P) |
| `wkt_cost`: runs lost by a dismissal now | the wicket term in NPV |
| `p`: next-ball outcome probabilities | leverage, diagnostics |

**Leverage**, computed exactly for every ball:

```
swing(state) = Σ_k  P_k · | WP(next state after outcome k) − WP(now) |
LI           = swing / average swing over all balls          (so the average LI = 1)
```

- The average win-probability swing per ball is 1.1 percentage points.
- In a tight final over of a chase, LI reaches 4–5 (`fig_leverage_heatmap.png`).

**Martingale check.** With an exact, consistent model, the *expected* change in win probability on any ball is zero. After calibration, the mean absolute drift is 0.0009, which confirms the tables are internally consistent.

---

## 5. The pressure definition: SPI v2

### 5.1 Principle

> **Pressure on a ball = how much is at stake on it, how badly the side is placed, and the situational stressors a player feels.**

v1 measured pressure only from situational stress: chase tension, resources lost, time squeeze. But a requirement of 14 an over is not pressure if the game is already lost, and 8 an over with 2 wickets in hand in the last over is huge pressure. Leverage captures that difference directly, because it is literally the expected effect of this ball on the result.

### 5.2 Components (each scaled 0 to 1)

**Shared, from the Markov model:**

| Code | Name | Definition |
|---|---|---|
| **L** | Leverage | percentile rank of `LI` among all balls (0.9 means "more is at stake than on 90% of balls") |
| **J** (batter) | Jeopardy | `clip(1 − 2·WP_bat, 0, 1)`: how far the batting side is *behind* (0 at evens or better, 1 when the game is nearly lost) |
| **J** (bowler) | Jeopardy | `clip(2·WP_bat − 1, 0, 1)`: how far the bowling side is behind |
| `live` | Game alive | `clip(4·WP·(1−WP) / 0.36, 0, 1)`: 1 while win probability is between 10% and 90%, falling to 0 as the game is decided |

**Batter components:**

| Code | Name | Definition |
|---|---|---|
| **C** | Chase difficulty (2nd innings) | runs needed ÷ expected remaining runs `E[b,w,m]`, i.e. the required rate *relative to the resources left*. Starts at 0.8 × par, capped at the 95th percentile. |
| **P** | Projection deficit (1st innings) | `max(0, expected total − (score + E[b,w,m]))`, capped at the 95th percentile |
| **R** | Resources lost | `1 − E[b,w,0] / E[120,0,0]`, from the Markov table (DLS-like, but fitted to this data) |
| **D** | Dot squeeze | `dots in last 6 / 6`, plus 0.15 if the current dot streak is 4 or more ("acute"), capped at 1 |
| **E** | Entry | new batter after a collapse (wicket in the last 14 balls when he came in): 1.0 for his first 10 balls, then tapering to 0 by ball 25. Any new batter: 0.3 for his first 10 balls. |

**Bowler components:**

| Code | Name | Definition |
|---|---|---|
| **Q** | Phase | powerplay 0.8, middle overs 0.3, death 1.0 |
| **F** | Wickets in hand (batting side) | `(10 − wickets) / 10` |
| **B** | Boundary momentum | `boundaries + wides/no-balls in last 6 / 3`, capped at 1 |
| **S** | Set-batter threat | `max(balls faced / 20, strike rate / 150 once he has faced 10 balls)` |

### 5.3 Weights

| Batter, 2nd innings | Batter, 1st innings | Bowler |
|---|---|---|
| L .35 | L .35 | L .35 |
| J .15 | J .15 | J .15 |
| C .20 | P .15 | Q .10 |
| R .10 | R .15 | F .05 |
| D .10 | D .10 | B .15 |
| E .10 | E .10 | S .20 |

```
SPI_raw  = Σ weight · component                               (0 to 1)
SPI      = 10 · SPI_raw · (1 − g + g·live),   g = 0.5          (0 to 10)
```

The `(1 − g + g·live)` factor is the **dead-rubber discount**. Once a game is effectively decided, the ball's SPI is halved, so slogging in a lost chase or bowling at 9 down with 60 needed isn't counted as high pressure.

### 5.4 Bands

Bands are percentiles of SPI within the dataset, computed separately for batters and bowlers:

| Band | Percentile range | Batter SPI | Bowler SPI |
|---|---|---|---|
| Low | below p50 | below 3.64 | below 4.44 |
| Medium | p50 to p80 | 3.64 to 5.10 | 4.44 to 6.01 |
| High | p80 to p95 | 5.10 to 6.26 | 6.01 to 7.28 |
| Extreme | p95 and above | 6.26 and above | 7.28 and above |

**High-pressure (HP) balls = High + Extreme**, the top 20%. For batters, 58% of HP balls fall in chases, 36% in the death overs and 14% in the powerplay.

### 5.5 v1 vs v2 pressure definition

| | v1 | v2 |
|---|---|---|
| What's at stake | not measured | **Leverage** from the exact Markov win probability (35%) |
| Who's behind | chase tension (required rate vs current rate) / defence margin K | **Jeopardy** from win probability, symmetric for bat and ball |
| Chase difficulty | required rate vs current rate and vs initial required rate | runs needed **÷ resources remaining** (Markov expected runs) |
| Resources | rough over × wicket table | Markov `E[b,w]` table, per format |
| Time squeeze T | separate component (20%) | dropped: leverage already rises late in close games |
| Dead games | counted in full | discounted (`live` factor) |
| Dot squeeze | dots in last 6 | + "acute streak" bonus at 4 dots |
| Bowler defence margin K | runs needed − expected runs | replaced by bowler jeopardy (win-probability based) |

---

## 6. Measuring performance under pressure

### 6.1 Expected outcome per ball

Four gradient-boosted models (scikit-learn `HistGradientBoosting`) predict what an **average** player would produce on each ball:

- `E[bat runs]`
- `P(batter out)`
- `E[runs conceded]`
- `P(bowler's wicket)`

**Features:** over, wickets, balls left, innings, required rate, runs needed, dot streak, dots in last 6, score, format, venue, pace/spin, left/right-handed batter, previous-ball class, Markov win probability.

All predictions are **out of fold** (5-fold, grouped by match), so a player's own balls never inform his expectation.

| Model | Out-of-fold | Baseline |
|---|---|---|
| Batter runs (RMSE) | 1.61 | 1.74 |
| Batter out (log-loss) | 0.192 | 0.211 |
| Runs conceded (RMSE) | 1.59 | 1.71 |
| Bowler wicket (log-loss) | 0.182 | 0.201 |

### 6.2 NPV: runs above expectation, with a state-dependent wicket cost

```
NPV_bat  = (bat_runs − E[runs])  −  wkt_cost(state) · (out − P(out))
NPV_bowl = (E[runs conceded] − runs conceded)  +  wkt_cost(state) · (wicket − P(wicket))
```

`wkt_cost(state)` comes from the Markov table (Section 4.5c). v1 used `λ × league-average runs per dismissal` (17.3 for batters, 30.7 for bowlers), which over-charged late-innings dismissals and under-charged middle-overs collapses.

### 6.3 WPA: win probability added

Each ball moves the batting side's (calibrated) win probability. The movement is split so each player is credited only for what he controlled:

- **Batter:** next state built from *his* runs and *his* dismissal only (byes, leg-byes and wides are not his).
  `WPA_bat = WP(after batter's outcome) − WP(before)`
- **Bowler:** next state built from runs conceded (excluding byes and leg-byes) and wickets credited to him.
  `WPA_bowl = −[WP(after bowler's outcome) − WP(before)]`

Because the model is a martingale, the expected WPA of an average player is about 0 in every state. So **WPA is "wins above average" already**. It is reported as *win-% points per 100 high-pressure balls*.

### 6.4 The key statistical finding: there is no detectable "clutch" skill

For each metric, the pipeline estimates how much players **truly** differ (between-player variance τ²) compared with per-ball noise, using empirical Bayes on player means (method of moments):

| | True spread between players, all balls (per 100) | Extra spread on HP balls only ("clutch") |
|---|---|---|
| Batter NPV | 16.0 runs | **≈ 0** |
| Bowler NPV | 12.0 runs | **≈ 0** |
| Batter WPA | detectable | **≈ 0** |
| Bowler WPA | detectable | **≈ 0** |

In other words, players genuinely differ in quality, but once overall ability is accounted for, there is **no measurable extra skill specific to high-pressure balls** in this sample. This matches decades of evidence on "clutch" performance in other sports.

v1 ranked players on raw high-pressure NPV shrunk toward the league mean. On HP balls alone, that is mostly noise: v1's own empirical-Bayes estimate for bowlers was effectively infinite (k ≈ 5×10¹⁰), meaning no signal at all.

### 6.5 Hierarchical, opposition-adjusted player model

For each role, one weighted ridge regression is fitted over **all** of that role's deliveries:

```
y_ball = μ + δ·HP  +  a_player  +  d_player·HP  +  o_opponent  + noise

  y          = NPV (or WPA) on the ball
  a_player   = all-situation ability                  prior strength k_main  (empirical Bayes)
  d_player   = extra performance on HP balls (clutch)  prior strength k_delta (EB, capped)
  o_opponent = quality of the bowler faced (for batters) or batter bowled to (for bowlers)
  δ          = league-wide HP shift (e.g. bowlers concede more under pressure)
```

**Pressure level = `a + d`.** This is the best estimate of how the player performs on high-pressure balls, built from:

- his overall ability, which is well estimated because it uses every ball he played;
- a shrunk adjustment from what he actually did on HP balls;
- an adjustment for the opponents he faced.

**Ball weights** `wt = recency × pressure weight`:

- **Recency:** half-life of 3 seasons, so a 2023 ball counts 50% of a 2026 ball.
- **Pressure weight:** `0.25 + 0.75 × SPI percentile`, so high-pressure balls count about 3–4 times as much as low-pressure balls in the ability estimate.

**Prior strengths** (the "k" in `n/(n+k)` shrinkage) are estimated from the data, not set by hand:

| | k_main (ability) | k_delta (clutch) |
|---|---|---|
| Batters, NPV | 222 | EB says ∞ → capped at **400** |
| Bowlers, NPV | 384 | EB says ∞ → capped at **400** |
| Batters, WPA | 128 | capped at 400 |
| Bowlers, WPA | 208 | capped at 400 |

Ridge regression with these penalties *is* empirical-Bayes shrinkage. A player with `n` effective balls keeps about `n/(n+k)` of his own deviation, and the rest is pulled toward the league average (for ability) or toward his own ability (for the clutch delta). The 400 cap is a judgement call, explained in Section 12.

Each player gets:

- **Posterior SD:** `√(σ²/(n_all + k_main) + σ²/(n_HP + k_delta))`
- **90% interval:** `level ± 1.645 · SD`
- **P(above average):** `Φ(level / SD)`

---

## 7. The reliability score

### 7.1 The five terms

| Term | Weight | Definition |
|---|---|---|
| **Level** | .40 | `a + d`, opposition-adjusted pressure NPV per 100 HP balls (runs) |
| **Consistency** | .25 | share of HP innings or spells (5+ HP balls) with NPV ≥ 0, as a **beta-binomial posterior** with prior strength estimated from the data (batters: 35 pseudo-innings; bowlers: 169, i.e. almost no detectable difference between players) |
| **Breadth** | .15 | expected number of pressure contexts where the player is above average, as a soft count `Σ Φ(context level / SD)`, smoothed `(hits + 1)/(eligible + 2)`. Batter contexts: chasing, setting, collapse entry, death, vs spin, vs pace. Bowler contexts: defending, bowling first, powerplay, middle, death, vs set batter. |
| **Impact** | .10 | WPA per 100 HP balls (win-% points), same hierarchical ridge model |
| **Confidence** | .10 | `n_HP / (n_HP + k_delta)`: how much pressure evidence exists |

### 7.2 Evidence-weighted standardisation

v1 converted every term to a z-score before weighting. That re-inflates a term that is mostly noise (such as consistency) to the same spread as a term that is mostly signal.

v2 divides each estimated term by its **estimated true between-player SD** instead:

```
τ²_term = Var(posterior means) + mean(posterior variance)        (law of total variance)
z_term  = (x − mean) / τ_term
```

A heavily shrunk term ends up with a small spread, so it automatically has little influence on the ranking. For example, bowler consistency carries almost no weight because the data shows no real difference between bowlers on it. Breadth and confidence use ordinary z-scores.

```
Reliability = .40·z(level) + .25·z(consistency) + .15·z(breadth) + .10·z(impact) + .10·z(confidence)
```

### 7.3 Eligibility and rank stability

- **Eligible:** at least 60 HP balls **or** at least 8 HP innings (framework rule). `--strict` requires both. This gives 174 eligible batters and 184 eligible bowlers.
- **Rank stability:** the whole player model is re-fitted on 60 bootstrap resamples of *matches*. For the top 30 players, this reports **P(top 10)** and a **90% rank interval** (`rank_p05` to `rank_p95`).

### 7.4 Current top 10

| # | Batter | Level (90% CI) | P(top 10) | | Bowler | Level (90% CI) | P(top 10) |
|---|---|---|---|---|---|---|---|
| 1 | Suryakumar Yadav | 25.3 (6.0, 44.7) | .87 | | Nathan Ellis | 22.6 (1.0, 44.2) | .73 |
| 2 | Tim David | 26.0 (4.8, 47.2) | .80 | | Matheesha Pathirana | 19.4 (−1.1, 40.0) | .67 |
| 3 | Nicholas Pooran | 24.5 (4.3, 44.7) | .87 | | Arshdeep Singh | 13.0 (−2.9, 29.0) | .50 |
| 4 | Rovman Powell | 22.9 (2.7, 43.0) | .63 | | Jasprit Bumrah | 15.1 (−3.7, 34.0) | .62 |
| 5 | Heinrich Klaasen | 23.9 (2.9, 44.9) | .65 | | Axar Patel | 14.7 (−3.6, 33.0) | .58 |
| 6 | Ashutosh Sharma | 21.8 (−3.2, 46.7) | .53 | | Mustafizur Rahman | 15.2 (−4.5, 34.9) | .45 |
| 7 | Tilak Varma | 15.9 (−2.7, 34.4) | .43 | | Sunil Narine | 14.8 (−5.6, 35.1) | .48 |
| 8 | Shreyas Iyer | 14.1 (−6.5, 34.7) | .33 | | Lungi Ngidi | 20.1 (−1.5, 41.8) | .37 |
| 9 | Andre Russell | 20.8 (−3.3, 44.9) | .32 | | Varun Chakravarthy | 12.4 (−6.3, 31.1) | .33 |
| 10 | Rinku Singh | 15.6 (−5.7, 36.9) | .35 | | Rashid Khan | 9.2 (−8.5, 26.9) | .33 |

Level is NPV in runs per 100 HP balls, relative to league average, opposition-adjusted. The wide intervals and P(top 10) values are the honest message: **ranks 1–5 are fairly solid; ranks 6–15 are close to interchangeable.**

---

## 8. Validation results

| Check (framework section 8, plus new checks) | Result | Verdict |
|---|---|---|
| Batter SPI rises after a wicket | 4.39 vs 3.60 | ✅ |
| Bowler SPI falls after a wicket (margin grows) | 3.44 vs 4.46 | ✅ |
| Chase SPI after an over below the required rate | 4.25 vs 3.34 after an above-rate over | ✅ |
| SPI vs leverage (Spearman correlation) | bat .76, bowl .77 | ✅ anchored but not identical |
| Top 15 kept under ±20% random weight changes (30 runs) | batters 14.7/15 (min 13), bowlers 13.8/15 (min 12) | ✅ (v1: about 12/15) |
| Top 15 kept as k_delta varies 30 → 1000 | batters 11–15, bowlers 12–15 | ✅ moderate |
| Top 15 kept as k_main is halved or doubled | 15/15 and 14–15/15 | ✅ |
| Markov memory effect, out of fold | +0.00002 nats per ball | ℹ️ no momentum effect |
| Win probability calibration error, out of fold | 0.062 raw → 0.023 calibrated | ✅ |
| Martingale drift | 0.0009 | ✅ |

**Split-half stability.** Matches are split into odd and even halves, and the Spearman rank correlation of each player's metric across the two halves is computed (players with 20+ HP balls in both halves):

| Metric | Batters (n = 173) | Bowlers (n = 196) |
|---|---|---|
| **Impact (WPA, hierarchical)** | **.28** | **.19** |
| Level (a + d, v2) | .12 | .13 |
| Ability only (a) | .13 | .15 |
| Breadth | .13 | .15 |
| Consistency | .18 | −.01 |
| Clutch delta (d) only | .05 | .08 |
| v1 metric (raw HP NPV, constant wicket cost) | .15 | −.01 |
| Raw WPA per 100 | .15 | .06 |

How to read this:

- **Bowlers:** the v1 metric had **zero** repeatability. v2's level and impact are clearly repeatable.
- **Batters:** v2's level roughly ties v1's raw NPV. Both are low, because each half has only about 100–200 HP balls per player.
- **Impact (WPA)** is the most repeatable metric for both roles. If the selectors want to put more weight on it, `rel_w` is the knob (Section 10).
- **Clutch delta** on its own barely repeats. That confirms Section 6.4.

---

## 9. Full change log, v1 to v2

| Area | v1 (original file) | v2 (new file) | Why |
|---|---|---|---|
| **Run accounting** | `runs` + 1 per wide/no-ball; byes and leg-byes lost | `runs` + 1 per wide/no-ball/bye/leg-bye + `extras_2_runs`; separate `bat_runs` / `bowl_runs` | `extras1_runs` is a type code; margin check 12% → 32% exact |
| **Target** | first-innings total + 1 | reconciled with the official result (115 matches) | final win probability must agree with the actual winner |
| **Dismissals** | `obstruct` filter missed "Obstracting" | `obst`; retired-out excluded from the batter's NPV | data spelling |
| **Game model** | none (lookup tables of runs by over × wickets) | **second-order Markov chain**, exact backward induction, win probability + expected runs, per format | gives win probability, leverage, wicket value |
| **Smoothing** | – | 4-level hierarchical Dirichlet, levels tuned by out-of-fold log-loss | sparse cells; let the data decide on memory |
| **Win probability calibration** | – | time-varying temperature, fitted out of fold | raw chain is over-confident early |
| **Pressure backbone** | C/P, R, T (situational) | **L (leverage) 35%** + J (jeopardy) | pressure = what's at stake |
| **Chase difficulty C** | (required − current rate) + required rate drift | runs needed ÷ Markov expected runs | resource-adjusted |
| **Time squeeze T** | 20% | removed | duplicated by leverage |
| **Bowler margin K** | runs-based margin | bowler jeopardy from win probability | symmetric with batters |
| **Dead rubbers** | full weight | SPI × (0.5 + 0.5·live) | no pressure in decided games |
| **Expected model** | gradient boosting, state features | + previous-ball class, + win probability, + venue/handedness/pace-spin | richer context |
| **Wicket cost in NPV** | λ × league-average runs per dismissal (constant) | **Markov wicket cost for the current state** | a wicket at over 19 ≠ a wicket at over 10 |
| **Win currency** | – | **WPA** per ball, split between batter and bowler | ties performance to results |
| **Shrinkage** | `n/(n+k)` toward the league mean on HP balls, fixed k = 60/48 | **hierarchical ridge**: ability (all balls) + clutch delta (HP) + opponent; k from empirical Bayes | HP-only signal is ≈ 0; v1's own EB check implied k ≈ 373 for batters and ∞ for bowlers |
| **Opposition** | ignored | opponent random effects in the same regression | T20Is include weak attacks |
| **Recency** | none | 3-season half-life weights | auction is about the next season |
| **Consistency** | Laplace-smoothed hit rate `(hits + 2)/(n + 4)` | beta-binomial with EB prior strength | measured repeatability differs by role |
| **Breadth** | count of contexts above average | soft expected count `Σ Φ(level/SD)` | rewards confident outperformance, not luck |
| **Composite scaling** | plain z for all terms | evidence-weighted z (divide by true SD) | stop re-inflating noise |
| **Composite weights** | NPV .45, cons .30, breadth .15, conf .10 | level .40, cons .25, breadth .15, **impact .10**, conf .10 | adds win currency |
| **Uncertainty** | innings-bootstrap CI on raw NPV (top 30) | posterior 90% interval + **match-bootstrap rank stability** (P(top 10), rank interval) | communicate how solid a rank is |
| **Validation** | direction, split-half, weight and k sensitivity, baseline | + Markov order test, out-of-fold win probability calibration, martingale check, old-vs-new split-half, HP composition, leverage correlation | |
| **Figures** | SPI distribution, sample pressure curve, shrinkage | + win probability "worm" panel, calibration plot, leverage heatmap; no dual axes | |

---

## 10. Configuration and command line

```bash
# full run (about 3 minutes)
python pressure_pipeline.py --bbb data/ball_by_ball/ball_by_ball.csv --data-dir data --out out

# quick iteration (fewer sensitivity / bootstrap runs, about 80 seconds)
python pressure_pipeline.py --bbb "data/ball_by_ball/*" --data-dir data --out out --fast
```

| Flag | Effect |
|---|---|
| `--bbb` | one or more CSV/ZIP files or glob patterns |
| `--data-dir` | folder with `matches.csv` and `competitions.csv` |
| `--k-bat`, `--k-bowl` | fix the clutch-delta prior strength (default: empirical Bayes, capped at 400) |
| `--half-life` | recency half-life in seasons (`0` = off) |
| `--min-hp-balls`, `--min-innings`, `--strict` | eligibility rules |
| `--n-boot` | rank-stability bootstrap replicates (default 60) |
| `--sample-match` | match id for the pressure-curve figure |
| `--skip-validation`, `--fast` | speed |

Main `CFG` settings at the top of the file:

| Setting | Default | Meaning |
|---|---|---|
| `w_bat_2nd`, `w_bat_1st`, `w_bowl` | Section 5.3 | SPI weights |
| `garbage_discount` | 0.5 | how much a decided game discounts SPI |
| `band_q` | (.50, .80, .95) | band cut-offs |
| `markov_alpha` | auto-tuned | smoothing strengths (wickets, required rate, memory, format) |
| `rb_edges` | (.75, 1.25, 1.6, 2.0) | required-rate buckets, runs per ball |
| `rel_w` | (.40, .25, .15, .10, .10) | reliability weights |
| `k_delta_max` | 400 | cap on the clutch prior strength |
| `pressure_weight_floor` | 0.25 | weight of the lowest-SPI ball in the ability model |
| `half_life` | 3.0 | seasons |

---

## 11. Output files

| File | Contents |
|---|---|
| `TOP10_batters.csv`, `TOP10_bowlers.csv` | top 10 with all metric columns |
| `ranking_*_full.csv` | all eligible players: reliability, level (+ 90% CI), `level_base`, `clutch_delta`, impact, consistency, breadth, confidence, P(top 10), rank interval, context levels, v1-style raw metrics for comparison |
| `baseline_comparison_*.csv` | top 25 ranked by v2 vs HP strike rate / average / economy / v1 NPV |
| `squad_fit_*_TEMPLATE.csv` | top 15 with per-format NPV, season trend and blank scouting columns |
| `ball_level_spi.csv.gz` | every ball: state, win probability, leverage, every SPI component, SPI, band, expectations, NPV, WPA |
| `markov_chain.npz` | outcome-probability tensor `p` and expected-runs table `E`, per format |
| `wp_calibration.csv` | out-of-fold calibration table (raw vs calibrated) |
| `face_validity_sample.csv` | top-SPI, top-leverage and random balls, for an eyeball check |
| `run_report.json` | config, data checks, Markov diagnostics, k values, every validation result |
| `fig_spi_distribution.png` | batter SPI histogram with band cut-offs |
| `fig_pressure_curve_sample_match.png` | win probability worm + SPI for one match |
| `fig_shrinkage_batters.png`, `fig_shrinkage_bowlers.png` | raw vs shrunk level against sample size |
| `fig_wp_calibration.png` | raw vs calibrated win probability |
| `fig_leverage_heatmap.png` | mean leverage by over × wickets in chases |

---

## 12. Judgement calls and limitations

1. **The clutch cap (`k_delta_max = 400`).** The data alone would shrink the clutch delta to zero, which would make the pressure rating *just* pressure-weighted overall ability. The cap lets a player with about 400 HP balls move his rating about halfway toward what he actually did under pressure. Lowering it (`--k-bat 60` matches the framework's original value) rewards observed HP performance more, but changes 2–4 names in the top 15 and adds noise.
2. **SPI weights are expert-set**, not learned. Pressure has no direct observable label to train against. The weights are shown to be robust (top 15 stable under ±20% changes), but they aren't "optimal".
3. **Extras run counts aren't in the data**, so innings totals are approximate (75% within 3 runs). The chase target is reconciled to the official result, but individual balls can still be off.
4. **The Markov chain treats both sides as league-average.** The calibration fixes over-confidence on average, but team strength, venue and toss aren't in the state.
5. **Ties count as 0.5.** Super overs aren't modelled.
6. **Small samples:** even the best players have only 200–800 HP balls. That's why the intervals are wide and why ranks below about 6 should be read as tiers, not exact positions.

---

## 13. References

- Bandyopadhyay, S. & Bhattacharjee, D. *Higher-order Markov models and a Pressure Index for T20 run chases.* arXiv:2505.01849. https://arxiv.org/abs/2505.01849
- Mishra. *Calibration-Leverage Tradeoff in Exactly Solvable Win-Probability Models* (2026). arXiv:2608.14696. https://arxiv.org/abs/2608.14696
- Misra & Rajaraman. *Delta: leverage-weighted player contributions* (2026). https://www.cs.columbia.edu/~misra/delta-2026.pdf
- Ganesh. *Simulation-based optimisation with James–Stein shrinkage.* arXiv:2604.13861. https://arxiv.org/abs/2604.13861
- Tango, Lichtman & Dolphin. *The Book: Playing the Percentages in Baseball* (2007): origin of the Leverage Index and WPA.
- *Pressure Index Framework v1.0* (`PRESSURE.pdf`), the hackathon framework this pipeline implements.
