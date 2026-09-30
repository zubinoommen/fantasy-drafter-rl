# Synthetic Fantasy Draft MDP

A research-oriented, reproducible Python environment for studying fantasy-football
draft decisions before adding reinforcement-learning algorithms. V1 models a
heterogeneous family of 8/10/12-team full-PPR snake drafts, configurable
lineups and benches, one controlled team, stochastic opponent/continuation
policies, a fourteen-week season, and lightweight weekly waiver replacements.

The package deliberately separates drafting, state representation, policies,
season uncertainty, utility, Monte Carlo evaluation, and datasets. It does not
implement policy iteration, value iteration, Q-learning, SARSA, or TD methods.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

For an executable end-to-end tour, open
[`notebooks/environment_walkthrough.ipynb`](notebooks/environment_walkthrough.ipynb).
It displays league configuration, player distributions, individual snake-draft
picks, state features, a completed roster, weekly lineups, Monte Carlo values,
and heterogeneous generated-state metadata.

## Draft as an MDP

A draft state contains all fantasy rosters, the undrafted player IDs, current
pick and round, league settings, and the synthetic player parameters. An action
is the ID of one legal available player. Once an action is chosen, the transition

\[
s_{t+1}=T(s_t,a_t)
\]

is deterministic: the player leaves availability, enters the current team's
roster, and the snake cursor advances. Randomness belongs to policies choosing
actions and to terminal season outcomes, not to `DraftEnvironment.step`.

Legality is intentionally physical and strategy-neutral. Any available player
may be drafted while the team has an open roster slot. Optional position limits
apply only when explicitly configured by the league. The environment does not
guarantee starter or FLEX feasibility: an agent may draft ten QBs or omit TE
entirely and receives the resulting season score. There are no ADP,
positional-scarcity, replacement-value, or hand-authored round rules.

`LeagueConfig` controls team count, draft position, season length, scoring
context, and roster configuration. `RosterConfig` independently specifies
required QB/RB/WR/TE starters, FLEX count and eligibility, bench capacity, and
optional physical position limits. Draft rounds default to roster size but may
be explicitly shorter.

Baseline policy preferences remain separate from legality. The
configuration-aware softmax baseline scores players using raw `mu`, marginal
value above a projected position-specific replacement level, and soft
starter/FLEX needs. Replacement thresholds derive generically from team count,
starter requirements, and FLEX eligibility, so QB preference rises naturally
in 2-QB leagues. Within a generated draft, the controlled team and every
opponent use the same sampled baseline policy; policy families still vary
across generated drafts.

## Synthetic players

Each `Player` has `player_id`, actual `position`, weekly `mu`, weekly `sigma`,
weekly unavailability probability, and a synthetic NFL `team_id`. The default
pool has 24 QBs, 36 RBs, 42 WRs, and 18 TEs. Position-specific noisy rank-decay
curves produce distinct full-PPR scoring, volatility, injury, and depth
profiles. The values are plausible abstractions, not calibrated NFL forecasts.
Profiles in `config.py` can be replaced independently with empirical data.

Weekly player outcomes are conditionally independent in V1. The explicit
`team_id` and player-keyed shock generation leave room for later stacking and
correlation models.

## State vector

Encoder version 2 returns 390 fixed `float32` features for every supported
league:

1. Configuration and draft context (64): encoder version, explicit team count
   and team masks, controlled draft position and one-hot mask, current round,
   normalized progress, intervening picks, total/remaining rounds, total/open
   roster slots, bench and season lengths, scoring context, starter counts,
   FLEX count/eligibility, and optional position-limit values/masks.
2. Available board (160): 120 player values for the top ten QB/RB/WR/TE players
   plus forty occupancy masks.
3. Controlled roster (160): twenty canonical slots, each containing
   `(mu, sigma, injury_probability)`, a position one-hot vector, and occupancy
   mask. Ordering is deterministic by actual position, mean, and ID.
4. Remaining requirements (6): QB, RB, WR, TE, FLEX deficits and unconstrained
   reserve openings.

Canonical maxima are 16 teams, 20 roster slots, four starters per actual
position, and four FLEX slots. Larger configurations fail explicitly instead
of silently changing dimensionality. `feature_names()` and `slices()` expose
the versioned layout. IDs, full draft history, ADP, hand-built scarcity, future
actions, simulated scores, utilities, and Monte Carlo estimates are absent.

The waiver pool is also omitted from the model-facing vector in V1. Waiver
quality is terminal environment dynamics. A future season-management or trade
state may need to expose it explicitly.

## Season and waiver simulation

The completed draft retains all undrafted players as free agents. Each week:

1. Player availability is sampled from injury probabilities.
2. Lineup holes are computed after healthy rostered depth is considered.
3. An injected replacement model selects eligible, currently available free
   agents without seeing realized scores or future outcomes.
4. The legal lineup is selected ex ante from known player projections (`mu`);
   realized weekly production is not available to the lineup decision.
5. Selected starters then receive `max(0, Normal(mu, sigma))` scores.

`PoolReplacementLevelModel` is the default. It samples from each position's
actual undrafted depth with exponentially decreasing rank weights. It neither
always grants the best free agent nor permanently claims weekly replacements,
and one free agent cannot fill two slots in the same week. This gives an
injury cost closer to

\[
p_{\text{inj}}\left(E[S_{\text{starter}}]-E[R_p]\right)
\]

than to a forced zero. `NoReplacementModel` and `StochasticWaiverPolicy` permit
comparisons. The protocol can later support persistent transactions without
changing the draft or evaluator.

The simulator returns weekly scores and lineup diagnostics. Utility is a
separate callable. The default terminal utility is

\[
U(s_T,\omega)=\frac{1}{W}\sum_{w=1}^{W}S_w.
\]

Median, quantile, risk-penalized mean, and threshold-probability helpers are
also provided.

## Monte Carlo state evaluation

```python
from fantasy_draft import (
    DraftEnvironment,
    RosterAwareSoftmaxPolicy,
    evaluate_state,
    generate_players,
)

players = generate_players(seed=7)
state = DraftEnvironment().initial_state(players)

result = evaluate_state(
    state=state,
    continuation_policy=RosterAwareSoftmaxPolicy(),
    n_draft_rollouts=20,
    n_season_simulations=25,
    seed=42,
)
print(result.mean_value, result.standard_error)
```

For each of \(M\) independent draft continuations, the evaluator simulates
\(K\) seasons and computes

\[
\widehat V^\pi(s)=\frac{1}{M}\sum_{m=1}^{M}
\left[\frac{1}{K}\sum_{k=1}^{K}U_{m,k}\right].
\]

One completed draft estimates only one terminal roster's value. Multiple draft
rollouts are required to integrate over stochastic future policy decisions.
For a declared deterministic continuation policy, the identical draft
continuation is computed once and reused across the outer statistical groups.
The reported sample standard deviation is across the \(M\) per-rollout means;
its standard error is `std / sqrt(M)`. Raw nested utilities, terminal roster
IDs, per-rollout means, and all seeds are retained.

`EvaluationSeedPlan` can be reused for matched comparisons. Season shocks are
indexed by player and week over the full player universe, so shared players
receive common random numbers across candidate-state evaluations.

## Reachable states and datasets

State generation samples a Cartesian configuration family spanning 8/10/12
teams, 1/2 QB, 1/2 FLEX, and multiple bench sizes. It varies controlled draft
position, position-specific player distributions, player-pool and history
seeds, and roster-aware policy parameters. The same configuration-aware
replacement-value softmax policy controls every team.
Generation and evaluation are separate:

```bash
python scripts/generate_states.py --states 100 --seed 42 --output states.jsonl
python scripts/evaluate_states.py states.jsonl --draft-rollouts 10 \
  --season-simulations 10 --workers 8 --seed 42 --output values.npz
```

JSONL state records contain the full reconstructable `DraftState`, league
configuration ID, team/roster/lineup settings, controlled draft position,
player-pool and history seeds, opponent-policy configuration, encoder version,
and format version. NPZ observations retain this structural metadata beside
the vector, estimate, standard error, policy, parameters, state ID, and seed.

For model training, generate post-action states directly in Parquet:

```bash
pip install -e ".[data]"
python scripts/generate_training_parquet.py \
  --samples 10000 --draft-rollouts 5 --season-simulations 10 \
  --workers 8 --seed 42 --output data/post_pick_values.parquet
```

Every row is the exact state \(s'=T(s,a)\) immediately after the controlled
team's pick. `reward` is the nested Monte Carlo estimate \(V^\pi(s')\), where
the canonical `RosterAwareSoftmaxPolicy` makes all remaining picks for every
team. The fixed-size `state_vector` has 390 float32 features; feature names and
reward/state semantics are stored in Parquet schema metadata. The rows cover
all supported league configurations and all draft phases, with rounds 1–3
weighted four times as heavily as late rounds. At inference, enumerate legal
actions, encode each resulting \(T(s,a)\), and choose the action whose predicted
value is largest.

Benchmark correctness-scale workloads before scaling:

```bash
python scripts/benchmark.py --states 100 --draft-rollouts 10 \
  --season-simulations 10 --seed 42
```

For the proposed larger run, change the arguments to `--states 5000
--draft-rollouts 20 --season-simulations 25`; do not assume its runtime from
sample counts alone.

On the development machine, the original homogeneous implementation evaluated
the seeded 100-state / 10-rollout / 10-season workload in 44.821 seconds. The
heterogeneous version evaluates it in 7.901 seconds (1,265.6 season
samples/second), an 82.4% wall-time reduction despite broader configurations.
The matched profiler case fell from 2.315 to 0.178 seconds (92.3%): cumulative
`legal_actions` time fell from 2.069 to 0.0006 seconds, `continue_draft` from
2.127 to 0.0076 seconds, and season simulation now accounts for about 0.174
seconds. A linear single-process projection for 2.5 million season samples is
about 33 minutes; process-based state evaluation can reduce wall time further.

## NFLfastR calibration and hindsight benchmark

The default full-PPR player profiles are calibrated to 2018–2025 regular-season
data from the official nflverse player-stats and weekly-rosters releases. Run:

```bash
pip install -e ".[ml]"
python scripts/calibrate_nflfastr.py
```

Downloads are cached under `data/nflverse_cache/`. The reproducible outputs in
`data/nflfastr/` include:

- `player_seasons.parquet`: healthy-week mean and standard deviation plus
  observed weekly unavailability for each eligible player-season.
- `position_rank_summary.parquet` and `calibration.json`: rank curves,
  quantiles, fitted generator profiles, filters, and source URLs.
- `hindsight_benchmark.parquet` and its JSON summary: paired MLP versus
  `RosterAwareSoftmaxPolicy` results on each season's ex-post distributions.

Bye weeks are removed by joining weekly rosters to observed team-game weeks.
An active player with no stats in a game receives zero; non-active roster
statuses estimate unavailability. The checked-in defaults in `config.py` are a
snapshot of this calibration, so normal package use has no network dependency.

The benchmark is deliberately labeled *hindsight distribution evaluation*. It
simulates snake drafts using distributions learned after each NFL season. It
does not reconstruct real fantasy drafts or historical player availability,
because nflfastR does not provide fantasy ADP/draft-room histories. The MLP is
an approximate value model under distribution shift, so outperformance is an
empirical result—not a policy-improvement guarantee.

## Monte Carlo fitted policy iteration

[`notebooks/fitted_policy_iteration.ipynb`](notebooks/fitted_policy_iteration.ipynb)
runs three evaluate/fit/improve rounds with 20,000 newly generated states per
round. The controlled team starts on `RosterAwareSoftmaxPolicy`; after each
value fit it chooses

\[
\pi_{i+1}(s)=\arg\max_a \widehat V^{\pi_i}(T(s,a)).
\]

Opponents remain on the roster-aware baseline, so this learns an approximate
best response rather than a self-play equilibrium. Complete on-policy drafts
provide Monte Carlo returns: every controlled-team post-pick state in one
trajectory receives the expected weekly score of that trajectory's terminal
roster. A small configurable epsilon supplies action diversity. Each round
uses only fresh labels for the policy currently being evaluated, avoiding
stale targets from previous policies.

Set `SMOKE_TEST=True` in the notebook for a quick two-round validation. The
production run writes resumable datasets and metrics under
`data/policy_iteration/`, checkpoints under `models/policy_iteration/`, and
selects `best_value.pt` using a fixed-seed held-out paired policy benchmark.
Restarting the run skips iterations whose full artifact set is already present.

This is fitted/generalized policy iteration, not exact tabular policy
iteration. Finite Monte Carlo labels, neural approximation, and greedy
maximization can cause regressions. Use the reported paired advantage and
confidence interval—not model test error alone—to select a checkpoint. The
production loop is intended for a GPU-backed Colab session and can take
substantial time depending on simulation counts.

## Organization and extension points

- `players.py`: replaceable synthetic player generation
- `draft.py`: state, snake order, legality, deterministic transitions
- `policies.py`: common draft-policy protocol and baselines
- `state.py`: model-facing fixed-dimensional encoding
- `waivers.py`: weekly replacement protocols and implementations
- `season.py`: outcomes, lineup optimization, and utility functions
- `evaluation.py`: nested Monte Carlo values and seed plans
- `generation.py`: reachable partial-state sampling
- `dataset.py`: reconstructable JSONL and compact NPZ formats
- `scripts/`: generation, evaluation, and benchmark commands
- `tests/`: deterministic behavioral and statistical checks

Future algorithms can evaluate candidate actions through

\[
Q^\pi(s,a)=V^\pi(T(s,a))
\]

while intermediate rewards remain zero. The same value machinery is not tied
to drafts: a future trade can be represented by a deterministic transition
from `state_before` to `state_after`, followed by a matched comparison
\(\Delta V=V(state_{\text{after}})-V(state_{\text{before}})\). Season,
replacement, utility, and common-random-number components are reusable for that
purpose.
