# Adaptive-HRC in Overcooked and Burrito

This is a cross-environment replication of the Adaptive-HRC evaluation. The
learner, model settings, paired seeds, baseline training regimes, longitudinal
timing, frozen probes, and replay audits are held fixed. The changed component
is the environment adapter: demonstrations contain completion-checked cooking
task options backed by physical Overcooked/Burrito execution.

## One full evaluation, no profiles

There is exactly one experiment config: `configs/full.json`. It uses:

- seeds `1337`, `2024`, `7`, `9001`, and `31415`;
- homogeneous, heterogeneous, and container-axis holdout scenarios;
- MaxEnt IRL cold/warm steps `100/40`, horizon `45`, grace `50/6`, replay
  capacity `64`, and semantic RMS threshold `0.20`;
- shared Full-system routing, pre-event probes, frozen panels covering all 44
  behaviourally distinct (recipe, preference) pairs, and active-only audits
  every two events;
- the human-first turn-taking protocol (`lead_actor_policy: human_first`);
- the exact Adaptive-HRC roster: `full`, `frozen`, `offline_default`,
  `unpinned`, `latest`, `fixed`, `no_decay`, `bc`, `ewc`, `replay_bc`, and the
  non-deployable `memory_oracle`.

`frozen` is trained offline on deterministic 50% recipe and preference subsets.
`offline_default` is trained on every selected recipe under its canonical
preference. In the holdout, both frozen controls train on the exact ordered
source progression. The memory oracle is future-aware only for replay
retention; it is not given the correct next action.

## Nontrivial task catalog

Single-ingredient orders are excluded. The catalog has 12 tasks:

- seven Overcooked onion/tomato multisets of size two or three;
- steak-onion and chicken-onion compatibility tasks;
- native steak-burrito and mushroom-burrito tasks;
- `burrito_combo`, a native two-order task that serves one steak and one
  mushroom burrito in an 18-step episode.

Each task has two to six behaviourally distinct order preferences, and every
one of them is verified to realize a *distinct physical ordering* -- not merely
a distinct abstract one. That distinction matters: a preference selects over the
task-graph frontier and the human waits for its choice, so "plate the protein
first" means waiting for the protein. Choosing greedily from whatever happened to
be cooked collapsed all three assembly preferences into one behaviour, because
readiness timing rather than preference decided the order. Native
burrito recipes decompose plate assembly into `PLATE_RICE`, `PLATE_TORTILLA` and
`PLATE_<protein>`, which the pinned environment accepts in any of the six orders;
standard Overcooked exposes `START_COOKING_SOUP` as its own option. Only
preferences that change some recipe's ordering are declared, and `catalog.py`
refuses to import if a declared preference is behaviourally inert.

The compatibility label is retained because the pinned Burrito interaction
handler does not natively complete the two legacy onion dishes. Their missing
station transitions *and* their cook durations are restored by the wrapper and
reported through `compatibility_calls`. Every record carries a three-valued
`environment` stratum -- `overcooked`, `burrito_native`, `burrito_compat` -- and
the summary groups on it, so compatibility results cannot pool with native
execution.

## Evaluation ladders

All schedules use three demonstrations per recurrence-gap unit and one human
observation on a recipe's first natural occurrence. Every later occurrence is
assist mode. A preference shift is acquired on its first ordinary occurrence,
committed and retrained at episode end, and evaluated only after intervening
episodes; generated schedules enforce at least three intervening episodes.

- Homogeneous: seven macro phases and exactly 210 episodes. Every participating
  recipe receives the same ordered abstract strategy set in a phase. The
  abstract support/container strategy is grounded to the effective ordering of
  each recipe.
- Heterogeneous: seven macro phases. Each selected recipe independently retains,
  adds, removes, or swaps preferences and may carry one to three active
  preferences. Macro climbs are serialized using the same decay rule and
  bounded heavy-tailed gaps (mean 15, range 3–40) as Adaptive-HRC. Consequently
  this ladder is intentionally longer: 1,080–1,395 episodes for the five fixed
  seeds in the current catalog.
- Axis holdout: 14 stages and 630 episodes, exactly matching Adaptive-HRC's
  eight-source/six-target timing and 45 episodes per stage. Sources are chosen
  to maximise the number of structurally distinct targets, and any target whose
  role-level task DAG matches its source is labelled
  `holdout_transfer_isomorphic`: transfer across an isomorphic pair is a
  relabelling of the container axis, not a generalisation of it. With the
  `holdout_transfer_is_generalisation` requires a non-isomorphic target in
  *every* stratum, and probe sampling draws a second recipe from a different
  structure class so this holds for all five fixed seeds -- `burrito_combo` is
  what makes it satisfiable on the Burrito side. Container-first is
  absent throughout source training. It is introduced on one known Overcooked
  and one known native Burrito source, then applied to already-known held-out
  recipes from both environments. With fewer cooking preferences, the eight
  source stages sustain the available non-container orderings rather than
  inventing synthetic preference labels.

Cooking has two to six effective preferences per recipe, versus the
larger symbolic preference set. An add/remove/swap request can therefore become
impossible. The generator conditions the 0.40/0.20/0.20/0.20 lifecycle draw on
the feasible operations and records the feasible set and realized operation
counts. This is the one schedule-level domain constraint; it is not hidden by
retrying schedules until an easy draw succeeds.

## Accuracy and the earlier 99% result

The physical task graph often leaves only one legal next option. Those decisions
are useful for safety and execution but trivial for prediction. Overall Top-1
therefore remains a diagnostic only. As in Adaptive-HRC, the primary workload
metric is normalized human action load; because the protocol is human-first
strict alternation, even a perfect robot performs `ceil(n/2)` of an n-step
recipe, so the summary also reports `normalized_human_action_load_floor` and the
floor-rescaled `human_action_load_excess`. The primary accuracy metric is pooled
`preference_discriminating_top_1`, computed only on robot turns where the legal
frontier contains a genuine preference-dependent choice. Every assist decision
is additionally scored as a teacher-forced shadow prediction -- human turns and
the opening move included, matching `src.hrc_simulation.simulate_episode` --
and pooled as `teacher_forced_preference_discriminating_top_1`. This matters
because for several recipe/preference pairs the opening move is the *only*
preference-discriminating decision, and under `human_first` it is never a robot
turn. Those pairs are listed in `summary.assistance_unscored_cells`: they are
measured for prediction, not for assistance. Setting
`lead_actor_policy: counterbalanced` alternates the opening actor across a
recipe's assist exposures and exercises them as robot decisions, at the cost of
departing from the symbolic protocol.

Seeds are the unit of replication, so `summary.by_seed` carries per-seed rates
and their macro-average; the pooled figures weight a 1,395-episode heterogeneous
seed more heavily than a 1,080-episode one and are descriptive only. Accuracy is
also split by whether the semantic-distance fallback fired
(`semantic_fallback_discriminating_top_1` against
`own_model_discriminating_top_1`), because identity-masked features place every
structurally identical recipe at distance zero, so the fallback always fires
between them and its value must be measured rather than assumed. Results also report
the single-legal-action fraction, nontrivial-choice Top-1, corrections per robot
decision, and mutation-free pre-event probe accuracy. Never cite the earlier
smoke run's approximately 99% overall accuracy as adaptation performance.

## Run

From the repository root:

```bash
./burrito/bootstrap.sh
./burrito/.venv/bin/python -m adaptive_hrc_burrito verify
./burrito/.venv/bin/python -m unittest discover -s burrito/tests -v

# Full publication evaluation. Git provenance and dirty paths are recorded.
./hrc cooking

# Equivalent direct wrapper invocation.
./burrito/.venv/bin/python -m adaptive_hrc_burrito validate \
  --config burrito/configs/full.json

# Resume an interrupted run: completed seed/scenario/arm cells are replayed
# from checkpoints/ and only the missing cells are executed.
./burrito/.venv/bin/python -m adaptive_hrc_burrito validate \
  --config burrito/configs/full.json --resume <run_dir>

# Cells are independent, so they run in parallel processes (default: one per
# CPU).  Processes, not threads: the pinned environment keeps recipe
# configuration, a shared mutable completion list and the planner's RNG in
# process-global state, so threads would serialise on the executor locks that
# guard it and would corrupt each other's planner stream.
./burrito/.venv/bin/python -m adaptive_hrc_burrito validate \
  --config burrito/configs/full.json --workers 8
```

`HRC.ipynb` invokes `./hrc cooking` between the standard Adaptive-HRC run and
the ablations. Set `HRC_COOKING_WORKERS` before starting Jupyter to cap its
process count. Set `HRC_COOKING_RESUME` to an interrupted cooking run directory
to make that notebook stage reuse completed cells. Both launcher and direct
wrapper invocations use the config-owned `eval_results/cooking/` output root.

Artefacts are sorted before being written, so a run is byte-identical whatever
the worker count; only wall-clock fields differ.

CI runs a bounded protocol check and unit/integration tests; it does not create
a reduced evaluation profile. A full run writes `manifest.json`, `episodes.json`,
`probes.json`, `audits.json`, `schedules.json`, `failures.json`, `summary.json`,
and `validation.json` under an immutable run directory. Every completed
seed/scenario/baseline cell is first written atomically under `checkpoints/`,
and `manifest.json` tracks completed cells during the run. A cell that raises
stops at the failing episode; its rows are marked `cell_complete: false`,
excluded from every pooled statistic, and listed in
`summary.excluded_incomplete_cells`.

## Interpretation boundary

This setup supports a matched cross-domain replication, not pooled exchangeable
samples. Report symbolic Adaptive-HRC, Overcooked, native Burrito, and Burrito
compatibility strata separately, then test the pre-specified
environment-by-method interaction. Physical task-option legality is necessarily
environment-specific and is explicitly quantified through the forced-choice
fraction.
