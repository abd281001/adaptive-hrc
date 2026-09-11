# Adaptive-HRC in Overcooked and Burrito

This is a cross-environment replication of the Adaptive-HRC evaluation. The
learner, model settings, paired seeds, baseline training regimes, longitudinal
timing, frozen probes, and replay audits are held fixed. The changed component
is the environment adapter: demonstrations contain completion-checked cooking
task options backed by physical Overcooked/Burrito execution.

## One full evaluation, no profiles

There is exactly one experiment config: `configs/full.json`. It uses:

- the eight Adaptive-HRC paper seeds, taken from `src.evaluation.PAPER_SEEDS`
  rather than restated here: `1337`, `2024`, `7`, `9001`, `31415`, `42`,
  `271828`, and `8675309`;
- homogeneous, heterogeneous, and container-axis holdout scenarios;
- MaxEnt IRL cold/warm steps `100/40`, horizon `45`, grace `50/6`, replay
  capacity `64`, and semantic RMS threshold `0.20`;
- shared Full-system routing, pre-event probes, frozen panels covering all 44
  behaviourally distinct (recipe, preference) pairs, and active-only audits
  every two events;
- the human-first turn-taking protocol (`lead_actor_policy: human_first`);
- the exact Adaptive-HRC roster: `full`, `frozen`, `offline_default`,
  `offline_all`, `unpinned`, `latest`, `fixed`, `no_decay`, `bc`, `ewc`,
  `replay_bc`, and the non-deployable `memory_oracle`; and
- one zero-learning reference, `canonical_order`, which is not part of that
  roster and is held apart from the baseline-parity check.

`frozen` is trained offline on deterministic 50% recipe and preference subsets.
`offline_default` is trained on every selected recipe under its canonical
preference. `offline_all` is trained on every selected recipe under every
preference that changes its ordering, which is the full 44-pair panel: the
most an offline model could be handed before the stream starts. The three
corpora nest. In the holdout, all three frozen controls train on the exact
ordered source progression. The memory oracle is future-aware only for replay
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
  this ladder is intentionally longer, and exceeds 210 episodes for every
  fixed seed in the current catalog.
- Axis holdout: 14 stages and 630 episodes, exactly matching Adaptive-HRC's
  eight-source/six-target timing and 45 episodes per stage. Sources are chosen
  to maximise the number of structurally distinct targets, and any target whose
  role-level task DAG matches its source is labelled
  `holdout_transfer_isomorphic`: transfer across an isomorphic pair is a
  relabelling of the container axis, not a generalisation of it. The
  `holdout_transfer_is_generalisation` requirement therefore demands a
  non-isomorphic target in *every* stratum, and probe sampling draws a second
  recipe from a different structure class so this holds for every fixed seed --
  `burrito_combo` is what makes it satisfiable on the Burrito side.
  Container-first is
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

## What is measured

This is a prediction experiment. The learner, protocol, hyperparameters and
seed grid are held at their Adaptive-HRC values; the environment adapter is the
changed component. So the question the artifacts have to answer is how well the
predictor does in an environment that is not the one it was developed in, and
what it costs there.

### The primary metric is the symbolic one

The headline is pooled `teacher_forced_top_1`, scored on every assist decision
-- robot turns, human turns, and the opening move -- matching
`src.hrc_simulation.simulate_episode`, which records exactly these shadow
predictions. It is primary because it is what
`src.evaluation` reports as its own `primary_prediction_metric`, so the two
environments are compared on one definition. The
preference-discriminating family below is cooking-specific and has no symbolic
counterpart; promoting it left the replication with no figure that could be set
beside the symbolic result at all.

### Prediction inputs cannot depend on the answer

The candidate list handed to the predictor is the structural task-graph
frontier and nothing else. It used to be the physically-legal subset at the
first tick where the *preferred* option became executable, which made the
candidate list a function of the ground truth: competing options that were not
ready yet silently disappeared from the set the predictor was scored on. On one
seed, 670 of 45,641 forced robot decisions were forced only by that waiting.
Readiness is now waited out after the decision, against whichever option is
actually executed, so waiting cannot feed back into scoring.

This also makes candidate lists method-independent. The executed prefix is the
human's ordering under every arm, so the structural frontier at a given step is
identical across arms; the physically-legal subset was not, because two arms
reach the same step with different pot timers. Physical execution measurements
(`task_low_level_ticks`, `passive_wait_ticks`) remain per-arm and are reported
separately in `summary.performance`.

### Every assist decision is scored

A decision an arm could not answer is a miss, not an absence. Accuracy
denominators previously filtered on "the arm emitted a prediction", which
removed the failures of the weakest arms from their own denominators: on seed
1337 that inflated `fixed` by 9.1 points and `frozen` by 3.0, while leaving
`full` untouched, because `full` answered every decision. Availability is now
reported beside the rates instead of subtracted from them, as
`performance.prediction_availability` and
`performance.robot_prediction_availability` with their unanswered counts.
Robot decision counts come from the scheduled turns, not from the answers, and
an episode record whose corrections exceed its robot turns -- or whose robot
turns disagree with the schedule -- raises rather than being aggregated. For the scoring rule there is one
convention, shared with the symbolic evaluator: the ground-truth probability is
floored at `settings.min_probability` (currently `1e-6`, which is what
`src.evaluation` floors at) and `teacher_forced_nll` is the total loss divided
by the total number of scored decisions. Both halves were wrong before. A
uniform-over-candidates fallback for an unanswered decision substitutes a
distribution the predictor never emitted, and on a single-candidate frontier it
scores silence at zero loss -- a perfect score for saying nothing; silence and
"emitted a distribution excluding the truth" are the same event for a scoring
rule, so both are charged the floor. And averaging the per-episode NLL means,
as this previously did, is a macro-average over episodes of unequal length: one
nat over a single decision and one nat over nine pool to 0.2 per decision and
average to 1.0. The floor in force is recorded as
`performance.nll_probability_floor`.

### The opening move is reported on its own

Under `human_first` the opening move is never a robot turn, so no robot-turn
metric can see it -- and it is the most preference-informative decision in
these task graphs. `opening_top_1` and `opening_discriminating_top_1` are
reported per group, per holdout stage, and in the seed-level cells. This is not
a refinement: on the container-axis transfer cell the opening move is
discriminating in **every** episode, which makes it the transfer measurement.

### Two ambiguity denominators, and only one is an accuracy

The physical task graph often leaves a single legal next option. Those
decisions matter for execution and are trivial for prediction, so `robot_top_1`
pools forced turns and stays a diagnostic. Restricting to choice points is
necessary but not sufficient:

- `preference_discriminating_top_1` counts robot turns where the recipe's
  declared preferences *could* disagree. It is a task-intrinsic count of choice
  points and it over-counts badly as an accuracy denominator. Under
  `human_first` the human takes step 0, the widest frontier of the episode, so
  by the robot's turn the surviving preference set is usually a singleton.
  Across the 44 behaviourally distinct (recipe, preference) cells, 99 robot-turn
  decisions carry this flag.
- `prefix_conditioned_top_1` counts only the robot turns where the preferences
  *still consistent with the episode's own prefix* disagree. Of those 99, 17
  survive. This is the denominator that isolates tracking a preference from
  following an ordering the prefix has already determined.

Both are reported, with their denominators, per (arm, scenario, stratum), and
both have teacher-forced variants that additionally score human turns and the
opening move. That last part matters because for several pairs the opening move
is the only discriminating decision and under `human_first` it is never a robot
turn; those pairs are listed in `summary.assistance_unscored_cells`, measured
for prediction and not for assistance. Setting
`lead_actor_policy: counterbalanced` alternates the opening actor across a
recipe's assist exposures and exercises them as robot decisions, at the cost of
departing from the symbolic protocol.

### The zero-learning reference

Roughly half of this catalog's robot turns have one legal option, and the human
resolves the widest frontier of every episode before the robot chooses, so a
fixed rule that never looks at the human scores far above chance. A
Full-versus-baseline margin cannot be read without that number, so
`canonical_order` -- always take the legal option that comes first in
declaration order, no memory, no model, no fit -- is a required arm. It is
excluded from `adaptive_hrc_baseline_parity` because parity is about matching
the deployable symbolic roster and this arm has no symbolic counterpart, and
the `zero_learning_reference` requirement asserts that it completed and that it
never fitted a predictor or consulted a learned component.

### Cost, and metrics that were dropped

`summary.performance`, and the same block per group, reports what the run cost:
episode wall time, mean per-decision prediction latency, fit count, cumulative
fit wall time, p50/p95 per-fit wall time (the blocking wait between two
demonstrations), estimated fit FLOPs, peak dense-array bytes, mean active
replay variants and transitions, pooled `teacher_forced_nll`, invalid
predictions, and task completion. Every column was already recorded per episode
and aggregated nowhere, so the run could not answer a question about the
system's cost.

Three metrics were removed from the summary and groups. `robot_top_k` at k=3
sits on a frontier of at most four options, so it reports a structural ceiling.
`nontrivial_choice_top_1` tracked `preference_discriminating_top_1` to within a
thousandth. The `normalized_human_action_load` family is an affine restatement
of `robot_top_1`, bounded below by the alternation floor, and reads as
saturated for that reason rather than as a result; `corrections_per_robot_decision`
and `correction_free_rate` carry the same information and remain. The raw
per-episode counts behind all three are still written, so any of them can be
recomputed from `episodes.json`.

### Reporting

The unit of replication is one (seed, arm, scenario, environment) cell.
`summary.by_seed` carries `per_cell` at that grain, `seed_means` per (arm,
scenario, environment), and `paired_vs_full` -- the Full-minus-arm difference
paired within seed, with its standard deviation and a Student-t 95% interval.
Grouping by seed alone, as this previously did, averaged all twelve arms into a
single per-seed rate, which is not a quantity anyone can draw an inference
from, and it left the run with no unit in which a Full-versus-arm difference or
an environment-by-method interaction could be tested at all. The pooled figures
weight a longer heterogeneous seed more heavily and are descriptive only.

`summary.holdout_transfer_groups` splits the holdout by the ladder's own
`strategy` label, one row per seed, into `holdout_source_training` (the axis is
absent throughout), `holdout_axis_source_introduction` (it is taught on an
already-known source), and `holdout_axis_target_composition` -- the last split
again by whether the episode's preference *is* the container axis
(`CONTAINER_FIRST_PREFERENCE`) and whether it is that pair's first exposure.
Exactly one cell is the transfer measurement, flagged
`is_transfer_measurement`, and `summary.holdout_transfer_paired_vs_full`
carries the Full-minus-arm contrast on it, paired within seed.

These are unadjusted pairwise intervals, and an interval containing zero means
the difference is unresolved at eight seeds -- not that the two methods are
equivalent. In the rescoring of the saved run, 17 of the 72 deployable
teacher-forced contrasts contain zero, concentrated in the homogeneous scenario
and the compatibility stratum; do not describe those as ties.

The split has to be built this way. Keying on `holdout_target` plus
`exposure_after_change <= 1`, as an earlier version did, labelled ordinary
source training and acquisitions of unrelated preferences as transfer: on the
saved eight-seed run that called 1,500 Overcooked and 257 native-Burrito
episodes a first transfer exposure, against **24 and 12** actual ones. Later
target exposures outnumber first ones by roughly fifty to one, so a pooled
holdout figure is almost entirely adaptation reported as transfer.

At two dozen first exposures per environment the seed is the only honest unit,
and the column to read is `opening_discriminating_top_1`: on a first axis-target
exposure the opening move is discriminating in every episode and is always the
human's, so `prefix_conditioned_robot_decisions` is zero there and the
robot-turn metrics are empty by construction.

Accuracy is also split by whether the semantic-distance fallback fired (`semantic_fallback_discriminating_top_1`
against `own_model_discriminating_top_1`), because identity-masked features
place every structurally identical recipe at distance zero, so the fallback
always fires between them and its value must be measured rather than assumed.
Results additionally report the single-legal-action fraction, corrections per
robot decision, and mutation-free pre-event probe accuracy under both ambiguity
denominators. Never cite the earlier smoke run's approximately 99% overall
accuracy as adaptation performance.

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
seed/scenario/baseline cell is first written atomically under
`checkpoints/<arm>/<seed>__<scenario>.json`, with a per-arm roll-up at
`checkpoints/<arm>/summary.json`, and `manifest.json` tracks completed cells
during the run. A cell that raises stops at the failing episode; its rows are
marked `cell_complete: false`, excluded from every pooled statistic, and
listed in `summary.excluded_incomplete_cells`.

Cells are scheduled arm-major: one arm completes the whole seed/scenario grid
before the next starts, and each arm keeps its cells in its own folder. Cooking
cells are independent -- unlike the symbolic evaluation there is no shared
route to publish -- so the order is purely about where results land and what
has to be recomputed. Deleting `checkpoints/<arm>/` and resuming re-runs that
arm alone; every other arm's cells are reused from disk.

Each checkpoint records the result schema version, the config digest, and a
digest of the wrapper source *and* the core learner modules under `src/`
(`adaptive_agent`, `baselines`, `models`, `memory`, and the rest of
`_CORE_SOURCE_MODULES`) -- hashing only the wrapper left a checkpoint valid
across an edit to the agent or the memory policy, which is where everything the
learner does actually lives. A resume reuses a cell only if all three match,
the cell names the seed, scenario and arm being asked for, it is failure-free,
and its episodes cover the planned schedule exactly once by `event_index`. A
completion flag is a claim, not evidence: a checkpoint holding one of its 630
planned episodes with the flag set was accepted as a finished cell. Every check fails
closed -- a rejected checkpoint just re-runs its cell. Both gaps mattered: the
config digest alone did not notice an edit to the evaluator or the protocol,
which changes what a cell means without changing the config; and an incomplete
cell used to be absorbed as though it had finished, so a cell that died partway
through could never be recovered by resuming and stayed permanently excluded.

## Known-task replication, and what that constrains

The learner is told which recipe it is about to perform. The cooking state
carries its recipe code in slot zero and the engineered features carry a recipe
one-hot, from the empty prefix onward. The symbolic environment does not do
this: every symbolic episode starts from the same state, with all items in
storage, so its 470 predicates do not reveal which recipe is coming until
actions begin.

That is a defensible design for a replication whose question is preference
prediction rather than task recognition, and it is the condition these results
were produced under. It is not a free choice, and two claims have to be read
against it. `open_set_recipe_separation` asserts that the learner allocates one
identity per recipe, which is a much weaker guarantee when identity is an
input. And "the same information" is the wrong description of the difference:
the symbolic learner has to infer the task from what it observes, and this one
does not. State it as a known-task setting, or drop the one-hot and evaluate
recipe recognition as a separate result -- do not describe it as an easier
encoding of the same inputs.

## The container-axis transfer result, and what it does not show

On the first exposure of container-first on a held-out target -- 36 episodes
across the eight seeds, 24 Overcooked and 12 native Burrito -- the opening move
is preference-discriminating in every episode, is always the human's turn, and
`full` predicts it correctly **0/36**. From the second exposure onward the same
pairs run at 0.94-0.98. During the source-introduction stage `full` is at
0.64-0.73 on the axis. So the system learns the axis on a source, does not
carry it to a target, and acquires it immediately once demonstrated there. That
is the finding, and it survives the known-task and
original-candidate-list qualifications.

The non-deployable `memory_oracle` answers the same decision 29/36. That
establishes sensitivity to future-informed replay selection. It does **not**
establish that the decision was equally answerable from `full`'s deployed
memory, because the oracle's advantage is partly *removal*: it prunes variants
that will never recur, using information `full` does not have.

It is specifically not evidence that `full` forgot the source axis variant, and
the diagnostic below shows it did not. At seed 1337's transfer episode
`full` holds seven active variants at minimum weight 1.0 -- nothing decayed,
and only one episode intervenes since the previous axis episode, below the
six-demonstration minimum grace. What differs is that the oracle's prediction
there uses the semantic-distance fallback and `full`'s does not. There is a
mechanism in the code that would produce exactly that:
`MaxEntIrl.action_distribution` returns an exact-state policy and floors every
*unlearned* candidate as soon as any candidate has a learned Q-value at that
state, so one retained variant that visited the opening state is enough to
prevent the fallback from ever being consulted. Retained obsolete
target-specific knowledge suppressing the fallback and genuine forgetting are
different failures with different fixes, and the oracle comparison cannot
separate them.

`python -m adaptive_hrc_burrito.mechanism_probe` is the bounded diagnostic that
separates the explanations. It replays one seed's holdout to the first
container-first exposure on a held-out target, dumps the active set with
weights and pins, and re-predicts that single opening after removing (a) only
the obsolete target-specific variants, (b) only the source axis variants, and
(c) both, refitting each time. Run over all eight seeds in each stratum --
`--stratum burrito_native` is needed for the Burrito side, because the earliest
first exposure in every paper seed's schedule is an Overcooked target -- it
gives 16 seed-decisions:

| condition | gate reaches fallback | top-1 correct | mean p(container-first) |
| --- | --- | --- | --- |
| unmodified | 0/16 | 0/16 | **exactly 0.0000, all 16** |
| minus obsolete target variants | **16/16** | 7/16 | 0.41-0.45 |
| minus source axis variants | 0/16 | 0/16 | exactly 0.0000, all 16 |

Removing the obsolete target variant lifts p(container-first) by
**+0.4528, 95% CI [+0.345, +0.561]** on the Overcooked targets and
**+0.4107, [+0.329, +0.493]** on the native Burrito ones.

The chain is the same in all 16 and every link is checkable in the artifacts.
The latest-preference pin holds one variant per recipe; the axis is absent from
source training, so the target's own axis variant is never resident before its
first exposure (0/16) and the pinned variant is by construction the target's
most recent *other* preference (pinned in 16/16, weight 1.0). That variant has
a learned action at the opening state, so
`MaxEntIrl.action_distribution` returns an exact-state policy and assigns every
unlearned candidate -- including container-first -- probability exactly zero,
and the semantic fallback is never attempted. Removing it flips the gate in
16/16 and the retained source axis knowledge then supplies a real estimate.

So this is fallback suppression, not forgetting: the source axis variants are
resident and pinned throughout, and removing them on their own changes nothing,
because under an exact-state policy they were already inert.

Three qualifications. Suppression is not the whole story -- with the fallback
reachable the canonical action still wins the argmax in 9 of 16, so the
fallback's *ranking* is a second factor. The "minus both" condition does not
measure the source axis contribution cleanly: in 5 of 16 it returns
`fallback_rejected` or `unavailable`, meaning no estimate was produced at all,
so that contribution is unresolved rather than small. And this is one scenario,
one arm, on the original ground-truth-conditioned candidate lists -- a
mechanism diagnostic, not a corrected result.

What it does license is a specific claim with a specific target: the pin, one
of the proposed components, is what blocks this transfer, and the fix is not to
retain more but to make the exact-state gate condition on whether the resident
variants actually cover the current candidate set rather than on there being at
least one learned action. Testing that attribution against the roster needs a
`full_no_pin` arm, which the deployable `unpinned` baseline is not -- it drops
the pin, the semantic fallback and the latent residual together, and it scores
1/36 against Full's 0/36 on this cell, which resolves nothing. That arm is 24
cells.

## Interpretation boundary

This setup supports a matched cross-domain replication, not pooled exchangeable
samples. Report symbolic Adaptive-HRC, Overcooked, native Burrito, and Burrito
compatibility strata separately, then test the pre-specified
environment-by-method interaction. Physical task-option legality is necessarily
environment-specific and is explicitly quantified through the forced-choice
fraction.
