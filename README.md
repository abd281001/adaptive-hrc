# Adaptive Preference Memory for Human–Robot Collaboration

This repository contains the simulation and evaluation code for longitudinal
human workflow adaptation in a symbolic kitchen. A simulated user provides
demonstrations over time; the system predicts robot actions, retains active
recipe-preference variants, decays obsolete variants, and retrains online.

The repository is organized around one executable notebook, `HRC.ipynb`. The
notebook runs the standard evaluation, the Overcooked/Burrito replication, and
the reviewer-facing ablations through their application runners. It contains
no model, evaluator, orchestration, or result-schema implementation and is
committed without execution outputs.

## Setup

The core project has two environments and the pinned cooking replication has
one isolated environment. There is no separate smoke or test environment.

| Workflow | Environment | Python | Dependency file |
| --- | --- | --- | --- |
| `HRC.ipynb`, normal evaluation, ablations, and tests | `venv` | 3.13 | `requirements.txt` |
| Overcooked/Burrito evaluation | `burrito/.venv` | 3.10 | `burrito/requirements-runtime.txt` |
| Optional frozen Qwen LLM baseline only | `.venv-llm` | 3.12 | `requirements-llm.txt` |

The `./hrc` launcher always selects the correct installed interpreter, so
activation is not required. Keep these dependency sets isolated.

### Full notebook workflow and tests

From a fresh clone:

```bash
git clone https://github.com/abd281001/adaptive-hrc.git
cd adaptive-hrc
./hrc setup
./hrc kernel
./hrc test
./burrito/bootstrap.sh
```

Open `HRC.ipynb` in a Jupyter-compatible frontend, select the
`Python (adaptive-hrc)` kernel, and run the cells from the repository root.
Running all cells performs these three stages in order:

1. the standard evaluation: three scenarios times eight paired seeds, written
   as one new directory under `eval_results/runs/`;
2. the validated Overcooked/Burrito evaluation, executed through its isolated
   Python 3.10 runtime and written under `eval_results/cooking/`; and
3. the matcher, routing, latent-strategy, memory, representation, component,
   and retention ablations, contributed as extra arms to the
   standard run directory.

The three scenarios are stored inside one standard run directory; they are not
three separate top-level runs. By default, every notebook execution creates new
immutable run directories rather than overwriting an earlier execution. An
explicit cooking resume reuses only verified completed cell checkpoints.

Generated artifacts are written beneath `eval_results/`. The full workflow is
computationally expensive. The standard evaluation has a 10-process cap (set by memory, not core count;
see `DEFAULT_EVALUATION_WORKER_CAP`) and admits only complete seed cohorts:
with the default eight seeds, one scenario runs at a time using 8 workers, so
the three scenarios run in sequence rather than in parallel. Every evaluation
worker is pinned to the host's performance cores only (P-cores on a hybrid
Intel part, detected from `/sys/devices/cpu_core/cpus`; a no-op on a uniform
part), so cross-baseline wall-clock comparisons are not confounded by a
worker landing on a slower efficiency core. With five seeds, all three
scenarios instead run in parallel using 15 workers. Set `HRC_WORKERS` before
starting Jupyter to request a lower cap. Ablation arms run one at a time,
each spreading its own scenario-seed grid across 8 worker processes by default; `HRC_ABLATION_WORKERS` may raise
that to at most one worker per scenario-seed job. Running the arms in turn
keeps unrelated arms out of contention for the same performance cores, which
the per-arm wall-clock metrics depend on. Every longitudinal group uses the
same eight paired seeds as the standard evaluation; the matcher group
generates one stress dataset from a single generation seed and has no
paired-seed grid. The cooking config uses all
available CPUs by
default; `HRC_COOKING_WORKERS` caps that count. Set
`HRC_COOKING_RESUME` to an interrupted cooking run directory to reuse its
completed seed/scenario/arm checkpoints.

The launcher routes each workflow to its owning environment:

```bash
./hrc test
./hrc run
./hrc cooking
./hrc ablation
./hrc ablation --groups matcher
```

`./hrc run` runs only the standard three-scenario, five-seed evaluation;
`./hrc cooking` runs the sole full cooking config; and `./hrc ablation` runs
every ablation group. `--groups` takes a comma list of `matcher`, `components`,
`retention`, `latent`, `memory`, `representation` or `routing`. Additional
evaluator arguments can be appended to the runners.

The Overcooked/Burrito replication uses a separate pinned Python 3.10
environment under `burrito/.venv` because it is an external replication with
incompatible dependencies. `./hrc cooking` invokes that interpreter but does
not create it or install its dependencies, so it cannot perturb either core
environment. See
[`burrito/README.md`](burrito/README.md) for its isolated setup. It includes a
physical task-option adapter and the same one-observation-then-assist correction
protocol used by the symbolic evaluation. Because it is a replication, it
reports `teacher_forced_top_1` as its headline -- the same
`primary_prediction_metric` this evaluation reports -- so the two environments
are compared on one definition, and it runs a zero-learning `canonical_order`
arm beside the deployable roster because roughly half of its robot turns have a
single legal task option. Its sole publication experiment is
`burrito/configs/full.json`; reduced CI/validation/publication profiles were
removed. `HRC.ipynb` remains a thin ordered launcher: all cooking settings,
validation, checkpoints, and reporting stay in the wrapper.

### Stretch 3 real-robot interface

The repository also includes an operator-mediated Stretch 3 interface for a
reduced marker-object task. Both learner and hardware processes run locally on
the robot, but the Stretch SDK remains isolated behind a loopback bridge so it
does not alter the normal `venv`. Start with a motion-disabled dry run:

```bash
./hrc robot-bridge
./hrc robot-ui --hardware-url http://127.0.0.1:9100
```

The operator UI requires explicit approval before every robot action. A
hardware transition reaches replay memory only after completion is confirmed.
The Stretch/D405/ArUco motion runtime is self-contained under
`src/real_robot/stretch_runtime`; deployment does not require the prior lab
demo directory.
The sample station geometry is intentionally uncalibrated and cannot enable
motion. See [`src/real_robot/README.md`](src/real_robot/README.md) for robot
directory locations, migration of existing local data, and dry-run commands.
Use `./hrc robot-doctor` for a read-only deployment preflight and
`./hrc robot-report RUN_DIRECTORY` for an integrity/outcome summary. Live UI
startup requires the explicit `--require-motion` handshake, a matching full
configuration digest, and a ready motion-enabled bridge.
Publication collection additionally requires `--publication-run` and
`--schedule PARTICIPANT.json`; its report validates the frozen schedule, calibration and
runtime artifacts, per-episode metadata/postconditions, and shadow baselines.
Uncalibrated supervised hardware trials use the isolated
`./hrc robot-calibration-probe` path; a calibration-mode bridge is rejected by
the experiment UI.

## Experiment design

- Three scenarios cover homogeneous deployment, heterogeneous deployment, and
  a controlled cleanup-axis compositional holdout. Each paper seed uses a
  20-recipe user panel.
- A recipe's first stream event is its sole observation-mode demonstration.
  Later exposures, preference changes, reintroductions, and climb events run
  in assist mode under the shared evaluation schedule.
- Lifecycle probabilities are retain/add/remove/swap =
  0.40/0.20/0.20/0.20. Reintroduction is a 0.05 conditional subtype of swap.
- Homogeneous runs use seven global climb/settle phases and 210
  demonstrations. Heterogeneous durations are generated from bounded,
  heavy-tailed recurrence gaps with mean 15 and range 3–40.
- The controlled holdout trains without the `when_free` cleanup value,
  introduces it through `cleanup_when_free`, and evaluates unseen
  compositions.
- The memory oracle is explicitly non-deployable. It follows the realized
  interaction schedule while using future recurrence information only for its
  retention decisions. It matches Full exactly on a pair's first exposure,
  because nothing can be retained or pruned before a pair is learned, and
  thereafter reports its own future-filtered outcome including when that is
  worse than Full. It is a reference curve, not a per-event upper envelope.
- Teacher-forced and live Top-1 are reported separately. The primary workload
  measure is normalized human action load: scheduled human actions plus
  corrections, divided by recipe steps. Note that the alternating protocol
  gives the human every other turn, so this measure is bounded below by 0.5.
- Retrain cost is reported both cumulatively and as a per-fit latency
  distribution. The p95 fit wall time is the blocking wait between two
  demonstrations and is reported per phase alongside the cumulative totals.
- Removing a variant from replay warm-starts the next fit and does not count
  toward the cold-restart threshold: a removal shrinks the training set, so the
  incumbent weights stay valid for the surviving subset. Only additions
  advance that threshold.

- Three offline-pretrained controls share `FrozenAgent`: they train before the
  stream starts, lock, and never update again. They differ only in the corpus.
  `frozen` takes a deterministic 50% of the seed's recipes crossed with 50% of
  its scheduled preferences; `offline_default` takes every one of the seed's
  recipes under the default preference only; `offline_all` takes every one of
  the seed's recipes under every behaviourally distinct preset, which is that
  seed's complete frozen-probe panel. The three corpora nest, so the roster
  spans the offline-coverage axis from a half subset to the whole panel.
  Recipes are always restricted to the seed's own 20-of-30 draw, because the
  other ten never appear in the stream and no other arm could have seen them.
  Preferences are not restricted that way: the candidate preference set is
  identical for every seed, so `offline_all` trains on the full library even
  where a homogeneous seed only ever schedules two of them. In the holdout all
  three instead train on the exact ordered source progression, which is what
  keeps the held-out cleanup value unseen.

The fixed paired seeds are `1337`, `2024`, `7`, `9001`, `31415`, `42`,
`271828`, and `8675309`.
Experiment manifests record the configuration, Git commit, dirty-tree state,
runtime package versions, and completion status. A resumed run must match its
recorded configuration and experiment label.

## Ablations

An ablation is an arm plus a contrast. There is no separate ablation runner
and no separate ablation output: arms are run by the evaluation's own per-cell
runner into the same run directory, so `baselines/<arm>/` holds the component
arms beside the deployable roster. An arm named by several groups -- or shared
with the roster -- is therefore computed once and read by every contrast that
names it. `src/ablations.py` declares three tables and nothing else executes
that is not in them:

- `ARMS`: 24 diagnostic conditions, each an agent plus settings overrides plus
  a route. The deployable roster is referenced by name, never redeclared; a
  roster cell that is missing is an error telling you to finish `./hrc run`
  rather than a silent re-run.
- `CONTRASTS`: every planned comparison, with one primary per group. Deltas are
  signed so positive always favours the treatment; comparisons whose arms do
  not carry identical components are marked `descriptive` and are not evidence
  for their name.
- `INVARIANTS`: what each group's contrasts depend on -- which component an arm
  moved, which arms must agree on a memory policy -- read back from the
  recorded state of the run rather than from the declared configuration.

Import-time validation rejects a table that cannot mean what it says: a
duplicate arm name, two arms that are the same condition, a contrast naming an
arm that does not exist, a metric declaring two directions, or a group with no
primary contrast.

Collapsing the six longitudinal suites this way removed 480 of 1,344 stream
runs per full grid (36%). The duplicates were `full` (declared by four suites
and the roster), the joint component removal (identical to the deployable
`unpinned`), the MaxEnt-only latent arm (identical to the no-latent-residual
component arm), the full-history cloner (identical to the 2x2's BC cell), and
the routing suite's reference and shared arms, which reproduced the standard
run exactly. Routing now declares only the three local arms that can differ:
a method that never retires a variant cannot route differently, so its local
run was provably its shared run.

The matcher stress suite is the one group that runs no deployment stream. It
generates synthetic identification cases and scores several matchers offline,
and it is kept in its own section of the file.

### Attribution suites

The deployable baseline roster is a system-level comparison: every memory
baseline is built through `_without_proposed_components`, so it drops the
latest pin, the semantic fallback and the latent residual together and a
Full-versus-baseline margin cannot be assigned to any single mechanism. The
component and retention groups exist to make those assignments, and both hold
everything except one declared factor at Full's configuration.

The `components` group removes one predictor-support component at a time
(`full_no_pin`, `full_no_semantic_fallback`, `full_no_latent_residual`) and
then all three together. The joint removal is settings-identical to the
deployable `unpinned` baseline, so that roster arm *is* the joint condition
and is read rather than re-run under a second name.

The `retention` group moves one memory mechanism at a time under Full's predictor
support. `constant_grace` is the control the roster's `fixed` arm is often
mistaken for: `ReplayMemory.step` gates its grace check on the adaptive policy,
so `fixed` has no grace period at all *and* uses a different post-grace
decrement, changing three things at once. `constant_grace` keeps the grace
period, the decrement, the pin and every component and replaces only the
per-pair horizon with one constant, which is what isolates estimating horizons
from recurrence. `shuffled_horizon` additionally holds the horizon distribution
fixed and destroys only its assignment to pairs, separating the retention
policy from the amount it happens to retain. `recent_set_pin` widens protection
from one variant per recipe to every recently demonstrated sibling, which is
the diagnostic for the concurrently-active-preference regime.

The memory factorial's MaxEnt retain-all cell is `full` with its retention
policy switched off, not the `no_decay` baseline; using `no_decay` moved the
two semantic components as well and left both the MaxEnt simple effect and the
interaction term confounded.

Every ablation arm is a `Settings` override applied to the `full` agent, and
each new field defaults to the deployed configuration, so the standard
evaluation is unaffected and does not need to be re-run.

## Optional language-model baseline

The frozen in-context baseline is implemented in `src/llm_baseline.py` and is
intentionally excluded from `HRC.ipynb`. It uses one separate GPU environment,
`.venv-llm`:

```bash
./hrc setup-llm
./hrc doctor
```

This arm is an ablation of the predictor head, not a peer of the other
comparison baselines. Alone among the entries in `BASELINE_AGENTS`, it keeps
Full's memory policy -- adaptive decay with the latest-preference pin -- and
replaces only the MaxEnt head with the frozen LLM. `bc`, `ewc`, `replay_bc`, and
the decay controls all run without those components, so a Full-versus-LLM gap
isolates the predictor while a Full-versus-`bc` gap does not.

It is given the same experimental inputs as the other arms: the same plan, seed
and realized observe/assist schedule; the same active replay demonstrations,
their retention weights and their action sequences; and the same shared
state-only action mask. It is given no recipe label, preference label or goal
predicate.

One input is not matched, and it is a hardware limit rather than a design
choice. The numeric arms train on the `(state, action)` pairs behind those
demonstrations, and this arm can be given the same states -- as named
predicates plus a per-step delta, under `--llm-context-encoding state_delta` --
but that prompt is 2.7 times larger and on a 12GiB card shared with a display
it fits only about ten active demonstrations, well short of a full scenario. The
runner therefore defaults to `action_only`, which completes a run and withholds
the per-step state. Whether a run had it is recorded per turn in
`llm_context_encoding` and once per run in `context_encoding_policy`. Taking the
display off the card restores the parity; see the VRAM section below.

Two measurements are not cross-arm comparable and are labeled as such in the
manifest. Fit FLOPs are zero for this arm because its "training" is prompt
assembly, so compare `llm_inference_wall_s` (measured GPU time) and
`llm_uncached_inference_wall_s` (the same work without prompt memoization)
instead. Its probabilities are a softmax over length-normalized action
log-likelihoods with no fitted temperature, so `teacher_forced_mean_nll` mixes
calibration with accuracy; the top-1 and top-k rates do not.

Run `./hrc doctor` after both core environments have been installed. It checks
the normal environment and the pinned LLM runtime without starting an
experiment. The launcher configures the GPU and thread defaults and
automatically resolves the cached Qwen snapshot. The baseline remains separate
from the normal notebook experiment. The default LLM runner pairs Full and the
LLM on the same full shared realized schedule used by the main experiment. Its
only sampling difference is that the expensive LLM evaluation uses seed `1337`
rather than all eight seeds.

On the RTX 5070/SM120 host, the checkpoint's original NF4 codes and scales are
decoded with PyTorch tensor operations. The unstable bitsandbytes native 4-bit
inference entrypoints are disabled after model loading and fail closed if they
are reached. This changes the inference implementation, not the checkpoint,
prompts, candidate scores, plans, seeds, routing, or information available to
the baseline.

### VRAM budget, and why the desktop matters

On a single-GPU desktop this baseline shares the card with the display server.
That sharing is not a performance question: a GPU that cannot serve the
compositor hangs the whole session, with the display frozen and input dead, and
nothing is written to the journal afterwards, so no Xid or OOM record survives
to explain it. Because the prompt grows as replay memory grows, the failure
arrives partway through a long run rather than at startup.

The baseline therefore caps its own VRAM with
`torch.cuda.set_per_process_memory_fraction` before loading any weights, and
sizes the prompt budget from what is measurably left. The cap is the total
device memory minus what other GPU clients already hold, minus
`--llm-vram-headroom-gib` for their spikes. An over-budget prompt raises
`PromptTooLongError` instead of taking the desktop down; the agent responds by
shedding the prompt's per-step state annotation, which keeps every retained
demonstration.

Measured on the RTX 5070 (11.50GiB usable) with a GNOME session on the same
card:

| | |
| --- | --- |
| Other GPU clients (Xorg, shell, browser) | 0.96 GiB |
| Model weights, NF4 | 7.23 GiB, of which 1.16 embeddings, 1.16 untied `lm_head`, 1.77 layers left unquantized in bf16 |
| Key/value cache | 144 KiB per prompt token |
| Peak prefill cost | 240 KiB per token plus 96 MiB unchunked; 153 KiB plus 193 MiB at a 1024-token chunk |
| Allocator slack held back | 384 MiB |
| Prompt budget, 1.25 GiB headroom | 6624 tokens unchunked, 9914 chunked |

Peak memory is affine in prompt length rather than proportional, because a
prefill holds the cache for every token plus intermediates for the positions in
flight. The budget is the tighter of the allocator cap less the resident
weights and device-free memory less the reserve: device-free memory alone does
not know this process is capped, and using it reported limits that then failed
to allocate.

A full 210-demonstration seed can outgrow the budget while a desktop is
running; the startup log prints the budget and warns when it is below the
requested window. Seed resume is coarse, so a stop late in a seed re-runs that
seed rather than continuing from the last event.

#### Fitting a full run with the desktop still attached

The LLM runner sets two of these for itself, because they are what make a full
scenario fit as a single condition on a GPU that also drives a display:
`--llm-prefill-chunk-tokens 1024` and `--llm-context-encoding action_only`.
Passing either explicitly overrides the default, and the manifest records
whichever values were used. Measured on this host with a GNOME session running,
where "demos" is how many active demonstrations the budget holds at roughly 350
prompt tokens per demonstration without the state annotation and 945 with it:

| Configuration | Prompt budget | Demos, action-only | Demos, annotated |
| --- | --- | --- | --- |
| `--llm-prefill-chunk-tokens 0` | 6624 | 18 | 7 |
| runner default (chunk 1024) | 9914 | 28 | 10 |
| the same, `--llm-vram-headroom-gib 0.75` | 13287 | 37 | 14 |

`--llm-prefill-chunk-tokens N` fills the cache N positions at a time. Attention
intermediates then scale with the chunk instead of the whole prompt, which cuts
the per-token cost from 240 KiB to 153 KiB against a 144 KiB cache. The
arithmetic is the same incremental attention the scorer already uses for
candidate tokens, but a chunk boundary reorders a bf16 reduction, so candidate
probabilities move by up to a few percent relative: hold this setting fixed
across everything being compared, and treat a change to it as a new condition.
Prefill also costs wall time roughly in proportion to the chunk count (a
9914-token chunked prefill took 5.0s against 2.7s for a 6624-token single
forward).

Lowering `--llm-vram-headroom-gib` to 0.75 buys about 3400 more tokens and
leaves 0.90GiB spare instead of 1.50GiB. Close the browser first; Firefox alone
held 235MiB in one measurement.

`--llm-context-encoding` decides what the budget is spent on, and the runner
pins `action_only`. That holds one encoding for the whole run and fits about 28
demonstrations, at the cost of the per-step state the other arms train on.
`auto` sends that annotation and drops it once the prompt stops fitting, which
on this GPU happens partway through and leaves a run whose early and late
episodes are not the same condition. `state_delta` keeps the parity and fails
rather than shedding it, which here means a run stops once memory exceeds
roughly ten active demonstrations.

So on a 12GiB card shared with a desktop, a complete run and full state parity
are not both available. Pin `action_only` for a complete run here, or take the
display off the card for both:

#### Native crashes and per-event resume

On this GPU the run does not only risk running out of memory; it also crashes.
The worker segfaults inside bitsandbytes' native 4-bit dequantization:

```
bitsandbytes/backends/cuda/ops.py  _dequantize_4bit_impl
bitsandbytes/functional.py         dequantize_4bit
src/llm_baseline.py                direct_forward
```

This is third-party native code, reached once per quantized module per model
forward. Measured on the homogeneous scenario, a scored decision costs about 16
forwards and an event about 224, so one scenario makes roughly 10.8 million
native dequantization calls. Three observed crashes landed after 1.0M, 1.9M and
0.8M calls, which puts the failure rate near one per one to two million and
makes several crashes per scenario the expected case rather than a surprise.

Replacing that call with a pure-PyTorch NF4 dequantization was measured and
rejected: it agrees with the native kernel to one bfloat16 unit, but it is 18
times slower (5.38ms against 0.30ms for the largest module, so about 1.2s per
model forward) and needs seven times the transient memory (672MiB against
96MiB). Keeping the weights dequantized instead needs 5.58GiB in bfloat16,
which does not fit beside a display.

So the crash is not currently avoidable, and the runner is built to survive it
instead. `--event-resume`, which the LLM runner enables for itself, saves each
stream's agent and accumulated rows after every event and continues from the
last completed one. Re-running the same command finds its own incomplete run
directory and continues there rather than starting a new one:

```bash
./hrc llm all
# crashes after a few hours; run the identical command again to continue
```

The saved state excludes the GPU model and keeps both random streams, so a
resumed stream reproduces an uninterrupted one row for row; the test suite
asserts that against a real stream. Reported wall time carries the retained
work forward, so it measures producing the results rather than the last
process. A resume state is only accepted for the same experiment label, config
hash, baseline, scenario, seed and event count; anything else starts over
rather than splicing two different runs together. Checkpoints past the resume
point are deleted before replaying those events.

When a worker does die, `worker_fault.log` in the seed directory holds the
native stack, and the raised error names that file. An empty or absent log
means the process was killed by a signal `faulthandler` cannot catch, which
distinguishes a fault from an out-of-memory kill.

#### Reclaiming the display's memory

For a full-length annotated run, give the GPU no display work:

```bash
sudo systemctl isolate multi-user.target     # stop the desktop session
./hrc llm all --llm-vram-headroom-gib 0.25 --llm-context-encoding state_delta
sudo systemctl isolate graphical.target      # restore it afterwards
```

Run this from a TTY or over SSH, since it ends the graphical session. With a
second GPU or an integrated GPU driving the display, pass
`CUDA_VISIBLE_DEVICES` for the compute card and `--llm-vram-headroom-gib 0`
instead. The cap stays active either way, so an unexpectedly long prompt still
fails as an exception rather than a hang.

The LLM protocol therefore retains the 20-recipe panel, seven homogeneous
climb/settle phases, 210-demonstration budget, heterogeneous recurrence process,
complete controlled holdout, and 48-pair frozen probes. Its single-seed results
are descriptive and do not support seed-level confidence intervals or
significance claims.

One command runs all three scenarios in sequence, and re-running it continues
whatever an earlier attempt left unfinished:

```bash
./hrc llm all
```

A scenario that already completed is skipped at the seed level; the one that
crashed continues from its last completed event. Each scenario can also be run
on its own without changing its plan, metrics, or paired schedule:

```bash
./hrc llm homogeneous
./hrc llm heterogeneous
./hrc llm holdout
```

Aliases `homo`, `hetero`, and `axis` are accepted. The dedicated runner accepts
exactly one seed; `--seeds <seed>` can replace `1337` but cannot specify a list.
LLM artifacts are written separately under `eval_results/llm_runs/`. Every
completed event is atomically saved under the seed's `partial/checkpoints/`
directory, and `partial/summary.json` identifies the newest checkpoint. The
terminal prints one concise line per LLM event and at Full phase boundaries.

## Code map

- `src/environment.py`: symbolic states, actions, and reference recipes.
- `src/domain.py`: injectable state, feature, legality, and workflow-role boundary.
- `src/preferences.py`: goal-preserving workflow transformations.
- `src/memory.py`: replay variants, recurrence horizons, decay, pruning,
  and the horizon-estimator/pin-scope modes the retention ablation selects.
- `src/models.py`: MaxEnt inverse reinforcement learning and semantic fallback.
- `src/latent_strategy.py`: lightweight episode-level timing residual.
- `src/adaptive_agent.py`: online state machine, retraining policy, and the
  active-replay influence audit.
- `src/baselines.py`: deployable and diagnostic comparison agents.
- `src/evaluation.py`: scenario generation, metrics, and immutable artifacts.
- `src/ablations.py`: matcher, routing, latent-strategy, memory, representation,
  component, and retention diagnostics.
- `src/plotting.py`: figures and paired holdout inference.
- `src/hrc_simulation.py`: alternating human–robot interaction simulator.
- `src/real_robot/`: physical task domain, live protocol, operator UI, and local Stretch bridge.
- `src/real_robot/robot_configs/stretch3_lab.json`: uncalibrated five-station pilot catalog and marker map.
- `src/llm_baseline.py`: optional frozen in-context action predictor.
- `burrito/wrapper/adaptive_hrc_burrito/`: physical-domain adapter,
  completion-checked task options, and the two-player HRC protocol.

## Result artifacts

Every completed evaluation is immutable under `eval_results/runs/<run>/`.
`eval_results/latest` identifies the most recent completed run. Generated
results and external model weights are not part of the source repository.

A run is organized by arm, not by scenario:

```
eval_results/runs/<run>/
  manifest.json                  run configuration, code provenance, cell list
  status.json                    completed / expected cells
  baselines/<arm>/
    status.json                  this arm's roll-up
    scenarios/<scenario>/seeds/<seed>/
      plan.json  summary.json  status.json
      tables/*.jsonl.gz          this arm's episode, turn, frozen-probe,
                                 diagnostic and transfer rows
      partial/                   per-event checkpoints and resume state
  shared/routing/<scenario>/<seed>.json    full's realized schedule
  aggregate/
    suite_summary.json  paired_bootstrap.json
    cells/<scenario>/<seed>/     the cross-arm view: per_baseline for every
                                 arm, plus the oracle-gap table
```

The unit of work is one `(arm, scenario, seed)` cell, and the suite runs
**arm-major**: `full` completes all three scenarios across every seed, then
`no_decay`, and so on, with the scenario/seed grid parallel inside each arm.
Cells resume individually, so changing one arm's behaviour costs one arm's
cells rather than the whole grid -- delete `baselines/<arm>/` and resume.

That works because `full` publishes its realized observe/assist schedule to
`shared/` rather than passing it in memory to the arms that replay it. The
route outlives the process that produced it, so a later arm is scored on the
identical sequence of observations and assists without `full` being re-run. An
arm that needs the route and cannot find one fails rather than quietly scoring
its own schedule, and a route whose length no longer matches the plan is
rejected -- if the schedule itself changed, every arm has to be re-run.

Only the cross-arm claims -- the oracle gaps and the paired full-minus-baseline
deltas -- are assembled after the arms finish, into `aggregate/`. Running the
arms apart is numerically identical to running them together; the test suite
asserts that directly, field by field, for everything except wall-clock
readings.

Real-robot sessions are written under `eval_results/real_robot_runs/`. Bridge
execution state stays under `src/real_robot/real_robot_bridge_state/`; it is
local runtime data, separate from evaluation results. Both locations are ignored
by Git.
