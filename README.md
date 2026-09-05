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

1. the standard evaluation: three scenarios times five paired seeds, written
   as one new directory under `eval_results/runs/`;
2. the validated Overcooked/Burrito evaluation, executed through its isolated
   Python 3.10 runtime and written under `eval_results/cooking/`; and
3. the matcher, routing, and latent-strategy ablations, written as one new
   directory under `eval_results/ablation_runs/`.

The three scenarios are stored inside one standard run directory; they are not
three separate top-level runs. By default, every notebook execution creates new
immutable run directories rather than overwriting an earlier execution. An
explicit cooking resume reuses only verified completed cell checkpoints.

Generated artifacts are written beneath `eval_results/`. The full workflow is
computationally expensive. The standard evaluation has a 20-process cap and
admits only complete seed cohorts: with five seeds, all three scenarios run in
parallel using 15 workers; with eight seeds, two scenarios run at a time using
16. Set `HRC_WORKERS` before starting Jupyter to request a lower cap. Each
longitudinal ablation suite defaults to one worker; `HRC_ABLATION_WORKERS` may
raise that to at most three. The cooking config uses all available CPUs by
default; `HRC_COOKING_WORKERS` caps that count. Set
`HRC_COOKING_RESUME` to an interrupted cooking run directory to reuse its
completed seed/scenario/arm checkpoints.

The launcher routes each workflow to its owning environment:

```bash
./hrc test
./hrc run
./hrc cooking
./hrc ablation
./hrc ablation --suite matcher
```

`./hrc run` runs only the standard three-scenario, five-seed evaluation;
`./hrc cooking` runs the sole full cooking config; and `./hrc ablation` runs all
three ablation suites. Passing `--suite matcher`, `routing`, or `latent` selects
one ablation. Additional evaluator arguments can be appended to the runners.

The Overcooked/Burrito replication uses a separate pinned Python 3.10
environment under `burrito/.venv` because it is an external replication with
incompatible dependencies. `./hrc cooking` invokes that interpreter but does
not create it or install its dependencies, so it cannot perturb either core
environment. See
[`burrito/README.md`](burrito/README.md) for its isolated setup. It includes a
physical task-option adapter and the same one-observation-then-assist correction
protocol used by the symbolic evaluation. Its sole publication experiment is
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
motion. See [`docs/real_robot.md`](docs/real_robot.md) for the architecture,
calibration checklist, remote-GUI options, safety gates, and robot commands.
Use `./hrc robot-doctor` for a read-only deployment preflight and
`./hrc robot-report RUN_DIRECTORY` for an integrity/outcome summary. Live UI
startup requires the explicit `--require-motion` handshake, a matching full
configuration digest, and a ready motion-enabled bridge.
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

The fixed paired seeds are `1337`, `2024`, `7`, `9001`, and `31415`.
Experiment manifests record the configuration, Git commit, dirty-tree state,
runtime package versions, and completion status. A resumed run must match its
recorded configuration and experiment label.

## Ablations

The notebook asks the application runner to launch three independent diagnostic
suites concurrently:

- matcher stress tests for recipe/preference identification;
- observation-routing comparisons under matched and local recovery schedules;
- a four-arm comparison of MaxEnt-only and latent timing strategies.

Each suite writes its JSON results, command, timing, logs, and completion state
under a unique run directory. The routing and latent suites use seed `1337`
across all three scenarios by default.

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

It receives the same information as the other arms: the same plan, seed, and
realized observe/assist schedule; the same active replay demonstrations and
retention weights; the same `(state, action)` step content, rendered as named
predicates and per-step deltas; and the same shared state-only action mask. It
receives no recipe label, preference label, or goal predicate. When the
annotated prompt would exceed the context window, the per-step state annotation
is dropped and every retained demonstration is still sent; the turn rows record
which encoding was used in `llm_context_encoding`.

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
rather than five seeds.

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

Two settings trade prompt budget against other things, measured on this host
with a GNOME session running. "Demos" is how many active demonstrations the
budget holds, at roughly 350 prompt tokens per demonstration without the state
annotation and 945 with it:

| Configuration | Prompt budget | Demos, action-only | Demos, annotated |
| --- | --- | --- | --- |
| default | 6624 | 18 | 7 |
| `--llm-prefill-chunk-tokens 1024` | 9914 | 28 | 10 |
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

`--llm-context-encoding` decides what the budget is spent on. The default
`auto` sends the state annotation and drops it once the prompt stops fitting,
so a long run can change encoding partway through and its early and late
episodes are not the same condition. Pinning `action_only` holds one encoding
for the whole run, fits about 28 demonstrations with chunking, and gives up the
per-step state that the other arms train on. Pinning `state_delta` keeps that
parity and fails rather than shedding it, which on this GPU means a run stops
once memory exceeds roughly ten active demonstrations.

So on a 12GiB card shared with a desktop, a complete run and full state parity
are not both available. Pin `action_only` for a complete run here, or take the
display off the card for both:

#### Reclaiming the display's memory

For a full-length annotated run, give the GPU no display work:

```bash
sudo systemctl isolate multi-user.target     # stop the desktop session
./hrc llm homogeneous \
    --llm-vram-headroom-gib 0.25 \
    --llm-prefill-chunk-tokens 1024 \
    --llm-context-encoding state_delta
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

Each scenario can be run independently without changing its plan, metrics, or
paired schedule:

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
- `src/memory.py`: replay variants, recurrence horizons, decay, and pruning.
- `src/models.py`: MaxEnt inverse reinforcement learning and semantic fallback.
- `src/latent_strategy.py`: lightweight episode-level timing residual.
- `src/adaptive_agent.py`: online state machine, retraining policy, and the
  active-replay influence audit.
- `src/baselines.py`: deployable and diagnostic comparison agents.
- `src/evaluation.py`: scenario generation, metrics, and immutable artifacts.
- `src/ablations.py`: matcher, routing, and latent-strategy diagnostics.
- `src/plotting.py`: figures and paired holdout inference.
- `src/hrc_simulation.py`: alternating human–robot interaction simulator.
- `src/real_robot/`: physical task domain, live protocol, operator UI, and local Stretch bridge.
- `robot_configs/stretch3_lab.json`: uncalibrated five-station pilot catalog and marker map.
- `src/llm_baseline.py`: optional frozen in-context action predictor.
- `burrito/wrapper/adaptive_hrc_burrito/`: physical-domain adapter,
  completion-checked task options, and the two-player HRC protocol.

## Result artifacts

Every completed evaluation is immutable under `eval_results/runs/<run>/`.
`eval_results/latest` identifies the most recent completed run. Each seed
stores a compact summary plus compressed episode, turn, frozen-probe,
diagnostic, transfer, and oracle-gap tables. Generated results and external
model weights are not part of the source repository.
