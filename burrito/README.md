# Adaptive-HRC Burrito integration

This directory isolates the Overcooked/Burrito evaluation from the existing
symbolic experiments.  It intentionally is **not** a Python package named
`burrito`: that name belongs to the upstream simulator.

## Layout

```text
burrito/
  wrapper/                         our `adaptive_hrc_burrito` package
  third_party/talents-zsc/         pinned TALENTS Git submodule
    overcooked/overcooked_ai/      pinned nested Overcooked-AI submodule
  requirements-runtime.txt         pinned minimal simulator dependencies
  bootstrap.sh                     create `.venv` and install the wrapper
  pins.json                        machine-readable revision contract
```

The learning stack remains in `src/`. The wrapper injects a Burrito
`DomainAdapter` into the same `AdaptiveAgent`; it does not copy MaxEnt IRL,
replay memory, semantic fallback, latent strategy, or decay code.

## Decision abstraction

Adaptive-HRC acts at completion-checked task-option boundaries rather than at
individual movement ticks. Each recipe has eight handoff-safe macros: fetch and
stage protein, prepare and stage protein, start protein cooking, start boiling
rice, stage a clean plate, collect and stage cooked rice, assemble the burrito,
and serve. Steak and mushroom recipes use the same workflow roles; only their
grounded ingredient options differ.

Ground truth is not a fixed action string. A physical task graph recomputes its
legal dependency frontier after every macro, then one of four partial-order
policies filters it: `protein_first`, `rice_first`, `plate_early`, or
`plate_jit`. Every remaining action is recorded as acceptable. When several
actions are equally consistent, a seeded continuous Gaussian draw selects the
human reference action without an alphabetical tie-break.

Each option uses the unmodified upstream navigation planner, verifies its
physical postcondition, and returns the acting player empty-handed to a neutral
parking cell. This is necessary because an upstream action-completion flag can
represent only a partial chop/wash interaction, and a stationary teammate can
otherwise block the planner's selected station goal.

The controller also applies two preference-neutral safety constraints required
by the longer layout: rice is started before protein cooking (the
`protein_first` preference still requires protein *preparation* before rice),
and plate-JIT begins once both components are cooking rather than waiting until
they are already finished. Both physical players remain in collision-aware
planning; the non-actor receives STAY. Controlled handoff cells are versioned
in source and copied into every experiment manifest.

`BurritoDomainAdapter` retains grounded physical state for exact MaxEnt states,
uses navigation-invariant reward features, and supplies an ingredient-masked
semantic view for steak/mushroom transfer. Unobserved legal options receive a
deterministic task-progress projection; observed simulator transitions replace
that projection as soon as evidence exists.

## HRC protocol

`BurritoHrcRunner` enforces the normal evaluation contract:

- the first exposure to a recipe is one human observation demonstration;
- every later exposure is assist mode, beginning with a human option and then
  alternating human and robot decisions;
- an incorrect robot proposal is logged but never applied physically; the
  human performs an action from the acceptable ground-truth set and the next
  turn remains a robot turn;
- both players move physically through the same Burrito state, and every task
  must end in one simulator-recorded dish delivery;
- passive cooking frames and native navigation are logged as durations and
  primitive calls, but only the eight semantic macros enter the demonstration;
- the replay/decay clock advances once per completed recipe, never once per
  simulator frame.

## Reproduce the upstream runtime

From the repository root:

```bash
./burrito/bootstrap.sh
./burrito/.venv/bin/python -m adaptive_hrc_burrito verify
./burrito/.venv/bin/python -m adaptive_hrc_burrito smoke
./burrito/.venv/bin/python -m adaptive_hrc_burrito protocol-smoke
./burrito/.venv/bin/python -m unittest discover -s burrito/tests -v
```

No Conda installation is required. `bootstrap.sh` is idempotent, creates an
isolated Python 3.10 environment at `burrito/.venv`, and installs only the
packages needed by the simulator wrapper. It automatically uses a working
`python3.10`, or Python 3.10 managed by pyenv. Set `BURRITO_PYTHON` to an
explicit interpreter when neither is discoverable.

`verify` fails if either Git revision differs from `pins.json`. `smoke` checks
one raw upstream macro. `protocol-smoke` executes a complete physical steak
observation followed by an assist episode using the real Adaptive-HRC learner.
Its accuracy is a wiring diagnostic, not a paper result.

## Authoritative experiments

The CLI and committed JSON configs—not an interactive notebook—define Burrito
experiments:

```bash
# Fast deterministic wiring check
./burrito/.venv/bin/python -m adaptive_hrc_burrito validate \
  --config burrito/configs/ci-v1.json

# Progressive two-layout/two-seed validation and all four registered arms
./burrito/.venv/bin/python -m adaptive_hrc_burrito validate \
  --config burrito/configs/validation-v1.json

# Paper-scale matrix; intentionally refuses a dirty working tree
./burrito/.venv/bin/python -m adaptive_hrc_burrito validate \
  --config burrito/configs/publication-v1.json
```

Every invocation creates a new immutable directory under `burrito/results/`
(or `--output`). It contains `config.json`, `manifest.json`, `episodes.json`,
`failures.json`, `summary.json`, and, after `validate`, `validation.json`.
Manifests record the root, TALENTS, and nested Overcooked revisions; dirty
paths; exact package versions; seeds; layouts; macro definitions; feature
versions; hardware; command; and completion status.

The physical planner uses one fixed `planner_seed` to generate the same
deterministic episode/option route stream for every paired arm, while the
configured seeds control learning and human acceptable-action sampling.
Navigation/option failures are reported separately from invalid or inaccurate
predictions.

Reported measures include reference and acceptable-set top-1/NLL, robot and
human-shadow accuracy, interventions, human action load, adaptation latency,
recovery AUC, recurrence retention, delivery/time, invalid predictions,
planner failures, estimated training FLOPs, compute time, model storage, and
adaptive-memory evidence per macro decision. Per-decision logs deliberately
omit full probability vectors to keep artifacts compact.

## Current controlled scope

The controlled protocol uses the two-player `burrito_1-2_2p` and `burrito`
layouts, steak and mushroom burritos, eight task macros, and four partial-order
preferences. Expanding layouts or option sets is an experiment change and
must be added through a new versioned config rather than silently changing the
benchmark. See [`AUDIT.md`](AUDIT.md) for the step-by-step implementation map.
