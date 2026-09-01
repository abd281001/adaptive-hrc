# Adaptive Preference Memory for Human–Robot Collaboration

This repository contains the simulation and evaluation code for longitudinal
human workflow adaptation in a symbolic kitchen. A simulated user provides
demonstrations over time; the system predicts robot actions, retains active
recipe-preference variants, decays obsolete variants, and retrains online.

The repository is organized around one executable notebook, `HRC.ipynb`. The
notebook runs the standard evaluation and the reviewer-facing ablations using
the implementation in `src/`. It deliberately contains no model or evaluator
implementation and is committed without execution outputs.

## Setup

The normal workflow uses one Python 3.13 environment, `venv`, for
`HRC.ipynb`, the standard experiments, ablations, and tests. From a fresh
clone:

```bash
git clone https://github.com/abd281001/adaptive-hrc.git
cd adaptive-hrc
./hrc setup
./hrc kernel
```

Open `HRC.ipynb` in a Jupyter-compatible frontend, select the
`Python (adaptive-hrc)` kernel, and run the cells from the repository root.
Running all cells launches the normal evaluation and the notebook ablations;
no separate command or environment is involved.

Generated artifacts are written beneath `eval_results/`. Full evaluation is
computationally expensive. The notebook defaults to one seed worker for
portability. Set `HRC_WORKERS` to at most 5 and `HRC_ABLATION_WORKERS` to at
most 3 before starting Jupyter only when sufficient CPU and memory are
available.

The same `venv` is used for the command-line normal workflows:

```bash
./hrc test
./hrc run
./hrc ablation --suite matcher
```

Additional evaluator arguments can be appended to any command.

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
  retention decisions.
- Teacher-forced and live Top-1 are reported separately. The primary workload
  measure is normalized human action load: scheduled human actions plus
  corrections, divided by recipe steps.

The fixed paired seeds are `1337`, `2024`, `7`, `9001`, and `31415`.
Experiment manifests record the configuration, Git commit, dirty-tree state,
runtime package versions, and completion status. A resumed run must match its
recorded configuration and experiment label.

## Ablations

The notebook runs three independent diagnostic suites:

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
./hrc llm
```

The launcher configures the GPU and thread defaults and automatically resolves
the cached Qwen snapshot. The baseline remains separate from the normal
notebook experiment and uses the same deterministic evaluator protocol. The
default command runs Full and the LLM together on the shared realized schedule.
Each scenario can also be run independently without changing its plan, seeds,
metrics, or paired schedule:

```bash
./hrc llm homogeneous
./hrc llm heterogeneous
./hrc llm holdout
```

Aliases `homo`, `hetero`, and `axis` are accepted. Add `--seeds 1337` (or any
comma-separated subset of the paper seeds) to make each invocation shorter.

## Code map

- `src/environment.py`: symbolic states, actions, and reference recipes.
- `src/preferences.py`: goal-preserving workflow transformations.
- `src/memory.py`: replay variants, recurrence horizons, decay, and pruning.
- `src/models.py`: MaxEnt inverse reinforcement learning and semantic fallback.
- `src/latent_strategy.py`: lightweight episode-level timing residual.
- `src/adaptive_agent.py`: online state machine and retraining policy.
- `src/baselines.py`: deployable and diagnostic comparison agents.
- `src/evaluation.py`: scenario generation, metrics, and immutable artifacts.
- `src/ablations.py`: matcher, routing, and latent-strategy diagnostics.
- `src/plotting.py`: figures and paired holdout inference.
- `src/hrc_simulation.py`: alternating human–robot interaction simulator.
- `src/llm_baseline.py`: optional frozen in-context action predictor.

## Result artifacts

Every completed evaluation is immutable under `eval_results/runs/<run>/`.
`eval_results/latest` identifies the most recent completed run. Each seed
stores a compact summary plus compressed episode, turn, frozen-probe,
diagnostic, transfer, and oracle-gap tables. Generated results and external
model weights are not part of the source repository.
