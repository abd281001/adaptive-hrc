
| `src/adaptive_agent.py` | Defines the main adaptive HRC agent. It controls observation vs online-assist modes, registers demonstrations, maintains memory/prototypes/posterior state, retrains the prediction heads, predicts the next robot action, handles frozen evaluation, and commits online preference shifts. This is the core proposed system. |
| `src/baselines.py` | Defines comparison agents and ablations used in experiments. These include memory-policy baselines (`latest_only`, `no_replay`, `fixed_decay`, `no_decay`), behavior-cloning/replay baselines, EWC/online-EWC regularization baselines, simple frequency floors, and the oracle ceiling. Each baseline modifies either the memory policy, training data, prediction architecture, or continual-learning mechanism relative to the full agent. |
| `src/environment.py` | Implements the symbolic kitchen world. It defines objects, ingredients, locations, tools, state features, action execution, action validity checks, and the recipe generator. This is the simulator substrate from which demonstrations and recipe action sequences are produced. |
| `src/experiments.py` | Contains the paper-facing evaluation harness. It builds recipe/preference pairs, defines experiment configs, runs scenario streams, evaluates baselines, records metrics, aggregates seed results, and writes structured JSON/JSONL outputs. This is where deployment streams, transfer tests, decay/reentry tests, ablations, stress tests, and tuning runs are orchestrated. |
| `src/hrc_simulation.py` | Implements the alternating human-robot collaboration protocol. It simulates robot turns, human correction after wrong robot actions, interaction timing/cost accounting, workload metrics, and per-turn metadata. This keeps HRC episode mechanics separate from the learning agents. |
| `src/memory.py` | Defines the non-parametric memory and disambiguation layer. It provides variant hashing, Jaccard/Kendall sequence distances, known-variant classification, variant storage, active/pruned replay entries, adaptive decay, latest-variant protection, and reentry tracking. |
| `src/models.py` | Implements the predictive model heads and shared configuration. It defines `Config`, state/action trajectory construction, feature extraction/normalization, MaxEnt IRL training and prediction, state-aware n-gram prediction, ensemble fusion, and utility functions such as top-k extraction. |
| `src/posterior.py` | Implements recipe and preference prototypes plus the online posterior over `(recipe_id, latent_pref_id)`. Recipe prototypes summarize task structure and frontiers; preference prototypes summarize role-ordering styles; the posterior combines recipe, preference, and memory-prior evidence during online assistance. |
| `src/preferences.py` | Defines workflow preference presets and recipe-order transformations. It encodes preference axes such as ingredient flow, equipment setup, serving setup, and cleanup timing, then applies those preferences to base recipes while preserving validity where possible. |
| `src/representations.py` | Converts symbolic kitchen actions into state-transition observations and abstract role labels. It provides action transition vectors, role extraction, role n-grams, task signatures, and preference signatures used by prototypes and posterior inference. |

## High-Level Flow

1. `environment.py` generates valid base recipes.
2. `preferences.py` creates preference-specific versions of those recipes.
3. `representations.py` converts actions into transition vectors and role abstractions.
4. `adaptive_agent.py` observes demonstrations, stores variants through `memory.py`, builds prototypes through `posterior.py`, and trains prediction heads from `models.py`.
5. `hrc_simulation.py` runs alternating human-robot assist episodes.
6. `experiments.py` coordinates scenarios, baselines, metrics, and output files.

