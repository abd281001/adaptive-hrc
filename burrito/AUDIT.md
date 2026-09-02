# Adaptive-HRC Burrito implementation audit

This audit maps the implementation to the ten-step integration gameplan. It
describes source guarantees; publication claims still require running the
versioned publication matrix from a clean commit.

| Step | Status | Machine-checkable evidence |
| --- | --- | --- |
| 1. Freeze symbolic system | Complete | Commit `eee03781276f4cefd7e419b663f2ba629234c192` is the pre-integration reference. `tests/golden/symbolic_freeze_v1.json` freezes the seeded policy and byte-level reward-weight digest; `tests/test_symbolic_golden.py` enforces it. |
| 2. Domain interface | Complete | `src/domain.py` defines all eight required methods. `SymbolicDomainAdapter` is the default and delegates to the legacy transition and feature functions. MaxEnt and `AdaptiveAgent` consume the injected adapter. |
| 3. Isolated Burrito wrapper | Complete | Per the repository-specific layout decision, integration code is under `burrito/wrapper/adaptive_hrc_burrito/`; the existing learner, memory, semantic fallback, and latent model remain in `src/` and are imported rather than copied. |
| 4. Pinned environment | Complete | `pins.json`, `.gitmodules`, recursive revision verification, fully pinned `requirements-runtime.txt`, and `bootstrap.sh` provide an isolated Python 3.10 environment. The wrapper imports the unmodified upstream `BurritoEnv` and planner. |
| 5. Burrito representation | Complete | `domain.py` encodes both players, held/shared objects, processing timers, readiness/burn state, public orders, actor, physical station capacity, and task progress. Reward/semantic features exclude grid coordinates and orientations. |
| 6. Burrito semantics | Complete | MaxEnt receives relational reward features; fallback receives protein-masked functional predicates. Feature versions and the calibrated `0.10` RMS threshold are recorded in every manifest. Burrito workflow roles are domain-provided to the unchanged latent model, and a test requires the declared role set to exactly equal the roles reachable from both macro action spaces. |
| 7. Dynamic task graphs | Complete | `task_graph.py` recomputes the physical dependency frontier and an acceptable set after every macro for four partial-order policies. Seeded Gaussian sampling avoids declaration/alphabetical tie bias. |
| 8. HRC protocol | Complete | `protocol.py` enforces one observation on first recipe exposure, human-first assist, alternating handoff boundaries, pre-execution veto/correction, robot retry, one memory-age event per recipe, and task-graph masking. Native actions and waits are logged but never become demonstration actions. |
| 9. Progressive validation | Complete | `validation-v1.json` covers observation-to-assist, within-recipe preferences, cross-recipe transfer, preference switches, long-gap adaptive memory, two layouts, two learner seeds, and the four registered core ablation arms. Unit tests cover individual options and protocol invariants. |
| 10. Reproducibility | Complete | The CLI and versioned configs are authoritative. Immutable run directories contain copied config, manifest, episode records, failures, summary, and validation. Manifests record commits, dirty paths, packages, seeds, layouts, macros, feature/role versions, metric definitions, hardware, and command. CI runs the pinned deterministic smoke. Publication config refuses dirty Git. Model footprint is an absolute retained-payload byte metric with explicit scope and exclusions; the invalid recipe-length-normalized storage proxy was removed in result schema v2. |

## Verification commands

```bash
venv/bin/python -m pytest -q
./burrito/.venv/bin/python -m unittest discover -s burrito/tests -v
./burrito/.venv/bin/python -m adaptive_hrc_burrito validate \
  --config burrito/configs/ci-v1.json --output /tmp/burrito-ci
```

The controlled planner seed is fixed separately from learner/ground-truth
seeds. It deterministically generates the same episode/option route stream for
every paired arm, so upstream route choice is not reported as model variance;
planner failures remain a separate outcome in experiment artifacts.
