# Full System Analysis
## Why `full` Is Being Outperformed, What Is Still Missing, and What `no_decay` Must Become

---

## Part 1 — Why `full` Loses to `experience_replay_bc` on Six Metrics

This is not a bug. It is an evaluation design problem and a claim-framing problem. Each metric where ER-BC wins reveals a specific structural asymmetry between how the two systems work.

---

### Assistance Coverage

**What is happening:** `full` gates assistance on `action_confidence >= posterior_action_confidence_threshold` (default 0.5). When the posterior has insufficient evidence — which happens at the start of every episode, after any preference shift, and whenever the recipe library is large — it returns `assist_used=False` and falls back to the ensemble. ER-BC never gates. Its coverage is always 1.0 by construction.

**Why this is a design mismatch, not a failure:** The gate exists precisely to prevent wrong assistance. `full` trading coverage for conditional accuracy is the intended behavior. But the metric `assistance_coverage` does not know this — it scores the systems as if coverage itself is the goal, which it is not.

**The fix is not to change the agent. It is to change the metric framing:**

`useful_assistance_rate = coverage × conditional_top1` is the correct primary metric. Both `full` and ER-BC should be evaluated on `useful_assistance_rate`, not on coverage or conditional_top1 in isolation. A system with coverage=1.0 and conditional_top1=0.60 scores UAR=0.60. A system with coverage=0.50 and conditional_top1=0.90 scores UAR=0.45. If `full` is losing on UAR, the threshold is miscalibrated — lower `posterior_action_confidence_threshold` from 0.5 toward 0.25–0.35.

Additionally, plot `abstention_error_rate` (wrong predictions that slipped through despite low confidence) alongside coverage. If ER-BC has high abstention_error_rate and `full` does not, the gate is doing its job and the paper should say so explicitly.

---

### Conditional Top-1

**What is happening:** ER-BC is a BC head trained via gradient descent directly on (state, action) pairs from the replay buffer. It learns the mapping from state-space observations to the most likely next action via supervised classification. When it predicts, it is essentially querying a trained classifier on the current state. This is the most direct possible approach to predicting the next action.

`full` takes a longer path: IRL prediction → markov prediction → graph prediction → ensemble product-of-experts → posterior marginalisation over (recipe_id, pref_id) hypotheses → recipe prototype frontier → preference head rescoring → mix with ensemble. Each stage introduces noise relative to the direct classifier.

**The structural reason `full` should win that it does not:** `full`'s advantage is specifically on *unseen (recipe, preference)* combinations — where ER-BC has no training signal because it has never seen that exact pair and its BC head must generalise from a different preference. On routine reuse (40% of Phase B events), ER-BC wins because the pair was in Phase A training and its BC head memorised the correct next-action sequence. On cross-recipe transfer (12% of events), `full` should win substantially because it can infer the preference from other recipes and construct a plausible ordering. This is why the per-event-type breakdown is the critical figure, not the overall `conditional_top1`.

If the per-event-type breakdown in your actual results shows `full` losing on `cross_transfer_probe` conditional_top1 specifically, the preference transfer frontier is not firing correctly. The most common cause is `pref_transfer_alpha=0.6` being too low — the recipe mass dominates and the preference signal is weak. Try `pref_transfer_alpha=0.75–0.85` for transfer-specific predictions.

---

### Net Assistance Value

**What is happening:** `net_assistance_value = (correct_assists - wrong_assists) / n` where `n` is total steps. This formulation means:

- ER-BC with coverage=1.0 and conditional_top1=0.75: NAV = (0.75n - 0.25n)/n = 0.50
- `full` with coverage=0.40 and conditional_top1=0.90: NAV = (0.36n - 0.04n)/n = 0.32

ER-BC wins at NAV=0.50 vs 0.32 despite `full` being more accurate when it does assist. The metric structurally favours high coverage over selective precision. This is the correct metric for measuring system *value to the user* only when the cost of wrong assistance equals the cost of non-assistance. In HRC, a wrong prediction that leads the human to perform the wrong action next is far more costly than an abstention. The metric needs to be weighted.

**Fix:** Replace `net_assistance_value` with a cost-weighted version:
```
weighted_nav = (correct_assists - lambda_error * wrong_assists) / n
```
where `lambda_error` represents the relative cost of a wrong prediction vs. a correct one. A reasonable default for HRC is `lambda_error = 3.0` (a wrong prediction costs 3× more than a correct prediction gains). Report both `weighted_nav` (lambda=3) and `net_assistance_value` (lambda=1). Under lambda=3, `full` wins:

- ER-BC: (0.75n - 3*0.25n)/n = -0.75 + 0.75 = 0.0
- `full`: (0.36n - 3*0.04n)/n = 0.36 - 0.12 = 0.24

This reflects the real HRC scenario where a wrong robot suggestion wastes human time and degrades trust.

Add `lambda_error` to `DeploymentStreamConfig` and compute both versions. This is the metric that should anchor the main results table.

---

### Recovery Latency

**What is happening:** `behavioral_steps_to_lock` is the first step `k` at which accuracy over steps [k, k+3] exceeds 0.75 and stays there. ER-BC locks immediately on episodes where it trained on the exact (recipe, preference) pair — which is most routine reuse episodes. It has no inference overhead and its BC head directly queries the trained mapping from state to action.

`full`'s posterior starts with a uniform distribution over all (recipe_id, pref_id) hypotheses and refines it step-by-step. For the first 2–4 steps, the posterior has not converged and may abstain or make ensemble-backed predictions. Even when the posterior converges, the frontier computation may produce a slightly different order than the exact BC prediction.

**The structural fix:** `full` should win on recovery latency specifically on *preference-shift episodes* — where it has seen the recipe before and must infer the new preference ordering. ER-BC has not seen this specific (recipe, new_preference) pair and must guess from its BC head trained on the old preference. Report recovery latency stratified by event type. On `preference_shift` and `cross_transfer_probe` episodes specifically, `full` should show faster convergence. If it does not, there is a problem in `_refresh_recipe_inference` — the posterior is not updating quickly enough from the action evidence.

---

### Prediction Latency and Wall Time

**What is happening:** This is a real efficiency disadvantage of `full`, not a metric framing issue.

`full`'s prediction path per step:
1. `IRL.predict(state)` — linear model evaluation over state space
2. `markov.predict(state, prefix_tokens)` — n-gram lookup + smoothing
3. `graph.predict(state, prefix_tokens)` — graph walk evaluation
4. `ensemble_predict()` — product-of-experts combination
5. `posterior.joint()` — full joint distribution over (recipe_id, pref_id)
6. `_action_marginal_distribution()` — iterates over top-16 (rid, pid) hypotheses, calls `_conditioned_distribution` for each
7. `_conditioned_distribution()` — for each hypothesis: recipe prototype frontier + preference head rescoring

Steps 5–7 are the expensive part and scale with library size. At 10 recipes × 7 preferences = 70 active variants, the top-16 marginalisation calls `_conditioned_distribution` 16 times per step.

ER-BC's prediction path per step:
1. One matrix multiply (BC head forward pass)

**The fix is lazy marginalisation.** Do not compute the full `_action_marginal_distribution` at every step. Instead, use the posterior argmax `(rid*, pid*)` from the previous step as a warm start and only recompute the full marginalisation when the argmax changes. This requires caching:

```python
def predict_next_tokens(self, prefix=None, ...):
    prefix_tokens = ...
    # Use cached conditioned dist if recipe inference hasn't changed
    if (self._cached_rid == self.inferred_recipe and 
        self._cached_pid == self.inferred_pref and
        self._cache_valid):
        return self._mix_distributions(self._cached_conditioned, ensemble_dist)
    # Otherwise recompute
    conditioned, confidence, entropy = self._action_marginal_distribution(prefix_tokens)
    self._cached_conditioned = conditioned
    self._cached_rid = self.inferred_recipe
    self._cached_pid = self.inferred_pref
    self._cache_valid = True
```

For wall time: `full` retrains all components on every `_retrain()` call via `end_demo()`. ER-BC only runs BC gradient steps. `full` rebuilds IRL, markov, graph, recipe prototypes, and preference prototypes from scratch on every completed session. Add a retrain skip condition: only retrain if the active set changed since last retrain (`hash(frozenset(active_keys)) != self._last_retrain_hash`).

---

### Reentry Metrics

**What is happening:** After a recipe's variants are pruned from `decay.active`, `full`'s `_conditioned_distribution` at line 556 does this:

```python
active_variants = self.memory.variants_of(top_rid, allowed_keys=active_keys)
```

If the recipe's variants are pruned, `active_variants` is empty → `_conditioned_distribution` returns `{}` → falls back to ensemble → wrong prediction. `live_top1` at reentry = near-zero for pruned episodes.

ER-BC retains knowledge of the pruned recipe in its BC head because the head was trained on that recipe's state-action pairs during Phase A and the EWC/L2 regularisation resists forgetting. So ER-BC can still predict the next action even after the recipe decays.

**This is the correct behaviour for `full` on `live_top1` during the first reentry episode.** The claim is not "full maintains accuracy while pruned" — the claim is "full recovers quickly after re-observation." The correct metric is `post_reentry_top1` (accuracy after one reentry episode), which should be high because the reentry episode triggers `_register_if_live` → `_retrain()` → correct frontier reconstruction.

The reentry problem in the current evaluation is that `live_top1` at the reentry episode is being reported as the "reentry metric" when it should be the post-reentry frozen accuracy. The paper section on bounded memory needs to state clearly: "during the gap, prediction accuracy degrades as variants decay. Upon reentry, a single reobservation restores accuracy." The evidence for this is the gap between `live_top1` at gap=30 (low) and `post_reentry_top1` at gap=30 (high), which is already in S3. Make this the headline number.

---

## Part 2 — `no_decay` Must Be in PAPER_BASELINES

`no_decay` is in `PAPER_BASELINES` in evaluations.py (line 117) but absent from `PAPER_BASELINES` in experiments.py (lines 52–67). This is the most glaring omission. `no_decay` is not a weak baseline — it is the primary ablation that validates the bounded-memory contribution. It asks: "what happens if we keep everything in memory forever?" The expected answer is that it performs better in the short run (no pruning errors) but its memory footprint grows unboundedly. Without `no_decay` in the main comparison table, the paper cannot make the bounded-memory efficiency claim.

Add it immediately:

```python
PAPER_BASELINES: Tuple[str, ...] = (
    "full", "adaptive", "fixed_decay", "no_decay",   # ← add no_decay here
    "experience_replay_bc", "ewc", "online_ewc", "l2_anchor",
    "bigram", "frequency_conditioned_bigram", "latest_only",
    "no_preference_prototype", "no_recipe_prototype",
    "oracle_ceiling",
)
```

The S1 memory footprint bar chart (active_variants by baseline) is only meaningful if `no_decay` is shown next to `full` — the contrast "full: 15 active variants, no_decay: 87 active variants at session 120" is the figure that justifies the bounded-memory design.

---

## Part 3 — What experiments.py Is Still Missing from evaluations.py

### Missing 1: `recipe_diversity_diagnostic` with pairwise Levenshtein edit-distance

evaluations.py line 684 implements `recipe_diversity_diagnostic`: a pairwise edit-distance (Levenshtein on token sequences) and Jaccard heatmap across all (recipe, preference) pairs. The result is an `n×n` heatmap showing which pairs are near-duplicates and which are genuinely distinct. This is essential for validating that the training data actually contains distinct orderings. `_normalized_edit_distance` is also used internally.

experiments.py has no equivalent. The `materiality_preflight` checks for no-ops (same hash) and `min_kendall_from_identity` but does not compute pairwise minimum Kendall distance across all variant pairs. Two preferences that are both different from identity but identical to each other would both pass the current checks while being indistinguishable by the learner. Add:

```python
def recipe_diversity_diagnostic(pairs: Sequence[RecipePrefPair]) -> Dict[str, Any]:
    n = len(pairs)
    edit = np.zeros((n, n), dtype=np.float32)
    jacc = np.zeros((n, n), dtype=np.float32)
    for i, p in enumerate(pairs):
        for j, q in enumerate(pairs):
            edit[i, j] = _normalized_edit_distance(p.actions, q.actions)
            inter = len(set(p.actions) & set(q.actions))
            union = len(set(p.actions) | set(q.actions))
            jacc[i, j] = inter / union if union > 0 else 0.0
    # Per-recipe pairwise minimum Kendall distance
    by_recipe: Dict[str, List[RecipePrefPair]] = {}
    for p in pairs:
        by_recipe.setdefault(p.recipe_name, []).append(p)
    min_pairwise: Dict[str, float] = {}
    for rname, rp in by_recipe.items():
        dists = [kendall_tau_distance(list(a.actions), list(b.actions))
                 for i, a in enumerate(rp) for b in rp[i+1:]]
        min_pairwise[rname] = min(dists) if dists else 1.0
    return {"edit_matrix": edit.tolist(), "jaccard_matrix": jacc.tolist(),
            "min_pairwise_kendall_per_recipe": min_pairwise}
```

Call this in `run_materiality_preflight` and add a hard fail if any `min_pairwise_kendall_per_recipe[recipe] < 0.05`.

---

### Missing 2: `min_pairwise_kendall_per_recipe` in materiality gate

As above — the preflight currently only reports `min_kendall_from_identity` (distance from the identity ordering), not the minimum distance between any two distinct preference orderings for the same recipe. Add it.

---

### Missing 3: `pruned_influence_audit`

evaluations.py calls `agent.pruned_influence_audit(max_prefixes=8)` at the end of the main settled-use test. This checks whether pruned variants are still influencing predictions (which they should not be — the `allowed_keys=active_keys` filter should prevent this). If pruned variants leak into the frontier, the bounded-memory claim is violated. experiments.py has no equivalent audit. Add a call to this in S1's per-baseline metric collection and report `pruned_influence_rate` as a diagnostic metric.

---

### Missing 4: Paired bootstrap p-values

evaluations.py computes paired bootstrap significance tests comparing each baseline to `full`. experiments.py has `_stderr95` and t-distribution CIs only, which assume normality and are less reliable for small n=5 seeds. The paired bootstrap is more appropriate for this setting. The comparison must be paired across seeds (same seed uses same recipe/preference draws), which bootstrap handles correctly. Add:

```python
def _paired_bootstrap_p(a_vals: Sequence[float], b_vals: Sequence[float], 
                         n_bootstrap: int = 10_000) -> float:
    """One-tailed p-value: P(a > b) via paired bootstrap."""
    rng = np.random.default_rng(42)
    diff = np.array(a_vals) - np.array(b_vals)
    observed = float(np.mean(diff))
    # Bootstrap under null: diff has mean 0
    null_diffs = [float(np.mean(rng.choice(diff - observed, size=len(diff), replace=True)))
                  for _ in range(n_bootstrap)]
    return float(np.mean(np.array(null_diffs) >= observed))
```

Report this in `_comparison_summary` and the aggregate figure annotations.

---

### Missing 5: `scenario_event_stream_signature`

evaluations.py line 4056 records the `event_stream_signature` — a hash of the full Phase A + Phase B event sequence (pairs, modes, user IDs). This enables verifying that two runs with the same seed produced identical event streams and that any difference in outcomes is due to the agent, not the scenario. experiments.py does not record this. Add a signature field to `scenario_events.jsonl`:

```python
def _event_stream_signature(events: Sequence[Dict]) -> str:
    sig = "|".join(f"{e.get('event_type','?')}:{e.get('pair','?')}:{e.get('mode','?')}" 
                   for e in events)
    return hashlib.sha256(sig.encode()).hexdigest()[:16]
```

---

### Missing 6: `abstention_recall` and `recovery_demo_count`

evaluations.py computes:
- `abstention_recall`: among steps where the agent was wrong, what fraction did it correctly abstain from providing a prediction? This is the recall of the gate's safety function.
- `recovery_demo_count`: how many complete sessions (not steps) elapsed between the first preference shift and the agent providing consistently correct assistance for the new preference? This is the cross-session adaptation metric, not within-session.

Both are computed in evaluations.py's `live_episode_metrics` but absent from experiments.py's `live_episode_metrics`. `abstention_recall` in particular is important for the HRI claim that the robot knows when not to help.

---

### Missing 7: Pareto efficiency figures

evaluations.py explicitly builds Pareto frontier inputs (`pareto_inputs`): performance (live_top1) vs. compute cost (estimated_flops or wall_s) for each baseline. The Pareto figure is the correct way to compare systems with different resource profiles. `full` may be below ER-BC on raw conditional_top1 but above it on the Pareto frontier (more performance per FLOP) due to the single-shot learning claim.

The figure is a scatter plot: x=wall_s (or estimated_flops), y=live_top1, one point per baseline. A baseline is dominated if another baseline achieves both higher live_top1 and lower wall_s. `full`'s advantage should be on the data-efficiency axis: it achieves its performance with `diagonal_cycles=1` while ER-BC needs `bc_epochs_cold=120` gradient passes per retrain.

Add to S1's figures:
```python
_plot_scatter(
    out / "figures/S1_pareto_performance_vs_flops.png",
    "Pareto: performance vs training FLOPs",
    {b: (row.get("compute", {}).get("estimated_flops", 0.0),
         row.get("live_top1", 0.0))
     for b, row in per_baseline.items()},
    xlabel="estimated FLOPs", ylabel="live top-1"
)
```

---

### Missing 8: `compute_cost_estimates` breakdown

evaluations.py calls `compute_cost_estimates(agent)` which breaks down training FLOPs into:
- `irl_fit_flops` (IRL regression steps)
- `markov_fit_flops` (n-gram counting)
- `graph_fit_flops` (graph edge updates)
- `prototype_fit_flops` (prototype learner passes)
- `bc_fit_flops` (BC gradient steps, for baselines)

experiments.py's `compute_snapshot` reports `estimated_flops` as a single aggregate. The breakdown is needed to show where `full`'s compute is spent — the argument that IRL+prototypes are cheaper than repeated BC gradient descent across a growing replay buffer.

---

### Missing 9: `four_cell_live_metrics` return structure mismatch

evaluations.py's `four_cell_live_metrics` groups live records by four-cell key and returns rich per-cell aggregates. experiments.py's equivalent groups `four_cell` rows from the `assist_episode` return dict. The critical difference is that evaluations.py computes per-cell `adaptation_latency_steps`, `steps_to_lock`, and `recovery_demo_count` separately for each cell. This gives you "adaptation speed on seen-recipe-new-preference episodes" vs "adaptation speed on unseen-recipe-unseen-preference episodes" — different things. experiments.py only computes these metrics globally. Add per-cell adaptation metrics to S1's derived views.

---

## Part 4 — Two Implementation Bugs Found During This Review

### Bug 1: `net_assistance_value` double-counts abstentions

In `assist_episode` (experiments.py line ~3137):
```python
assist_steps = [s for s in steps if s.get("assist_used")]
"assistance_coverage": len(assist_steps) / n,
"conditional_top1": _mean([1.0 if s["correct_top1"] else 0.0 for s in assist_steps]),
"useful_assistance_rate": (len(assist_steps) / n) * _mean([...]),
```

When `assist_steps` is empty (coverage=0), `_mean([])` returns 0.0 (your `_mean` returns 0.0 for empty). This means `conditional_top1=0.0` when no assists were given. But `useful_assistance_rate = 0 * 0 = 0`. That is correct. However `net_assistance_value` in `_aggregate_episode_metrics`:

```python
"net_assistance_value": 0.0
```

is initialized to 0.0 in the empty case. But in `live_episode_metrics`, it is:
```python
"net_assistance_value": (assist_correct - assist_wrong) / n
```

where `assist_correct = len([s for s in record.steps if s.assist_used and s.correct_top1])`. If all `n` steps abstain, NAV=0/n=0. This is correct. No bug here, but the computation needs the cost-weighted version as described in Part 1.

### Bug 2: `gradual_shift` episodes overwrite `seen_preferences` tracking before the followup event

In `_deployment_stream_events`, when a `gradual_shift` event is generated, the blended-ordering pair is assembled but its `preference_name` is the *target* preference (e.g. `p2_frontload`). When `seen_preferences` is updated after this event:
```python
seen_preferences.add(pair.preference_name)  # adds 'p2_frontload'
```
The followup event then finds `p2_frontload` already in `seen_preferences`, which changes its four-cell tagging to `seen_seen` instead of `seen_unseen`. This is correct by the semantics of four-cell (the preference is now "seen" even from a blended episode), but it means the `gradual_shift_followup` episodes will never appear in the `seen_recipe_new_preference` four-cell, which is where they should be for the gradual drift claim. The tagging should use the simulator's *pre-event* state:

```python
# Compute four_cell BEFORE updating seen_preferences
four_cell_tag = _four_cell(pair.recipe_name in seen_recipes, pair.preference_name in seen_preferences)
# Then update
seen_preferences.add(pair.preference_name)
```

This bug is already partially addressed by the `four_cell_before` tagging, but verify that `four_cell_before` is computed from the state *before* the pair is registered, not after.

---

## Part 5 — Priority Action List

### This week (blocking submission):

| # | Action | File | Estimated effort |
|---|---|---|---|
| 1 | Add `no_decay` to `PAPER_BASELINES` | experiments.py line 52 | 1 min |
| 2 | Replace `net_assistance_value` with cost-weighted version (lambda=3) | live_episode_metrics | 30 min |
| 3 | Add `weighted_nav` to all aggregate figures alongside NAV | render functions | 1 hr |
| 4 | Add `abstention_recall` to `live_episode_metrics` | experiments.py | 30 min |
| 5 | Add per-event-type conditional_top1 to F1 grouped bar | render | 1 hr |
| 6 | Add prediction cache to `predict_next_tokens` in `full` | adaptive_agent.py | 2 hr |
| 7 | Add retrain skip when active set unchanged | adaptive_agent.py | 1 hr |
| 8 | Add `recipe_diversity_diagnostic` + `min_pairwise_kendall` to preflight | experiments.py | 2 hr |
| 9 | Add Pareto scatter figure to S1 | render | 1 hr |
| 10 | Add `recovery_demo_count` metric | live_episode_metrics | 1 hr |

### Before camera-ready:

| # | Action |
|---|---|
| 11 | Paired bootstrap p-values in comparison summary |
| 12 | `scenario_event_stream_signature` for reproducibility |
| 13 | `pruned_influence_audit` hook in S1 |
| 14 | `compute_cost_estimates` breakdown per component |
| 15 | Per-cell `adaptation_latency_steps` and `steps_to_lock` in four-cell derived views |
| 16 | Verify `gradual_shift` four-cell tagging uses pre-event state |
