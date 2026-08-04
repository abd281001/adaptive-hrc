# Review: Adaptive Preference Learning for Recipe-Level HRC

**Reviewer stance:** HRI / ICRA / CoRL-style review. Critical, evidence-based, constructive.

**Methodology note:** I read all ten files in full (representations → models → memory → preferences → posterior → environment → adaptive_agent → baselines → hrc_simulation → evaluation), then wrote and ran diagnostic scripts against your actual code (not hypothetical examples) to verify or refute several suspicions before writing them up. Every empirical number below (Jaccard values, collision rates, co-occurrence probabilities, the `live_topk` bug) came from executing your code, not from reading it and guessing. I'll flag clearly which findings are "read the code" vs. "ran the code and here's the output."

**Scope caveat — please read before the rest:** You asked me to focus on the graphs, since that's what reviewers actually look at. The ten files you uploaded contain the full experiment harness (`evaluation.py`) down to JSON/JSONL metric rows, but **no plotting code** (no matplotlib/seaborn/plotly, no notebook, no figure-generation script). So I cannot review axis choices, error bars, aggregation-for-display, or chart type. What I *can* do — and what I've done below — is review everything that **feeds** the graphs: the metric definitions, the aggregation functions, and whether the numbers going into `summarize_stream()` mean what their names claim. If a graph is built on `aggregate_episode_metrics()`, and that function has a bug, the graph has the bug too, whatever it looks like. If you can share the plotting code in a follow-up, I'll do a second pass specifically on chart design.

---

## Top-line assessment

This is a substantially more sophisticated system than the framing document's prose suggests, and more sophisticated than most papers in this space attempt. The continual-learning baseline suite alone (EWC with correct multi-task Fisher accumulation, online/EWC++, L2-anchor, budget-matched replay, recency-prioritized replay, a clairvoyant future-aware pruning oracle) reflects real familiarity with the lifelong-learning literature, not a token gesture at "we compared to baselines." The freeze/snapshot contract for evaluation isolation, the active-only prototype rebuild discipline, and the `pruned_influence_audit` regression check are the kind of engineering that most HRC papers never bother with and that a sharp reviewer would normally have to demand in the rebuttal period.

That said, I found a **confirmed, reproducible bug in a metric that would land in your paper** (`live_topk` aggregation — Finding 13), a **confirmed recipe-identity collision** between two of your own "distinct" recipes that exceeds your own disambiguation threshold (Finding 1), and a **large gap between what the codebase can do and what the evaluation harness actually runs** — 12 of your 21 implemented baseline/ablation classes, including the two or three most important ones for your central claims, are never imported by `evaluation.py` and cannot currently appear in any result (Part IV). None of these are hard to fix. All of them would currently produce numbers or plots that don't say what you think they say. I'd treat fixing these three as a precondition for trusting anything else in the pipeline, because the failure modes are silent — nothing crashes, nothing warns, the numbers just come out wrong or incomplete.

Below, Part I covers implementation, Part II covers scenario/evaluation design, Part III covers metrics (the "what becomes graphs" ask), Part IV is specifically about the baseline suite, and Part V is missing experiments. Each numbered finding has: what/where, why it matters for the paper, and a concrete fix. I've tried to be precise about severity — I don't think everything here is equally important, and I say so.

---

## Part I — Implementation

### What's genuinely solid, briefly

I don't want the length of the issues list below to misrepresent the baseline quality of this code, so before I get critical: the freeze-context digest verification in `adaptive_agent.py` (assert nothing mutated during a read-only eval probe), the fact that `role_from_transition` and the disambiguator never touch ground-truth preference labels (checked this explicitly — labels really are evaluator-only metadata, not fed to the learner), the active-only prototype rebuild with stable-ID remapping via medoid hashing *and* an Adjusted Rand Index continuity check, and the EWC implementation's correct precision-weighted multi-task Fisher consolidation are all above the bar I'd expect from a first-round HRI/CoRL submission. I'll say more about specific strengths inline where relevant, but wanted that on record first.

### 1. [Critical] Two of your "distinct" recipes exceed your own recipe/preference disambiguation threshold

Your spec (and `memory.py`'s `Disambiguator.classify`) uses Jaccard similarity between action sequences to decide whether a new demonstration is (a) a preference variant of a known recipe or (b) a genuinely new recipe, at a threshold of `jaccard_threshold = 0.95` (`models.py` `Config`). I was suspicious of this threshold given how similar some of your `RecipeGenerator` methods look by eye (`environment.py`), so I ran your actual `jaccard()` function (`memory.py`) over all C(30,2)=435 pairs of your 30 recipes:

```
jaccard  len_a  len_b  recipe_a / recipe_b
0.9583     23     24  tomato_soup / tomato_garlic_soup   <<< ABOVE jaccard_threshold=0.95
0.9259     25     27  tomato_onion_soup_v1 / seasoned_mixture_soup
0.9000     18     20  banana_strawberry_fruit_bowl / yoghurt_fruit_bowl
```

`tomato_garlic_soup` is `tomato_soup` plus one `season(tomato, garlic, ...)` action and a reordered `turn_on(stove)`, nothing else. Under your own classifier, at **0.9583 > 0.95**, if a human demonstrates `tomato_garlic_soup` while the robot has already learned `tomato_soup` (or vice versa, in either order), your Disambiguator's default behavior is to classify it as a **preference variant of the existing recipe**, not a new recipe. That's not a hypothetical edge case — it's your own recipe library, and the two recipes in question are semantically different dishes that you clearly intended to be distinct entries in `recipe_library()`.

Two more pairs sit at 0.90–0.93, close enough that I'd call them "in the blast radius" even though they're nominally below threshold — `score_partial` (used for online, prefix-based classification) uses a *different* formula (0.6·set-overlap + 0.4·τ, vs. `jaccard()`'s 0.7/0.3 full-string/type blend — see Finding 3), so a pair that clears 0.95 on the offline metric is not guaranteed to clear whatever the online metric's effective threshold is, and I did not exhaustively verify the online path stays safe on these pairs.

**Why it matters for the paper:** this directly corrupts the thing your paper is about. If `tomato_garlic_soup` gets folded into `tomato_soup`'s variant library, the recipe prototype for "tomato soup" now blends garlic-seasoned and unseasoned demonstrations under one recipe identity, the TaskSignature's partial-order and terminal-state statistics get muddied, and — more importantly for the review — this happens **silently**. There's no error, no warning, just a wrong classification that degrades whatever downstream numbers depend on correct recipe identity.

**How often would this actually bite in your reported experiments?** Since `n_recipes = 6` (`EvaluationConfig`) and recipes are sampled via `shuffled_recipe_builders(seed)` (a uniform random 6-subset of the 30), I estimated the co-occurrence probability directly (200,000-trial Monte Carlo over your actual recipe list, matches the closed-form C(28,4)/C(30,6) ≈ 3.45%):

```
P(tomato_soup AND tomato_garlic_soup both drawn in a random 6-recipe sample) ≈ 3.4%
P(at least one of the three risky/near-threshold pairs above co-drawn)       ≈ 10.0%
```

Ten percent of your seeds drawing at least one confusable pair is not negligible, especially if you're running the modest seed counts I'd guess from the CLI defaults (see Finding 10). This adds *unexplained variance* to your results that has nothing to do with the research question and everything to do with recipe-library design — exactly the kind of thing that shows up as "unusually low outlier seed" in a table and that no one investigates.

**Fix, in order of effort:**
1. Cheapest: add a startup assertion in the evaluation harness — after `select_recipe_builders(seed, n_recipes)` picks its subset, compute all pairwise `jaccard()` scores within *that* subset and hard-fail (or resample) if any pair exceeds `jaccard_threshold - margin` for some safety margin (I'd suggest 0.10). This turns a silent corruption into a loud, fixable one.
2. Better: differentiate `tomato_garlic_soup` (and the other two flagged pairs) from their siblings with more than one action's worth of difference — e.g., a distinct prep step, not just a seasoning addition — so the recipes are robustly separable regardless of threshold.
3. Most rigorous: report the full pairwise-Jaccard distribution across your recipe library in the paper (even just as a supplementary histogram), explicitly showing your 0.95 threshold has margin against the *hardest* real pair, not just against the average pair. This is a two-line script (I've essentially already written it for you above) and it preempts exactly the question I'd ask as a reviewer: "how did you pick 0.95, and does it actually separate your recipes?"

### 2. [High] The "12-preset preference library" is behaviorally much smaller than it looks — and this is currently undocumented

`preferences.py` composes preference variants by reordering (never inserting/deleting) actions along 8 axes, with 12 named presets. I wanted to know how many of the 30×13 = 390 (recipe, preset) materializations actually produce a behaviorally distinct action sequence, so I ran `materialize_with_report` over the full cross product:

```
Total (recipe, preset) materializations: 390
Materializations with ≥1 failed_axes (axis transform silently no-op'd): 172  (44%)
Collision events (this preset's ordering duplicates an earlier preset, same recipe): 119  (30.5%)

Worst recipes:      smoothie, yoghurt_smoothie          — 7/13 distinct orderings
Best recipes:  meat_mushroom_skillet, garlic_chicken_salad — 11/13 distinct orderings
No recipe achieves 13/13.
```

To be fair to the code: this is *detected*, not silently corrupting your data. `evaluation.py`'s `distinct_pairs_for_recipe` and `_shared_preference_matrix` do check for duplicate orderings and either skip them or mark `ordering_is_distinct=False` / `duplicate_of_preference`. That's good engineering and I want to credit it — my initial worry reading `preferences.py` in isolation was that this would silently corrupt the "distinct preference" ground truth used for stratifying results, and it doesn't, because someone already thought about this.

What's still a problem: **a third of your nominal preference space is behaviorally inert for a given recipe, and this isn't reported anywhere I can see.** This matters for two reasons:
- **Framing risk.** If a table or claim says something like "12 preferences × 6 recipes," a reviewer who reads your code (or, now, this review) will note that the *effective* number is closer to 8-9 per recipe on average, and worse for simple recipes like the smoothies. Report the effective count.
- **The 44% failed-axes rate is worth diagnosing, not just accepting.** Some of this is certainly genuine — a salad has no cooking step, so `p9_deferred_cook_start` is inherently a no-op for it, and that's a real domain constraint, not a bug. But `_move_matching_block` / `_move_one_action` (I didn't have time to fully trace every axis's search strategy, but from what I read it's a greedy, order-dependent pipeline applied in a fixed `AXIS_ORDER`, not an exhaustive or constraint-solver search) could plausibly be *failing to find* a valid reordering that exists, rather than correctly determining none exists. I'd recommend auditing a sample of `failed_axes` cases by hand — if a meaningful fraction turn out to be search-algorithm limitations rather than genuine infeasibility, your effective preference-diversity numbers are artificially suppressed by an implementation shortcut, and that's fixable (try axis transforms in more than one order, or backtrack) rather than a fundamental limit worth reporting as such.

**Fix:** report `n_distinct_orderings / n_presets` per recipe (min/mean/max) as a supplementary table, and do the failed-axes audit above. Both are cheap and materially strengthen the honesty of the "we test 12 preferences" framing.

### 3. [High] The system has at least six independently hand-tuned scoring formulas, most invisible to `Config`

Your `Config` dataclass (`models.py`) has on the order of 90 fields, which is already a lot of surface area, but that's not actually my concern — a rich config is fine and often necessary. My concern is that a comparable number of *equally consequential* weights are **not** in `Config` at all; they're bare float literals inside formulas, some sitting in the exact same function as sibling terms that *are* configurable, which looks like an oversight rather than a decision. I found (there may be more; this is not exhaustive):

| Location | Formula | Configurable? |
|---|---|---|
| `models.py`, `NGramMarkov.predict` | `p = 0.6·p_state + 0.4·p_ngram` | No — hardcoded |
| `memory.py`, `jaccard()` | `0.70·full_string_jaccard + 0.30·type_jaccard` | No — hardcoded (coincidentally matches `cfg.recipe_match_token_weight`/`precedence_weight`, which are used *elsewhere*, in `posterior.py`'s `recipe_match`, and are not the same computation) |
| `memory.py`, `Disambiguator.score_partial` | `0.6·set_overlap + 0.4·τ_order` | No — hardcoded |
| `adaptive_agent.py`, `_online_commit_confidence` | 7-term weighted blend (`0.46, 0.18, 0.20, 0.08, 0.04, 0.04, 0.06`) + two ad hoc override branches (`≥0.99 → floor 0.90`; `not full_lib → ceiling 0.40`) | No — hardcoded, plus special-cased overrides on top |
| `adaptive_agent.py`, `_posterior_blend_strength` | Three increments (`0.15, 0.10, 0.10`) *in the same function* as sibling terms that **do** read from `cfg.posterior_assist_agreement_bonus` / `cfg.posterior_assist_disagreement_penalty` | Partially — some terms configurable, others not, in the same formula |
| `posterior.py`, `PreferencePrototypeLearner` clustering | 5 thresholds (`MATCH_THRESHOLD=0.70`, `NOVELTY_MAX_SIMILARITY=0.45`, `NOVELTY_ENTROPY_MIN=0.60`, `TAU_PREF_CLUSTER=0.5`, `AXIS_SPLIT_THRESHOLD=0.35`) | Module-level constants with constructor override params, but I could not confirm from `adaptive_agent.py`'s instantiation sites that `Config` values actually get threaded through to override these — worth double-checking |

**Why it matters for the paper:** this is a reproducibility and rigor concern, not (as far as I can tell) a correctness one. A reviewer who notices `_online_commit_confidence`'s 7-weight formula, with two more special-cased overrides bolted on, is going to ask "how were these tuned, and on what data?" If the answer is "hand-tuned while debugging against the same scenarios used for the final results," that's a soft form of overfitting to your own test set that no amount of good algorithm design elsewhere will excuse in review. Even if that's *not* what happened, the paper currently can't demonstrate it, because these numbers aren't tracked as named, reportable hyperparameters.

**Fix:** audit for bare float literals inside scoring/blending formulas across the codebase (I'd start exactly where I found them, above), hoist all of them into `Config` with descriptive names following your existing convention (`online_commit_*`, `posterior_blend_*`, etc.), and then run — and report — a sensitivity sweep over at least the ones with the most decision-theoretic leverage (`_online_commit_confidence`'s weights and the two override thresholds are my top pick, since they gate whether the system trusts an online recipe-identity switch at all). You don't need to sweep all ~15+ of these exhaustively for the paper; even a "we varied X in [low, default, high] and the headline result is stable" appendix table for the two or three most load-bearing ones would preempt the objection.

### 4. [High] "Bounded memory" is true for the training set, not for the full identity registry — and the unbounded part is on the hot path

Your spec's memory story (adaptive decay and pruning) is implemented carefully and — I want to be clear — it *works* for what it is scoped to: `DecayManager.active` (the set actually used to fit the IRL/Markov heads) excludes temporally decayed variants, but it has no absolute variant-count cap. `VariantMemory` (`memory.py`) also tracks every (recipe, preference-variant) pair ever demonstrated for identity/reentry purposes; pruning only moves an entry from `DecayManager.active` to `DecayManager.pruned`, while `VariantMemory.variants` keeps growing for the lifetime of the agent. That's a deliberate and reasonable design choice (you need it for pruned-variant reentry recognition), but it means:

- `_online_commit_confidence` (`adaptive_agent.py`) calls `self.disambig.classify(prefix, [v])` **once per variant in the full, ever-growing registry**, on every online commit decision — and does so in a loop that constructs a fresh `Counter` for the prefix on every single call, rather than batching (`classify()` already loops internally over whatever library you hand it — calling it N times with library size 1 instead of once with library size N is strictly wasted work).
- This means the **per-step compute cost of your online commit logic grows at least linearly with total lifetime demonstrations**, not with the active/bounded set size.

**Why it matters for the paper:** your headline claim is explicitly "adaptation can continue indefinitely" (framing doc) in contrast to closed-set prior work. That claim is about *behavior* (does the system keep working correctly), but a skeptical reviewer will also ask about *cost* — if indefinite adaptation means indefinitely growing per-step latency, that's a real caveat to the "indefinite" framing that the paper should either scope around explicitly or address. Right now nothing in the eval (as far as I read) stress-tests this: `deployment_events = 80` with `n_recipes = 6` is nowhere near long enough to reveal an asymptotic scaling problem.

**Fix:** (a) batch the disambiguator call in `_online_commit_confidence` — one call with the full library, not N calls with a singleton library each; cheap, immediate. (b) Add one long-horizon scalability run (many hundreds of sessions, more distinct recipes cycling through than currently tested) and plot wall-clock-per-commit or wall-clock-per-retrain against total lifetime demonstrations, not just against active-set size. If it's flat, that's a good, easy, reassuring figure. If it's not, better to know and scope the claim now than have a reviewer find it. (c) At minimum, explicitly state in the paper that "bounded memory" refers to the training/replay set, and that the identity registry is separately either unbounded or (if you add a cap) bounded by a stated mechanism.

### 5. [Medium] `pruned_influence_audit`'s tolerance is doing undocumented work

I like this audit (`adaptive_agent.py`) — it re-fits a fresh IRL/Markov head from active-only data and compares against the live, production-fitted heads, plus compares live conditioned-frontier predictions against an active-only recomputation, both gated by a tolerance (`active_only_audit_tolerance = 5e-2` default). This is a real, non-tautological check (I confirmed the Welford normalizer resets per fit, per your own code comment, so it's not just comparing a model against itself).

The issue: both the "live" fit and the "reference" fit draw from the *same* mutable `cfg.rng` (a single `np.random.default_rng` stored on `Config` and shared across everything that consumes randomness within one agent's lifetime — I traced this and confirmed each **baseline** gets its own freshly-seeded `Config` via `base_config()`, so this is *not* a cross-baseline contamination problem, only an intra-agent one). Since `MaxEntIRL2.fit` draws random init weights when not warm-started, and the audit's reference fit necessarily happens *after* the real fit has already advanced the shared RNG state, the two fits are not bit-for-bit comparable even with zero leakage — which is presumably why the tolerance is 5e-2 and not exact equality. That's a reasonable design, but it's currently unstated, and a reviewer (or you, six months from now) has no way to tell "this tolerance absorbs benign optimization noise" from "this tolerance is loose enough to also hide a real 4%-magnitude leak."

**Fix:** either (a) seed the reference fit independently/deterministically so it doesn't depend on shared-RNG state at time of call, tightening what the tolerance needs to cover, or (b) explicitly calibrate the tolerance: run the audit once with a deliberately-introduced leak (e.g., don't reset the normalizer, or reuse a stale theta) and report what magnitude of leak the current 5e-2 tolerance would and wouldn't catch. Either is a half-day of work and turns "we chose 5e-2" into a defensible number. Also, the audit samples only 4 prefix lengths per demo, capped at 24 total (`max_prefixes`) — fine for a cheap per-step regression check, but I'd run one *exhaustive* pass (all prefixes, uncapped) at least once for the paper's own verification, separate from the lightweight version used throughout the main experiment loop.

### 6. [Low] Small correctness/robustness items

- `representations.py`: `task_signature_from_tokens`'s signature uses `Optional[Mapping[str, str]] = None` but `Optional` is never imported (only `Dict, FrozenSet, List, Mapping, Sequence, Tuple` are). This doesn't crash today because `from __future__ import annotations` defers evaluation, but it will break under `typing.get_type_hints()`, most static type checkers, and some IDE tooling. One-line fix.
- `representations.py`, `apply_transition_vector`: length-mismatched vectors are a documented silent no-op ("by design... synthetic mock vectors... production callers always feed full-length vectors"). I don't think this is wrong given your fixed global feature vocabulary, but a silent no-op on a shape mismatch is a landmine for future-you if the feature vocabulary ever becomes recipe-dependent or grows — I'd at least log/counter a warning on the no-op path rather than have it be perfectly silent, since the cost of adding that is near zero and the cost of it firing unnoticed in three months is a very confusing debugging session.
- `role_from_transition` (`representations.py`): assigns exactly one role per action step via a fixed priority order (appliance-toggle > serve > wash > cut/season > cook > add-to-container > position-only). For any action whose state delta spans two categories simultaneously, the higher-priority category wins and the other signal is dropped for role-bigram/trigram purposes. I did not find evidence this actually happens in your current action set (most actions look single-effect), so I'm flagging this as a "watch for it if you add compound actions later" rather than a current bug.

---

## Part II — Evaluation & Scenario Engineering

This is the part of the project I'd scrutinize hardest as a reviewer, since it's what determines whether your numbers mean what you say they mean. Overall verdict: the three-scenario design (`ladder_heterogeneous`, `ladder_homogeneous`, `ladder_deployment_random`) is more thoughtful than what I usually see in this literature, and I'll say why below, but it has a few specific gaps that would need to be closed before I'd trust the headline numbers.

### What's genuinely well designed

- **The heterogeneous/homogeneous ladder split is a real, deliberate experimental control, not two redundant conditions.** `build_ladder_heterogeneous` assigns `matrix[recipe][(recipe_idx + rung) % len(matrix[recipe])]` — a Latin-square-style cyclic offset so different recipes sit at different points in their own preference sequence at the same nominal "rung." `build_ladder_homogeneous` instead searches (`_shared_preference_matrix`, via `itertools.combinations`) for one preference combination that is simultaneously distinct-orderinged across *all* selected recipes, and gives every recipe the *same* preference at a given rung. That's a genuine 2-condition design: heterogeneous stresses realistic, decoupled preference cycling; homogeneous isolates whether a specific preference behaves the same way when transplanted across recipes — which is a direct, controlled test of your cross-recipe transfer claim, not just an incidental byproduct of it.
- **`build_deployment_random`'s event mixture is directly traceable to your five research hypotheses.** New-recipe (10%), preference-shift (28%), cross-recipe transfer (22%), reentry (14%), forced observation (6%), routine reuse (remainder ≈20%) map cleanly onto `known_recipe_new_preference_adaptation`, `cross_recipe_transfer`, `selective_forgetting_reentry`, and `direct_retrieval_control` — the same tags that show up in `paper_hypothesis_views` at the very end of `summarize_stream`. I also want to flag the comment at the transfer-probe fallback (`"Do not relabel a routine reuse as transfer"`) — that's exactly the kind of self-discipline that prevents metric contamination, and it's good to see it was a deliberate decision, not an accident.
- **The clairvoyant memory oracle (`_apply_clairvoyant_memory_pruning` / `CLAIRVOYANT_MEMORY_ORACLE`) is a genuinely good idea and I'd keep it front and center in the paper.** Pruning active memory using perfect foreknowledge of which recipes recur later gives you an honest, non-deployable ceiling on how good *any* memory-management policy could be — which lets you report "gap to oracle," not just "beats naive baseline X." Clearly labeled (`CLAIRVOYANT_REFERENCE_TAG = "dashed_reference_not_deployable"`, an explicit leakage warning string), which tells me you're already planning to render it as a dashed reference line rather than a competing bar — good, keep doing that visually, and say explicitly in the caption that it's an oracle, every time it appears.

### 7. [Critical] The two diagnostics that support your core "selective forgetting without catastrophic forgetting" claim are **off by default**, and the summary functions produce a plausible-looking but vacuous result when they're off

`EvaluationConfig.frozen_eval_period` and `active_only_audit_period` both default to `0`, and the guard is `if config.active_only_audit_period > 0 and ...` / `if config.frozen_eval_period > 0 and ...` (`evaluation.py`). At `0`, neither ever fires. I checked what the corresponding summaries produce when fed an empty row list, since that's what happens by default:

```python
active_only_audit_summary([])
# -> {"n_audits": 0, "n_available": 0, "n_primary_contract_failed": 0,
#     "primary_contract_failure_rate": 0.0, "live_prediction_max_l1": 0.0, ...}
```

Read in isolation — which is exactly how a number gets read once it's copied into a table or a bar chart — `"primary_contract_failure_rate": 0.0"` and `"live_prediction_max_l1": 0.0` look like "we audited for memory leakage and found none." What they actually mean, at the default config, is "we ran zero audits." The same applies to `frozen_summary(stream.frozen_rows)` when `frozen_eval_period=0`: an empty checkpoint dict, not a benign one.

**This is the single most important finding in this review**, because `pruned_influence_audit` and the periodic frozen-benchmark are precisely the mechanisms that would let you *prove* your central "selective forgetting, not catastrophic forgetting, and no leakage from pruned data" claim, and they're implemented well (see Finding 5 and the Part I strengths note) — it would be a shame to have built the right verification machinery and then have it silently not run for the numbers that ship.

**Fix, immediately:**
1. First, check what period values were actually used to generate any numbers you're currently planning to report. If it was 0, those numbers need to be regenerated with the audits turned on before they mean anything.
2. Change the defaults to something like `active_only_audit_period=1` (or whatever cadence is affordable — even every 5-10 events is far better than never) and `frozen_eval_period` to a sensible non-zero cadence, so "off" has to be an explicit opt-out rather than the silent default.
3. Make the summary functions distinguish "ran and passed" from "didn't run" at the type level, not just via a `n_audits` field a reader has to remember to check — e.g., return `None`/a distinct `"not_run"` status when the input is empty, so it fails loudly if someone forgets to check `n_audits` before reading `primary_contract_failure_rate`.
4. Anywhere `primary_contract_failure_rate` or `live_prediction_max_l1` appears in a table or figure, display `n_audits` immediately next to it. Non-negotiable given what I just found.

### 8. [High] "Reentry" probes aren't actually conditioned on the target having decayed out of active memory

`build_deployment_random`'s reentry branch does `pair = rng.choice(seen_pairs)` — uniform over *every* previously-demonstrated pair, tagged with `hypothesis_tags=["selective_forgetting_reentry", "retention_after_interference"]` regardless of whether that specific pair is currently active or pruned. If the sampled pair hasn't actually decayed out yet, this event is a normal "routine reuse" wearing a "reentry" label — it doesn't test what its tag says it tests.

The good news: your agent's own online classification *does* know the true answer — `Classification.kind == "reentry_from_pruned"` exists and gets computed during the episode regardless of what the scenario generator intended. So the ground truth needed to fix this is already being computed; it's just not being used to gate or stratify the "reentry" metric as far as I can tell from `summarize_stream`/`_group_metrics`, which group by the scenario's `condition`/`hypothesis_tags` tags, not by the agent's actual `classification_kind`.

**Why it matters:** if your "reentry accuracy" number in the paper is computed over the scenario-tag group rather than filtered to genuinely-pruned cases, it's diluted by an unknown, seed-dependent fraction of trivially-easy "reentry" events that were never actually forgotten — which would make your selective-forgetting-and-recovery story look stronger (or noisier) than it actually is, in a direction you can't currently quantify.

**Fix:** two changes, both should be easy given the data you already collect. (a) In the metrics/rows, add an explicit boolean derived from `classification_kind == "reentry_from_pruned"` at the time of the probe, and stratify the reentry results by it — report "reentry from genuinely pruned state" and "reentry while still active" as separate numbers, not one blended one. (b) Consider biasing `build_deployment_random`'s reentry sampling toward pairs likely to have decayed (e.g., prefer `seen_pairs` entries whose recipe hasn't recurred in a while, using the same recency logic the DecayManager itself uses) so the *scenario* actually targets the interesting case more often, rather than relying entirely on post hoc stratification to find the signal in a mostly-easy sample.

### 9. [Medium] Realized event-type frequencies will drift from configured probabilities, and this isn't reported

`deployment_new_recipe_prob=0.10` with only `n_recipes=6` total and `deployment_onboarding_recipes=3` onboarded upfront means the "new recipe" pool (3 remaining recipes) is exhausted after just a few hits of that branch, out of `deployment_events=80` total. Once exhausted, `new_recipe()` returns `None`, the code doesn't `continue`, and execution falls through into the *next* branch (preference-shift) — meaning, for probably the large majority of an 80-event, 6-recipe stream, rolls that were nominally meant to produce a "new recipe" event silently become preference-shift events instead. This isn't a bug exactly (it's a sensible fallback, and it's not mislabeled — the resulting event correctly gets `event_type="deployment_preference_shift"`, not a fake "new recipe" tag), but it does mean the *configured* probabilities in `EvaluationConfig` are not the *realized* frequencies in your actual event stream, and I'd bet the gap is large given how quickly the recipe pool empties. **Fix:** log/report the realized event-type distribution per stream (you already compute `_support_counts` — just surface it next to the configured probabilities in the paper or appendix) so a reader can see what was actually tested, not just what was configured.

### 10. [High] No visible statistical methodology, and the harness defaults to a single seed

`EvaluationConfig.seeds: Tuple[int, ...] = (1337,)` and the CLI default (`parse_args`) is also `--seeds 1337` — a single seed unless explicitly overridden. Scenario construction is seed-dependent in multiple places (which 6 of 30 recipes get selected, the entire `deployment_random` event sequence), so a single seed gives you one draw from a distribution, with no way to distinguish a real effect from that draw's idiosyncrasies — and Finding 1 shows those idiosyncrasies aren't negligible (a ~10% chance any given seed contains a confusable recipe pair).

I want to be fair here: I can't tell from the code alone how many seeds you actually used to generate your current results — it's entirely possible you already run a sweep and just haven't wired the higher count into the checked-in default. But the default itself is worth fixing regardless, because defaults are what get used when someone (including you, in six months, or a reproducer) runs the script without reading every flag.

The good news is your design makes rigorous statistics unusually *cheap* to add: because `plan.seed` fixes the identical scenario plan across every baseline compared at that seed (`run_event_stream_for_baseline` builds each baseline's agent from `base_config(plan.seed, config)` against the *same* `ScenarioPlan`), your baselines are naturally **paired** by seed. That means you don't need independent-samples tests with their weaker power — a paired test (paired t-test, or Wilcoxon signed-rank if you don't want to assume normality) across, say, 15-20 seeds, comparing your system to each baseline on the same seed-by-seed scenario draws, is both the statistically correct choice and easy to compute from data you're already collecting. I'd make this a standard part of the reported results (mean ± SE or a CI band on every plotted line, plus a paired significance test for the headline comparisons), not an afterthought.

**Fix:** bump the default seed count substantially (I'd start around 15-20 for the final paper runs, more if compute allows, given how cheap the recipe-collision check in Finding 1 shows single-seed variance can be), and report paired statistics across seeds for every headline comparison, since your architecture already gives you paired samples for free.

### 11. [Medium] `oracle_gap_summary` averages across incommensurable metrics

`_oracle_row` computes `regret_to_clairvoyant = max(0, oracle_value - baseline_value)` (sign-adjusted per metric direction) as a **raw, unnormalized difference**. `oracle_gap_summary`'s `by_baseline[...]` then averages `regret_to_clairvoyant` across **all rows for that baseline**, and the grouping key is only `baseline` — not `(baseline, metric)`. Since a single baseline's rows span metrics on very different natural scales (`human_correction_rate` differences live in [-1,1]; `mean_nll_per_robot_turn` differences are in nats and can be much larger; `testing_normalized_interaction_cost` differences are ratio-scale) — `mean_regret_to_clairvoyant` for a given baseline is literally averaging probability-point gaps together with NLL-nat gaps together with time-ratio gaps. That number doesn't have a clean unit or interpretation, and I'd be cautious about anyone reading it as "baseline X is on average Y regret away from the oracle" — it isn't, not in any single unit.

**Fix:** group `oracle_gap_rows` by `(baseline, metric)`, not just `baseline`, before averaging, and report per-metric oracle-gap tables/plots (which `oracle_gap_rows` already has all the data for — this is a grouping change, not a new computation). If you want one combined "distance to oracle" scalar for a summary slide, normalize each metric's regret first (e.g., as a fraction of the oracle's own value, or min-max/z-scored across baselines) before combining — don't combine raw-unit differences.

### 12. [Low] Minor scenario-engineering notes
- `distinct_pairs_for_recipe` picks presets *in list order* until it hits `min_pairs`, skipping duplicates as it goes. This means different recipes in the same `ladder_heterogeneous` run can end up drawing from different subsets of the named 12 presets to fill their rungs (recipe A's "rung 3" might be `p3_clean_eager`, recipe B's might be `p5_prep_stage_clean`, if `p3` collapsed for B). That's fine — it's exactly what "heterogeneous" should mean — but I'd make sure the paper's prose is explicit that "rung number" is a *position* in a recipe-specific sequence, not a fixed global preference identity, since `ladder_homogeneous` (where rung really does mean a fixed shared preference) exists right next to it and a reader could conflate the two conditions' semantics if the distinction isn't stated plainly.
- I'd double check whether `assist_episode`'s recursive `commit=False` self-call (used to compute `post_commit_frozen_top1`/`commit_retrain_delta_top1`) is worth its cost at full scale — it re-runs the entire prediction pass a second time, plus a full `agent.snapshot()` deep-copy, for every single committed assist episode. Not a correctness issue (I traced the snapshot/restore discipline and it correctly avoids contaminating the real trajectory), just a "make sure this is buying you enough signal to justify roughly doubling your per-episode compute" sanity check, especially once you're running the longer/higher-seed-count experiments Finding 10 asks for.

---

## Part III — Metrics, and What Feeds the Graphs

As noted up top, I don't have your plotting code, so I can't comment on axis scaling, error bars as *drawn*, or chart type. Everything here is about the numbers your plotting code would presumably consume from `summarize_stream()`/`aggregate_episode_metrics()`. If a graph is downstream of a broken number, the graph is broken regardless of how it's drawn, so I'd treat this section as blocking for any figure that touches top-k accuracy or oracle-gap.

### 13. [Critical, confirmed by execution] `live_topk` is computed with a different, bug-prone aggregation formula than `live_top1` — and the discrepancy is real, not theoretical

`aggregate_episode_metrics` (`evaluation.py`) computes the two headline prediction-accuracy numbers differently:

```python
# live_top1: a clean pooled ratio — correct.
robot_turns = sum(row["hrc_robot_turn_count"] for row in rows)
live_top1 = safe_div(sum(row["hrc_robot_correct_count"] for row in rows), robot_turns)

# live_topk: a per-episode weighted average, with a weight floor — NOT equivalent.
weights = [max(1.0, row["hrc_robot_turn_count"]) for row in rows]
live_topk = sum(row["live_topk"] * w for row, w in zip(rows, weights)) / sum(weights)
```

The `max(1.0, ...)` floor means a **zero-robot-turn row** (any observe-mode episode — `observe_episode` hardcodes `hrc_robot_turn_count=0` and `live_topk=0.0`) still contributes weight **1.0** to the denominator while contributing **0** to the numerator, silently pulling `live_topk` down whenever observe-mode rows are mixed into the same aggregation as assist-mode rows. `live_top1` doesn't have this problem, because pooling by true turn-counts means a 0-turn row contributes 0 to *both* numerator and denominator — it's correctly invisible.

I didn't just read this and infer a bug — I ran `aggregate_episode_metrics` on a synthetic but realistic row set (10 observe-mode episodes + 10 assist-mode episodes with a true 80% top-1 *and* top-k hit rate, so the two numbers should agree):

```
live_top1 (pooled, correct):                          0.800
live_topk (weighted-avg with dilution, as shipped):    0.727
expected live_topk if computed like top1:              0.800
```

That's not a rounding difference — it's a ~9% relative understatement, and it will scale with however many observe-mode episodes get mixed into whatever row set is being aggregated. This is exactly the situation in `summarize_stream`'s `"all_episodes"` view (`aggregate_episode_metrics(stream.episode_rows)`, unfiltered by mode) and, more importantly, in `"per_hypothesis"` (`_group_metrics(stream.episode_rows, "hypothesis_tags")`, also unfiltered by mode) — which feeds directly into `paper_hypothesis_views`, a dict whose name I take as a strong signal it's meant to go straight into the paper. I traced one of its five keys concretely: `"direct_retrieval_control"` is attached both to onboarding *observe* events and to routine-reuse *assist* events (`build_ladder_heterogeneous`/`build_deployment_random`), so that specific hypothesis view **will** exhibit this dilution as shipped.

The good news, and I want to be precise about scope: `summarize_stream`'s `"assist_only"` view is computed from `assist_rows = [row for row in stream.episode_rows if row.get("mode") == "assist"]` — a clean, mode-filtered subset — so if that's what feeds your headline top-1/top-k accuracy figure, that specific figure is fine. But the bug is real, present, and reachable through at least the `"all_episodes"` view and the `"direct_retrieval_control"` hypothesis view, both of which are one dict access away from a table or plot.

**Fix:** make `live_topk`'s aggregation match `live_top1`'s — pool `hitsk`/turn-counts directly (you have `hrc_robot_turn_count` on every row; add an `hrc_robot_topk_hit_count` field alongside `hrc_robot_correct_count` if you don't already store raw hit counts per row, then `safe_div(sum(hitsk), sum(turns))`), rather than weight-averaging a pre-divided per-row ratio. This is a five-minute fix. Then, before finalizing any numbers already generated, re-run anything that touched `"all_episodes"` or `"per_hypothesis"` aggregates for top-k accuracy.

### 14. [Critical, confirmed by execution] Your own framing claims "equal action durations" — the shipped timing config isn't equal, and perfect assistance does not match human-only time

Your framing document says explicitly: *"the claim here isnt 'wall-clock speedup.' With equal action durations, perfect assistance matches human-only time, while errors add correction overhead."* That's a clean, correct way to justify `testing_normalized_interaction_cost` as your headline efficiency metric — **if** the durations are actually equal. I checked `DEFAULT_HRC_TIMING` (`hrc_simulation.py`):

```python
human_action_time: float = 4.0
robot_correct_action_time: float = 6.0   # 1.5x the human action time
robot_wrong_action_time: float = 2.0
human_correction_time: float = 8.0
```

They aren't equal — a *correct* robot action is modeled as taking 50% longer than the equivalent human action. I ran `run_alternating_hrc_episode` with a synthetic oracle predictor that is correct on every single turn (zero errors, the best case any method could ever achieve) over a 20-step sequence, using your actual timing config, to get the honest number rather than compute it by hand:

```
n_recipe_steps: 20, robot_correct_count: 10, robot_wrong_count: 0
hrc_total_time (perfect accuracy): 100.0
human_only_time: 80.0
normalized_interaction_cost at PERFECT accuracy: 1.25
```

**At 100% prediction accuracy — literally the best any system, including your oracle ceiling, could ever do — normalized interaction cost is 1.25, not 1.0.** The "matches human-only time" claim as currently written is not what the shipped timing config simulates; it's off by a fixed 25% floor that has nothing to do with prediction accuracy and everything to do with the robot being modeled as inherently slower per action.

This isn't necessarily wrong as a *modeling choice* — it's quite plausible that a real robot's physical action (motion planning, gripper actuation, safety-checked execution) genuinely takes longer than a human's practiced equivalent, and if that's the intended story, fine. What's wrong is the **mismatch between that modeling choice and the framing sentence that justifies your metric.** As written, a reader who takes "perfect assistance matches human-only time" at face value and then sees your normalized-interaction-cost plot sitting at, say, 1.3 for your full system, will read that as "worse than doing it alone," when it might actually be very close to the best achievable floor given the timing model. Without the floor stated explicitly, the metric is nearly impossible for a reader to interpret correctly, and it's an easy, obvious thing for a reviewer to catch and use to question your central "we're not claiming speedup, but we are claiming low overhead" framing.

**Fix — pick one:**
1. If "equal action durations" is the intended condition, set `robot_correct_action_time = human_action_time = 4.0` in the config actually used to generate results, so the code matches the claim.
2. If unequal durations (robot genuinely slower) is the intended, more realistic condition, **rewrite the framing sentence** to state the true floor, and — this is the important part for your figures — **compute and plot the perfect-accuracy floor as an explicit reference line** on every normalized-interaction-cost chart (exactly the number I derived above, 1.25 for this timing config and turn schedule, though note the true floor shifts slightly once real error rates are folded in because a wrong prediction changes who takes the *next* turn too — I'd generate it the same way I did above, from your own simulator with a synthetic oracle, rather than trust a hand-derived constant). Either way, don't let "1.0" implicitly stand in for "as good as unassisted" in a figure if 1.0 isn't actually achievable.

### 15. [Medium] EWC's baseline compute cost is ~2x the proposed system's, and I don't see this surfaced anywhere for comparison

`EWCAgent._fit_heads` (`baselines.py`) does a full primary `self.irl.fit(...)` call *and* a full auxiliary `task_irl.fit(...)` call (a completely separate MaxEnt IRL optimization, same iteration budget) every single retrain, purely to compute the new task's Fisher/theta anchor. You already track this carefully — `ewc_primary_estimated_flops`, `ewc_aux_estimated_flops`, `ewc_fisher_estimated_flops` are all separately recorded — which tells me you're aware compute cost varies by baseline. What I don't see is this actually being *used*: is there a compute-normalized comparison anywhere (e.g., accuracy per unit FLOP, or accuracy at matched wall-clock budget) alongside the raw accuracy comparison? Without one, a reader could reasonably ask whether some of your system's advantage over EWC would shrink if EWC were given the same compute budget your system uses elsewhere (prototype rebuilds, posterior updates, the `_online_commit_confidence` scan discussed in Finding 4). **Fix:** since you're already computing the FLOPs, add a compute-normalized view (even just a supplementary scatter of accuracy vs. estimated_flops per baseline) — this is a plot you can make from data you already have, and it preempts a fairness objection cheaply.

### 16. [Low] `commit_retrain_delta_top1` is a good diagnostic but needs a careful caption if plotted

`assist_episode`'s recursive re-evaluation (discussed in Finding 12) measures accuracy on the *exact same sequence just trained on* — closer to a memorization/consolidation check ("did committing this demo actually update the model the way we intended") than a generalization measure. That's a legitimate and useful thing to report, but if it ends up in a figure titled anything like "adaptation gain," I'd rename or caption it precisely (e.g., "immediate post-commit recall on the just-demonstrated sequence") so a reader doesn't read it as evidence of generalized preference learning, which it isn't measuring.

---

## Part IV — The Baseline Suite: What You Built vs. What You Run

This is worth its own section because the gap is large and, I think, the single highest-leverage fix in this whole review — most of what's "wrong" here is already *written*, just not *wired up*.

`baselines.py` defines **21 baseline/decay-manager classes**. I checked, by both direct string search across `evaluation.py` and by tracing every import and registry construction (there is no dynamic/reflection-based registration — I checked for `__subclasses__`, `getattr`, `importlib` patterns and found none), exactly what's reachable from the actual experiment harness:

```python
# evaluation.py, in full:
from .baselines import BASELINE_AGENTS, OracleCeilingAgent
...
registry = {"full": AdaptiveHRCAgent, "oracle": OracleCeilingAgent, **BASELINE_AGENTS}
```

`BASELINE_AGENTS` contains exactly 7 entries. Adding `"full"` and `"oracle"`, **9 of the 21 built classes are reachable.** The other **12 are dead code from the paper's perspective** — fully implemented, in some cases the most scientifically important ablation in the file, and never once instantiated by anything `evaluation.py` can call:

| Unused class | What it tests | Why I'd want it in the paper |
|---|---|---|
| `NoReplayAgent` | Trains only on the newest demo, no rehearsal at all | This is *the* textbook catastrophic-forgetting baseline — the cleanest possible demonstration of the problem your whole memory system exists to solve. Its absence is the one a reviewer notices first. |
| `BudgetMatchedUniformReplayAgent` | Full IRL/Markov/posterior predictor, uniform (not adaptively-weighted) replay, at the **same memory budget** as your system | This is the single most important ablation for your core claim. Without it, "our adaptive decay beats X" is confounded with "our system just has a different memory footprint than X." This isolates the *policy*, holding budget constant. |
| `OnlineEWCAgent` | EWC++ / Progress & Compress (Schwarz et al. 2018) — exponential-decay Fisher, explicitly for **non-stationary streams where old tasks become irrelevant** | Your own docstring for this class says it's designed for exactly your setting. The `EWCAgent` you *do* report (`"ewc"` in `BASELINE_AGENTS`) is the Kirkpatrick et al. 2017 formulation, which accumulates Fisher mass forever and is designed for a regime where *all* past tasks stay equally important — not your regime, where old preferences are supposed to fade. Reporting only the less-appropriate EWC variant, while a better-suited one sits unused in the same file, makes your EWC comparison look weaker for your baseline than it needs to, in a way I doubt is intentional but that a reviewer familiar with the continual-learning literature would flag immediately. |
| `AdaptiveDecayAgent` | Your exact adaptive-horizon decay mechanism, *minus* latest-preference-pin protection only — everything else identical to the full system | This is the clean, single-variable ablation of the "delayed latest-pin promotion" mechanism your framing document specifically calls out as a design contribution. Without it, that specific mechanism's contribution is unmeasured. (Note: I'd also rename it — "AdaptiveDecayAgent" reads like it could *be* the proposed method to someone skimming the code, when it's actually one specific ablation of it.) |
| `UniformWeightAgent` | Same active/pruned membership schedule as the full system, but ignores the *soft* decay weight at training time (hard in/out only) | Tests whether gradual down-weighting during the decay ramp earns its complexity over a simple cutoff at the same schedule. |
| `L2AnchorAgent` | Plain L2 anchor to previous theta (uniform "Fisher") | The natural, cheaper alternative to ask "do we need Fisher-weighting specifically, or would naive L2 anchoring do"? |
| `ProgressiveAnchorEWCAgent` | Single-task (sliding) anchor EWC | A useful middle point between `L2AnchorAgent` and full multi-task `EWCAgent`. |
| `IRLExperienceReplayAgent`, `IRLRecencyPrioritizedReplayAgent`, `RecencyPrioritizedReplayAgent` | ER variants using your full IRL/Markov/posterior predictor (not just BC) | `ExperienceReplayAgent` (the one you do report) is BC-only; these isolate "replay policy" from "predictor family," which the BC-based ER baseline conflates. |
| `FrequencyConditionedBigramAgent`, `MostFrequentNextAgent` | Additional weak floors | Minor gap — `BigramOnlyAgent` is already included, so this is lower priority than the rest of this list. |

To be clear about what I'm *not* saying: I'm not saying you need all 12 in the final paper — that's a lot of lines on a plot, and some of these (the frequency floors) are genuinely lower priority. But `NoReplayAgent`, `BudgetMatchedUniformReplayAgent`, `OnlineEWCAgent`, and `AdaptiveDecayAgent` specifically each answer a question your own framing document raises, they're already implemented and (from what I read) implemented carefully, and none of them currently produce a data point. That's the fix: register them in `BASELINE_AGENTS` (or a parallel dict if you want to keep the "default suite" small and add these as an "extended ablations" `--baselines` flag value), run them, and let the results tell you whether e.g. the budget-matched replay baseline actually closes most of the gap to your full system — if it does, that's important to know before submission, not after a reviewer asks for it in the rebuttal.

I'd treat this as the highest-value single change I found in the whole review: it's not new algorithm work, not new scenario design, just wiring code that already exists into the harness that already exists.

---

## Part V — Experiments/Figures I'd Add That Aren't Just "Fix Finding N"

Most of the highest-value additions I'd suggest are already captured as fixes above (re-run with the audits on — Finding 7; add the budget-matched/no-replay/EWC++/latest-pin-only baselines — Part IV; report the recipe-collision and axis-collapse rates — Findings 1-2; report paired statistics across ≥15 seeds — Finding 10). Two more I'd add that aren't fixes to something broken, just gaps:

**A calibration/reliability plot for the posterior's confidence score.** `posterior.py`'s joint (recipe, preference) posterior is a genuinely careful expert-combination construction (4 weighted experts, temperature-scaled, softmax-combined), but it's a *heuristic* scoring rule, not a probabilistic model derived from an explicit generative assumption — `recipe_match` and the compatibility scores aren't calibrated likelihoods, they're similarity scores treated as log-likelihoods. That's a completely standard and reasonable thing to do, but it means nothing currently guarantees that "confidence 0.75" corresponds to being right 75% of the time. This matters concretely because confidence *gates real decisions* in your system — `posterior_switch_min_confidence`, `online_commit_full_threshold`, the fast-path/hysteresis thresholds in the recipe-switching logic all fire off this number. A reliability diagram (bin predicted confidence, plot empirical accuracy per bin) or an ECE (expected calibration error) figure is a half-day addition using data you're already logging (you have `predicted`, `correct_top1`, and the confidence/entropy fields on every turn), and it would let you either confirm the thresholds are sane or catch that they're systematically miscalibrated before a reviewer asks "how did you pick 0.55 and 0.75?"

**Make the adaptive decay horizon itself the headline figure it deserves to be.** The mechanism I found most interesting reading this codebase is one you undersell in the framing doc relative to how much engineering went into it: `DecayManager.recipe_horizon_for` adapts each recipe's grace period based on the max of its recent reuse gaps (windowed at 3, floored at 6), which produces a genuinely non-obvious emergent property — a frequently-cooked recipe gets a *short* horizon (so its stale preference variants get pruned fast, since "the recipe is hot, but this particular variant clearly isn't the current one anymore"), while a rarely-cooked recipe gets a *long* horizon (since a big gap is normal for it, and you don't want to punish it for that). That asymmetry is, as far as I can tell, the actual scientific payload that distinguishes "adaptive decay" from "fixed decay" (which you do test — `FixedDecayAgent` is in `BASELINE_AGENTS`), but I don't see anywhere that the *horizon itself*, over time, per recipe, is plotted. A single figure — horizon value over the course of a `deployment_random` run, one line per recipe of differing reuse frequency, ideally next to the fixed-horizon baseline's flat line for comparison — would make the mechanism legible in a way that an accuracy-delta bar chart alone can't, and it directly visualizes the thing your ablation (`FixedDecayAgent` vs. full system) is supposed to be testing the *consequence* of.

---

## Prioritized fix list

Roughly in the order I'd tackle them — the first three are, in my view, prerequisites for trusting any number currently in the pipeline; they're also all cheap.

| # | Finding | Effort | Why it's this urgent |
|---|---|---|---|
| 7 | Turn on `active_only_audit_period` / `frozen_eval_period` (currently 0 = never runs) before regenerating any results | Trivial (config default) | Your core "no leakage / selective forgetting" claim currently has zero empirical backing if these ran at default | 
| 13 | Fix `live_topk` aggregation to pool hits/turns like `live_top1` does | ~5 min | Confirmed, reproducible bug reachable from `paper_hypothesis_views` |
| 14 | Reconcile the timing config with the "equal action durations → perfect assistance matches human-only time" claim, or compute and plot the true floor | ~1 hr | Affects the interpretation of every normalized-interaction-cost number/plot |
| 1 | Guard against/fix the `tomato_soup`/`tomato_garlic_soup`-class recipe collisions | ~1 hr (assertion) to ~1 day (redesign + report) | Silent, seed-dependent corruption of recipe identity; ~10% of seeds touch a risky pair |
| — | Register `NoReplayAgent`, `BudgetMatchedUniformReplayAgent`, `OnlineEWCAgent`, `AdaptiveDecayAgent` in the runnable baseline set and include them | ~1 hr wiring + compute time | These are your most important ablations for the memory-management story, and they're already written |
| 10 | Move off the single-seed default; report paired statistics across ≥15 seeds | Mostly compute time | Your paired-by-seed design makes this cheap to do correctly once you commit to it |
| 8 | Stratify "reentry" metrics by actual `classification_kind == "reentry_from_pruned"`, not just the scenario tag | ~1 hr | Otherwise the reentry number is diluted by an unknown fraction of non-reentry cases |
| 3 | Hoist the ~6+ hardcoded scoring-formula weights into `Config`; sensitivity-sweep the highest-leverage ones (`_online_commit_confidence`) | ~1 day + sweep compute | Reproducibility and pre-empting an overfitting objection |
| 11 | Group `oracle_gap_summary` by `(baseline, metric)` before averaging | ~30 min | Current cross-metric average has no clean unit |
| 2 | Report effective preference-diversity (distinct orderings / preset count per recipe) and audit `failed_axes` | ~half day | Framing honesty; cheap given the check is basically already written (see my script) |
| 4 | Batch the disambiguator call in `_online_commit_confidence`; add a long-horizon scaling run | ~2 hr batching, ~1 day for the run | "Indefinite adaptation" claim currently untested at the scale it claims |
| 5 | Decide/document the `pruned_influence_audit` tolerance calibration | ~half day | Currently an unjustified magic number on your key correctness check |
| 15 (EWC) | Add a compute-normalized (accuracy-vs-FLOPs) comparison view | ~2 hr, data already collected | Pre-empts a fairness objection on your strongest continual-learning baseline |
| A/B (Part V) | Calibration plot for posterior confidence; adaptive-horizon-over-time figure | ~1 day each | Turns your most interesting mechanism into legible evidence rather than an implicit claim |

---

## Closing note

I want to end where I started: this is a more careful piece of engineering than the issues list makes it look, and most of what I flagged is a matter of *finishing the wiring* on verification machinery you've already built (the audits, the extended baselines, the FLOPs tracking) rather than doing new fundamental work. If I had to compress this review into one sentence, it's: **you built the right instruments to prove your claims, and in a few important places they're not switched on** — the `live_topk` formula disagrees with `live_top1`, the leakage audit defaults to off, the timing model's floor doesn't match the framing sentence that explains it, and twelve of your twenty-one baselines (including the ones that would most directly test your central memory-management claim) never get called by the harness. Every one of those is fixable in days, not weeks, and fixing them before submission is far cheaper than fixing them in a rebuttal.
