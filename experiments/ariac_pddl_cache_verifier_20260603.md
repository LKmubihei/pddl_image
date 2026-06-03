# ARIAC PDDL-Cache Verifier Diagnostic

Date: 2026-06-03

## Method

Implemented `training/diagnose_cache_verifier.py`, a pure-evaluation cache
verifier for the existing H+640 placement checkpoint.

The script keeps the trained model as the legal-state candidate generator and
adds a non-parametric verifier:

```text
checkpoint support scores + object slots
-> top-K legal placement assignments
-> edge cache score + optional state-pattern cache score
-> reranked legal init state
```

Key design points:

```text
memory source: train split only
selection: train leave-one-out, current held-out train sample excluded
test: fixed selected hyperparameters, no test tuning
edge key: part slot, candidate slot, slot product, slot difference,
          base support-score z features, margin features, part/candidate type bits
edge buckets: at least table / region / support_part
state memory: gold state patterns and hard negative top-K legal state patterns
```

Default LOO grid excludes `edge_lambda=0` so the selected method cannot collapse
to the normal decoder. The normal decoder is reported separately.

## Main Results

| split | checkpoint | topK gold | selected config | normal EM/F1 | cache EM/F1 | changed | bad->good | good->bad |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 52/100 | H+640 d256 structured+CF | 0.9200 | k=3, beta=5, edge=0.2, state=0 | 0.7900 / 0.9603 | 0.7800 / 0.9591 | 1 | 0 | 1 |
| 100/52 | H+640 d256 structured+CF | 0.9423 | k=3, beta=5, edge=0.2, state=0 | 0.8462 / 0.9723 | 0.8462 / 0.9723 | 0 | 0 | 0 |
| 40/100 | H+640 d256 structured+CF | 0.8200 | k=3, beta=5, edge=0.2, state=0 | 0.6300 / 0.9190 | 0.6300 / 0.9213 | 2 | 0 | 0 |
| 30/100 | H+640 d256 structured+CF | 0.7900 | k=3, beta=5, edge=0.2, state=0 | 0.6100 / 0.9201 | 0.6100 / 0.9201 | 0 | 0 | 0 |

## Diagnostic Variants on 52/100

| variant | selected config | normal EM/F1 | cache EM/F1 | changed | bad->good | good->bad |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| coarse kNN-logit | k=3, beta=5, edge=0.2, state=0 | 0.7900 / 0.9603 | 0.7800 / 0.9591 | 1 | 0 | 1 |
| part-kind bucket | k=3, beta=5, edge=0.2, state=0 | 0.7900 / 0.9603 | 0.7800 / 0.9591 | 1 | 0 | 1 |
| Tip-style logsumexp | k=3, beta=5, edge=0.2, state=0 | 0.7900 / 0.9603 | 0.7900 / 0.9603 | 0 | 0 | 0 |
| forced stronger edge | k=3, beta=5, edge=0.5, state=0 | 0.7900 / 0.9603 | 0.7800 / 0.9591 | 1 | 0 | 1 |
| forced state memory | k=3, beta=5, edge=0.2, state=0.5 | 0.7900 / 0.9603 | 0.7900 / 0.9603 | 0 | 0 | 0 |

## Interpretation

This implementation tested the intended case-based verifier, but the current
object-slot/cache feature space does not provide a useful top-K disambiguation
signal.

The cleanest evidence is:

```text
52/100: top10 oracle = 0.9200, cache = 0.7800
100/52: top10 oracle = 0.9423, cache = 0.8462 unchanged
40/100: cache changes only bad-to-bad samples with gold not in top10
30/100: cache unchanged
```

The cache is not acting as a successful verifier. When it changes a top1 legal
state, it either hurts one already-correct sample (`picture_323`) or changes
samples whose gold state is not in the top10 candidate set.

A separate caveat: train LOO state-level selection is saturated because the
trained checkpoint predicts the training split exactly. This makes train LOO
poor at distinguishing cache weights. The script therefore excludes zero edge
weight by default and also reports forced stronger/state variants; neither
shows hidden positive signal.

## Conclusion

The result is a no-go for the current PDDL-cache verifier using model object
slots plus support-score/type features.

The original hypothesis was reasonable, but this test suggests the existing
slot feature space is not organized by reusable placement cases. The remaining
0.79 -> 0.92 gap is not closed by non-parametric retrieval over these internal
features. Reaching that oracle likely needs better observable grounding
features, stronger proposal/part-center evidence, or a different label signal,
not just a cache over current slots.

## Artifacts

```text
training/diagnose_cache_verifier.py
experiments/ariac_pddl_cache_verifier_52_100_20260603.md
experiments/ariac_pddl_cache_verifier_52_100_partkind_20260603.md
experiments/ariac_pddl_cache_verifier_52_100_tip_20260603.md
experiments/ariac_pddl_cache_verifier_52_100_strong_20260603.md
experiments/ariac_pddl_cache_verifier_52_100_state_forced_20260603.md
experiments/ariac_pddl_cache_verifier_100_52_20260603.md
experiments/ariac_pddl_cache_verifier_40_100_20260603.md
experiments/ariac_pddl_cache_verifier_30_100_20260603.md
```
