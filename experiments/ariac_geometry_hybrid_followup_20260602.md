# ARIAC Geometry / Occupancy / Hybrid Follow-up - 2026-06-02

Data and split are unchanged from the updated-label experiments:

```text
valid samples: 186
duplicate-part excluded: 34
usable samples: 152
train/test: 122/30
split_seed: 42
init_seed: 42
DINOv3: raw 448 dense tokens
```

## Implemented Changes

### Attention-Derived Geometry Features

`PaQModel` can now compute per-object soft geometry from query attention masks:

```text
center_x, center_y, spread_x, spread_y, entropy, peak
```

`BlocksworldSupportHead` can optionally append pairwise geometry features:

```text
top/support geometry
relative dx/dy
distance
soft horizontal/vertical overlap
soft contact
above signal
candidate-is-part flag
```

Enabled with:

```text
--support-geometry-type attention
```

### Token-Level Spatial Coordinate Features

The existing `--dinov3-add-coords` path was reused for 448-token features. A
coordinate cache was generated from the existing raw 448 cache, producing:

```text
features: [152, 784, 1290]
```

### Occupied/Clear Consistency Loss

Added:

```text
--occupancy-loss-weight
```

This trains:

```text
occupied(q) = exists p: p -> q
```

using a `logsumexp` over support scores for all active `p -> q`. It targets
false-clear and missed-stack errors such as `picture_54`.

### Hybrid Legal Decoder

Added:

```text
--hybrid-atom-decode-weight
```

The decoder still enumerates legal PDDL placement assignments, but can rerank
them with atom-branch log-likelihood:

```text
score(A) = placement_score(A) + lambda * atom_loglikelihood(derived_atoms(A))
```

## Results

All rows use `d_slot=256`, object queries, one relation layer, two-stage
support head, and `support_temperature=1.5`.

| model | EM | F1 | placement top1/top3/top10 | top1 error counts |
| --- | ---: | ---: | --- | --- |
| no-geo CE | 0.9000 | 0.9837 | 0.9588/0.9794/1.0000 | missed=1, loc=1, wrongpart=2 |
| no-geo structured | **0.9333** | **0.9898** | 0.9588/0.9897/1.0000 | missed=1, loc=1, wrongpart=2 |
| no-geo structured+CF | 0.9333 | 0.9878 | 0.9691/0.9897/1.0000 | missed=0, loc=0, wrongpart=3 |
| attention geometry CE | 0.9000 | 0.9796 | 0.9278/0.9794/1.0000 | missed=0, loc=3, wrongpart=3 |
| attention geometry structured | 0.9000 | 0.9797 | 0.9381/0.9794/0.9897 | missed=2, loc=2, wrongpart=2 |
| attention geometry structured+CF | 0.8667 | 0.9777 | 0.9588/0.9794/1.0000 | missed=1, loc=1, wrongpart=2 |
| coords structured | 0.9000 | 0.9817 | 0.9485/0.9897/1.0000 | missed=1, loc=1, wrongpart=3 |
| coords structured+CF | **0.9333** | **0.9898** | 0.9691/0.9897/1.0000 | missed=0, loc=1, wrongpart=2 |
| occupancy structured+CF | 0.9333 | 0.9878 | 0.9691/1.0000/1.0000 | missed=0, loc=1, wrongpart=1 |
| aux-atom hybrid structured+CF | 0.9333 | 0.9878 | 0.9691/0.9897/1.0000 | missed=0, loc=0, wrongpart=3 |
| CF weight 1.0 | 0.9000 | 0.9857 | 0.9485/0.9794/1.0000 | missed=1, loc=2, wrongpart=2 |

## Error-Level Findings

The best models and ensembles remain at:

```text
EM = 0.9333
F1 = 0.9898
wrong images = picture_183, picture_54
```

Attention geometry is not reliable enough as a direct scorer feature. Its
attention masks are often diffuse or collapse several location queries to
similar centers, which increases region/table errors.

Token coordinates are safer than attention geometry. They recover the best EM
when combined with structured+counterfactual loss, but do not exceed the
no-coordinate structured baseline.

Occupancy loss helps the intended mechanism:

```text
placement top3: 0.9897 -> 1.0000
missed_stack:   0
```

but it can change `picture_54` from table confusion to wrong support-part
confusion:

```text
gold: blue_battery -> green_pump
pred: blue_battery -> red_pump
```

Aux-atom hybrid reranking changes the error mix and removes region errors, but
the atom branch still does not provide strong enough local evidence to rerank
the remaining stack cases.

## Conclusion

The strongest confirmed improvement is still structured legal-state learning:

```text
CE single model:          EM 0.9000
structured single model:  EM 0.9333
best ensemble:            EM 0.9333
```

The failed geometry experiments are informative: object-query attention should
not be treated as a reliable object mask without an explicit localization or
concept supervision signal. The next useful direction is not more score
averaging, but a stronger grounding module:

```text
PDDL query -> localized object/contact evidence -> support score
```

Likely next steps:

```text
1. add explicit contact/support pair margin loss;
2. add a localized/deformable query refinement step before geometry extraction;
3. supervise or regularize query masks with weak region priors;
4. distill a small detector/SAM/segmentation teacher for part masks;
5. use top-k legal reranking only if a separate visual-contact score is added.
```
