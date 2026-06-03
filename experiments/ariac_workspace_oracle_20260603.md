# ARIAC Workspace Calibration Experiments - 2026-06-03

The new file:

```text
data/ariac/labels.csv
```

contains one-time static workspace calibration boxes for PDDL location symbols:

```text
table
pump_placement
regulator_placement
buffer_placement
battery_placement
```

These are not per-image dynamic part labels. They are interpreted as raw-image
pixel `xywh` boxes and converted to normalized image-plane coordinates.

## Workspace Boxes

Normalized `xyxy` boxes loaded from `labels.csv`:

```text
table:               (0.005, 0.000, 0.492, 0.988)
pump_placement:      (0.713, 0.177, 0.808, 0.402)
regulator_placement: (0.863, 0.085, 0.958, 0.315)
buffer_placement:    (0.705, 0.431, 0.816, 0.658)
battery_placement:   (0.688, 0.688, 0.846, 0.867)
```

## Implemented Variants

### 1. Workspace Top-K Rerank

The model is trained normally. At inference, the legal decoder enumerates top-K
legal placement assignments, and each assignment receives an additional
workspace consistency score:

```text
Score(A) = model_score(A) + lambda * workspace_score(A)
```

Only chain roots are checked. For example:

```text
blue_battery -> green_pump -> table
```

checks the center of `green_pump`, not `blue_battery`.

The dynamic part center is estimated from object-query attention masks, not from
manual part boxes.

### 2. Workspace Location Query Initialization

The static location query coordinates in `object_slot_init` are initialized from
the workspace box centers instead of the previous hardcoded coordinates.

## Results

### 52 Train / 100 Test

Current-code baseline rerun:

```text
EM 0.7200, F1 0.9442
top1/top3/top10 0.8679/0.9249/0.9880
missed_stack 8, location_region 29
```

Workspace variants:

| setting | EM | F1 | top1 / top3 / top10 | missed stack | location region |
| --- | ---: | ---: | --- | ---: | ---: |
| current baseline rerun | 0.7200 | 0.9442 | 0.8679 / 0.9249 / 0.9880 | 8 | 29 |
| workspace rerank w=1 top10 | 0.7200 | 0.9436 | 0.8679 / 0.9249 / 0.9880 | 8 | 29 |
| workspace rerank w=5 top10 | 0.7200 | 0.9454 | 0.8679 / 0.9249 / 0.9880 | 8 | 29 |
| workspace init location queries | 0.6800 | 0.9431 | 0.8649 / 0.9069 / 0.9820 | 8 | 32 |

### 100 Train / 52 Test

Baseline from the same H+640 setup:

```text
EM 0.8462, F1 0.9723
top1/top3/top10 0.9415/0.9766/1.0000
missed_stack 3, location_region 4
```

Workspace variants:

| setting | EM | F1 | top1 / top3 / top10 | missed stack | location region |
| --- | ---: | ---: | --- | ---: | ---: |
| H+640 baseline | 0.8462 | 0.9723 | 0.9415 / 0.9766 / 1.0000 | 3 | 4 |
| workspace rerank w=5 top10 | 0.8462 | 0.9723 | 0.9415 / 0.9766 / 1.0000 | 3 | 4 |
| workspace init location queries | 0.8269 | 0.9654 | 0.9298 / 0.9474 / 1.0000 | 4 | 6 |

## Attention-Center Diagnostic

Using the K=100 model, object-query attention centers were compared against
gold `part_at(part, location)` labels on the 52-image test split.

```text
gold named-region part_at cases: 33
attention center inside correct named region: 15 / 33 = 45.45%

gold table part_at cases: 123
attention center inside table residual: 121 / 123 = 98.37%
attention center inside any named region for table cases: 0 / 123
```

This explains why workspace reranking does not improve exact match:

```text
table residual evidence is already easy,
but named placement-region evidence depends on dynamic part attention centers,
and those centers are correct for less than half of region cases.
```

Example misses:

```text
picture_99 blue_pump -> pump_placement
  attention center: (0.300, 0.418)
  workspace box:    (0.713, 0.177, 0.808, 0.402)

picture_183 blue_regulator -> regulator_placement
  attention center: (0.179, 0.407)
  workspace box:    (0.863, 0.085, 0.958, 0.315)

picture_100 blue_battery -> battery_placement
  attention center: (0.754, 0.345)
  workspace box:    (0.688, 0.688, 0.846, 0.867)
```

## Interpretation

The workspace labels are useful as an environment calibration, but they do not
help the current model unless the dynamic part grounding is also reliable.

The current object-query attention masks are not calibrated object centers.
Therefore:

```text
workspace polygon + query-attention part center
```

is not a reliable geometry signal.

The previous hardcoded location query coordinates also should not be interpreted
as real workspace coordinates. Replacing them with calibrated box centers makes
the model worse, which suggests those coordinates are functioning more like
learned symbolic priors than literal image-plane anchors.

## Conclusion

Static workspace calibration is conceptually valid, but the current adapter
cannot exploit it directly.

Recommended next use:

```text
1. Keep labels.csv as workspace calibration metadata.
2. Do not use it for direct reranking with query-attention centers.
3. Do not replace location query initialization with labels.csv centers as a
   default method.
4. Use labels.csv only after adding a better dynamic part localizer or proposal
   layer, e.g. part bbox/center pseudo-labels, SAM/GroundingDINO proposals, or a
   trained object-region matching objective.
```

## Pure Evaluation Correction

The first workspace experiments retrained models, which made the 52/100
baseline drift from the historical checkpoint result. A stricter diagnostic was
therefore added:

```text
training/diagnose_workspace_rerank.py
```

This script loads the same checkpoint and support scores once, then compares:

```text
A. normal legal decoder
B. workspace rerank + object-query attention center
C. workspace rerank + available part bbox center, fallback to attention center
```

### 52 Train / 100 Test Checkpoint

Checkpoint:

```text
experiments/ariac_52_100_hplus640_d256_coords_structcf_20260602/k_52/placement/model.pt
```

| decode | EM | F1 | legal |
| --- | ---: | ---: | ---: |
| A normal | 0.7900 | 0.9603 | 1.0000 |
| B workspace + attention center | 0.7800 | 0.9615 | 1.0000 |
| C workspace + bbox center | 0.7900 | 0.9638 | 1.0000 |

Top-K and coverage:

```text
gold legal state in top10: 92/100 = 0.9200
assigned active part bbox centers: 149/333 = 0.4474
samples with all active part bbox centers: 45/100 = 0.4500
```

Changed images:

```text
A -> B:
  changed 4
  bad_to_good 0
  good_to_bad 1: picture_323

A -> C:
  changed 5
  bad_to_good 1: picture_307
  good_to_bad 1: picture_323
```

### 100 Train / 52 Test Checkpoint

Checkpoint:

```text
experiments/ariac_k100_hplus640_baseline_20260603/k_100/placement/model.pt
```

| decode | EM | F1 | legal |
| --- | ---: | ---: | ---: |
| A normal | 0.8462 | 0.9723 | 1.0000 |
| B workspace + attention center | 0.8462 | 0.9723 | 1.0000 |
| C workspace + bbox center | 0.8462 | 0.9723 | 1.0000 |

Top-K and coverage:

```text
gold legal state in top10: 49/52 = 0.9423
assigned active part bbox centers: 95/171 = 0.5556
samples with all active part bbox centers: 28/52 = 0.5385
```

Changed images:

```text
A -> B: 0
A -> C: 0
```

### Corrected Interpretation

The stricter diagnostic shows:

```text
1. Static workspace calibration is not harmful by itself.
2. Attention-center rerank is weak and can hurt.
3. BBox-center rerank can fix one image, but can also break one image.
4. On the stronger K=100 checkpoint, workspace rerank does not change top1 at all.
5. The limiting factor is not only workspace geometry; the model score margins
   and incomplete dynamic part bbox coverage also matter.
```

So `labels.csv` should **not** be rejected, but the current top-K reranking rule
is not enough to turn its geometry into a stable EM improvement.

Detailed reports:

```text
experiments/ariac_workspace_pure_eval_diagnostic_52_100_20260603.md
experiments/ariac_workspace_pure_eval_diagnostic_100_52_20260603.md
```
