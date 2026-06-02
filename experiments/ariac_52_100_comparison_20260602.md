# ARIAC 52 Train / 100 Test Comparison - 2026-06-02

This run uses the updated no-duplicate ARIAC set:

```text
usable samples: 152
train: 52
test: 100
split_seed: 42
init_seed: 42
```

The split is produced by:

```text
--test-size 100
--k-values 52
```

## Results

All rows use:

```text
DINOv3 raw 448 dense tokens
d_slot=256
object queries
1 relation layer
two-stage support head
support_temperature=1.5
epochs=160
```

| model | EM | F1 | placement top1/top3/top10 | mean/max rank | missed_stack | location_region | wrong_support_part |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| no-geo structured | 0.7100 | 0.9478 | 0.8739/0.9189/0.9910 | 1.508/13 | 8 | 27 | 3 |
| coords structured+CF | **0.7700** | **0.9512** | **0.8919/0.9369/0.9970** | **1.381/11** | **7** | **21** | 4 |
| occupancy structured+CF | 0.6600 | 0.9402 | 0.8619/0.9309/0.9940 | 1.384/13 | 10 | 32 | **2** |
| hybrid aux structured+CF | 0.6700 | 0.9359 | 0.8468/0.9039/0.9880 | 1.580/12 | 10 | 35 | 3 |

## Key Findings

With only 52 training samples, the methods separate clearly:

```text
coords structured+CF vs no-geo structured:
  EM: 0.7100 -> 0.7700  (+0.0600)
  F1: 0.9478 -> 0.9512  (+0.0034)
  placement top1: 0.8739 -> 0.8919
  placement top3: 0.9189 -> 0.9369
  location_region: 27 -> 21
```

`coords structured+CF` is the best model in this low-data split. It improves
global exact match and all placement ranking metrics.

`occupancy structured+CF` and `hybrid aux structured+CF` are worse than the
baseline on this split. They over-constrain or rerank from weak auxiliary
signals when the training set is small.

## Error Sets

```text
no-geo structured wrong: 29 images
coords structured+CF wrong: 23 images
occupancy structured+CF wrong: 34 images
hybrid aux structured+CF wrong: 33 images
```

`coords structured+CF` fixes these no-geo errors:

```text
picture_181
picture_208
picture_302
picture_320
picture_321
picture_336
picture_344
picture_359
picture_377
picture_379
```

It introduces these new errors:

```text
picture_307
picture_323
picture_329
picture_356
```

Net effect:

```text
10 fixed - 4 new = 6 fewer wrong images
```

The common hard set across all four methods contains:

```text
picture_100
picture_183
picture_201
picture_26
picture_27
picture_28
picture_310
picture_338
picture_357
picture_370
picture_371
picture_375
picture_382
picture_390
picture_54
picture_63
picture_70
picture_99
```

## Interpretation

The 52/100 split strengthens the previous conclusion:

```text
coords structured+CF is the most robust of the four tested methods.
```

The coordinate features help in the low-data regime because they give the
object-query extractor stable spatial cues. Counterfactual legal negatives then
help rank region/table and missed-stack alternatives.

The remaining errors are still dominated by:

```text
region/table or region/region confusion
missed stack/contact
wrong support part among visually similar candidates
```

This suggests the next useful improvement is not occupancy or atom reranking as
currently implemented, but stronger visual grounding:

```text
localized object/contact query refinement
weak region-prior supervision
support-pair contrastive loss
or mask/detector teacher features
```
