# ARIAC PDDL-Cache Verifier Diagnostic

This diagnostic loads one checkpoint and does not train a neural scorer.
The cache verifier is calibrated by train leave-one-out, then fixed for held-out evaluation.

## Setup

- checkpoint: `experiments/ariac_k100_hplus640_baseline_20260603/k_100/placement/model.pt`
- feature_cache: `experiments/ariac_k100_hplus640_baseline_20260603/ariac_dinov3_raw_s640_l1_last_coords_features.pt`
- train K from checkpoint: `100`
- test size from checkpoint metadata: `52`
- legal rerank topK: `10`
- cache mode: `knn_logit`
- bucket mode: `coarse`
- selected config: `k=3, beta=5, edge_lambda=0.2, state_lambda=0`

## Metrics

| decode | EM | F1 | precision | recall | legal |
| --- | ---: | ---: | ---: | ---: | ---: |
| train normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| train LOO cache | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test normal | 0.8462 | 0.9723 | 0.9678 | 0.9768 | 1.0000 |
| test cache | 0.8462 | 0.9723 | 0.9678 | 0.9768 | 1.0000 |

## Placement Ranking

test placement top1/top3/top10: 0.9415 / 0.9766 / 1.0000
test top1 error counts: missed_stack=3, location_region=4, wrong_support_part=3, false_stack=0

## Top-K Gold Coverage

train gold legal state in top10: 100/100 = 1.0000
test gold legal state in top10: 49/52 = 0.9423

## LOO Selection

| rank | k | beta | edge lambda | state lambda | LOO EM | LOO F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 5 | 0.2 | 0 | 1.0000 | 1.0000 |
| 2 | 3 | 10 | 0.2 | 0 | 1.0000 | 1.0000 |
| 3 | 3 | 20 | 0.2 | 0 | 1.0000 | 1.0000 |
| 4 | 5 | 5 | 0.2 | 0 | 1.0000 | 1.0000 |
| 5 | 5 | 10 | 0.2 | 0 | 1.0000 | 1.0000 |
| 6 | 5 | 20 | 0.2 | 0 | 1.0000 | 1.0000 |
| 7 | 10 | 5 | 0.2 | 0 | 1.0000 | 1.0000 |
| 8 | 10 | 10 | 0.2 | 0 | 1.0000 | 1.0000 |
| 9 | 10 | 20 | 0.2 | 0 | 1.0000 | 1.0000 |
| 10 | 20 | 5 | 0.2 | 0 | 1.0000 | 1.0000 |

## Changed Images

- changed: 0
- bad_to_good: 0
- good_to_bad: 0
- bad_to_bad: 0

## Wrong Samples - test normal

wrong_count: 8

### picture_99  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
extra:
  (part_at blue_pump table)

### picture_183  gold_in_topK=True
missing:
  (part_at blue_regulator regulator_placement)
extra:
  (part_at blue_regulator table)

### picture_54  gold_in_topK=True
missing:
  (on blue_battery green_pump)
extra:
  (part_at blue_battery table)

### picture_379  gold_in_topK=True
missing:
  (on blue_regulator green_pump)
extra:
  (on blue_regulator red_battery)

### picture_70  gold_in_topK=True
missing:
  (on blue_battery blue_pump)
extra:
  (part_at blue_battery table)

### picture_100  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at blue_pump pump_placement)
extra:
  (part_at blue_battery buffer_placement)
  (part_at blue_pump table)

### picture_26  gold_in_topK=False
missing:
  (on green_regulator red_pump)
extra:
  (part_at green_regulator table)

### picture_27  gold_in_topK=True
missing:
  (on green_battery red_pump)
extra:
  (part_at green_battery table)


## Wrong Samples - test cache

wrong_count: 8

### picture_99  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
extra:
  (part_at blue_pump table)

### picture_183  gold_in_topK=True
missing:
  (part_at blue_regulator regulator_placement)
extra:
  (part_at blue_regulator table)

### picture_54  gold_in_topK=True
missing:
  (on blue_battery green_pump)
extra:
  (part_at blue_battery table)

### picture_379  gold_in_topK=True
missing:
  (on blue_regulator green_pump)
extra:
  (on blue_regulator red_battery)

### picture_70  gold_in_topK=True
missing:
  (on blue_battery blue_pump)
extra:
  (part_at blue_battery table)

### picture_100  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at blue_pump pump_placement)
extra:
  (part_at blue_battery buffer_placement)
  (part_at blue_pump table)

### picture_26  gold_in_topK=False
missing:
  (on green_regulator red_pump)
extra:
  (part_at green_regulator table)

### picture_27  gold_in_topK=True
missing:
  (on green_battery red_pump)
extra:
  (part_at green_battery table)

