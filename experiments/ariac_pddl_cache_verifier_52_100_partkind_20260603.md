# ARIAC PDDL-Cache Verifier Diagnostic

This diagnostic loads one checkpoint and does not train a neural scorer.
The cache verifier is calibrated by train leave-one-out, then fixed for held-out evaluation.

## Setup

- checkpoint: `experiments/ariac_52_100_hplus640_d256_coords_structcf_20260602/k_52/placement/model.pt`
- feature_cache: `experiments/ariac_52_100_hplus640_d256_coords_structcf_20260602/ariac_dinov3_raw_s640_l1_last_coords_features.pt`
- train K from checkpoint: `52`
- test size from checkpoint metadata: `100`
- legal rerank topK: `10`
- cache mode: `knn_logit`
- bucket mode: `part_kind`
- selected config: `k=3, beta=5, edge_lambda=0.2, state_lambda=0`

## Metrics

| decode | EM | F1 | precision | recall | legal |
| --- | ---: | ---: | ---: | ---: | ---: |
| train normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| train LOO cache | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test normal | 0.7900 | 0.9603 | 0.9552 | 0.9654 | 1.0000 |
| test cache | 0.7800 | 0.9591 | 0.9540 | 0.9642 | 1.0000 |

## Placement Ranking

test placement top1/top3/top10: 0.9129 / 0.9700 / 0.9940
test top1 error counts: missed_stack=7, location_region=17, wrong_support_part=4, false_stack=1

## Top-K Gold Coverage

train gold legal state in top10: 52/52 = 1.0000
test gold legal state in top10: 92/100 = 0.9200

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

- changed: 1
  picture_323(gold_topK=True)
- bad_to_good: 0
- good_to_bad: 1
  picture_323(gold_topK=True)
- bad_to_bad: 0

## Wrong Samples - test normal

wrong_count: 21

### picture_99  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_pump table)
  (part_at green_regulator table)

### picture_183  gold_in_topK=False
missing:
  (part_at blue_regulator regulator_placement)
extra:
  (part_at blue_regulator table)

### picture_307  gold_in_topK=True
missing:
  (part_at blue_battery buffer_placement)
  (part_at red_pump table)
extra:
  (part_at blue_battery table)
  (part_at red_pump pump_placement)

### picture_54  gold_in_topK=True
missing:
  (on blue_battery green_pump)
extra:
  (part_at blue_battery table)

### picture_379  gold_in_topK=True
missing:
  (on blue_regulator green_pump)
extra:
  (part_at blue_regulator table)

### picture_70  gold_in_topK=True
missing:
  (on blue_battery blue_pump)
extra:
  (part_at blue_battery table)

### picture_100  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_pump table)
  (part_at green_regulator buffer_placement)

### picture_26  gold_in_topK=True
missing:
  (on green_regulator red_pump)
extra:
  (part_at green_regulator table)

### picture_201  gold_in_topK=True
missing:
  (part_at blue_battery battery_placement)
extra:
  (part_at blue_battery table)

### picture_27  gold_in_topK=True
missing:
  (on green_battery red_pump)
extra:
  (part_at green_battery table)

### picture_338  gold_in_topK=False
missing:
  (part_at blue_battery buffer_placement)
  (part_at green_regulator table)
  (part_at red_battery table)
extra:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
  (part_at red_battery battery_placement)

### picture_63  gold_in_topK=True
missing:
  (on red_regulator blue_pump)
extra:
  (part_at red_regulator table)

### picture_357  gold_in_topK=False
missing:
  (on blue_regulator green_pump)
  (part_at green_regulator table)
extra:
  (part_at blue_regulator table)
  (part_at green_regulator buffer_placement)

### picture_382  gold_in_topK=False
missing:
  (on blue_regulator green_battery)
extra:
  (part_at blue_regulator table)

### picture_371  gold_in_topK=True
missing:
  (part_at blue_battery battery_placement)
extra:
  (part_at blue_battery table)

### picture_375  gold_in_topK=True
missing:
  (part_at blue_battery battery_placement)
extra:
  (part_at blue_battery table)

### picture_370  gold_in_topK=True
missing:
  (part_at blue_battery battery_placement)
extra:
  (part_at blue_battery table)

### picture_28  gold_in_topK=False
missing:
  (on blue_battery green_regulator)
extra:
  (on blue_battery red_battery)

### picture_377  gold_in_topK=True
missing:
  (on blue_battery red_pump)
extra:
  (part_at blue_battery table)

### picture_310  gold_in_topK=False
missing:
  (part_at blue_battery buffer_placement)
  (part_at blue_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at blue_regulator table)

### picture_390  gold_in_topK=True
missing:
  (part_at red_battery table)
extra:
  (part_at red_battery battery_placement)


## Wrong Samples - test cache

wrong_count: 22

### picture_99  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_pump table)
  (part_at green_regulator table)

### picture_183  gold_in_topK=False
missing:
  (part_at blue_regulator regulator_placement)
extra:
  (part_at blue_regulator table)

### picture_307  gold_in_topK=True
missing:
  (part_at blue_battery buffer_placement)
  (part_at red_pump table)
extra:
  (part_at blue_battery table)
  (part_at red_pump pump_placement)

### picture_54  gold_in_topK=True
missing:
  (on blue_battery green_pump)
extra:
  (part_at blue_battery table)

### picture_379  gold_in_topK=True
missing:
  (on blue_regulator green_pump)
extra:
  (part_at blue_regulator table)

### picture_70  gold_in_topK=True
missing:
  (on blue_battery blue_pump)
extra:
  (part_at blue_battery table)

### picture_100  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_pump table)
  (part_at green_regulator buffer_placement)

### picture_26  gold_in_topK=True
missing:
  (on green_regulator red_pump)
extra:
  (part_at green_regulator table)

### picture_201  gold_in_topK=True
missing:
  (part_at blue_battery battery_placement)
extra:
  (part_at blue_battery table)

### picture_27  gold_in_topK=True
missing:
  (on green_battery red_pump)
extra:
  (part_at green_battery table)

### picture_338  gold_in_topK=False
missing:
  (part_at blue_battery buffer_placement)
  (part_at green_regulator table)
  (part_at red_battery table)
extra:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
  (part_at red_battery battery_placement)

### picture_63  gold_in_topK=True
missing:
  (on red_regulator blue_pump)
extra:
  (part_at red_regulator table)

### picture_357  gold_in_topK=False
missing:
  (on blue_regulator green_pump)
  (part_at green_regulator table)
extra:
  (part_at blue_regulator table)
  (part_at green_regulator buffer_placement)

### picture_382  gold_in_topK=False
missing:
  (on blue_regulator green_battery)
extra:
  (part_at blue_regulator table)

### picture_371  gold_in_topK=True
missing:
  (part_at blue_battery battery_placement)
extra:
  (part_at blue_battery table)

### picture_375  gold_in_topK=True
missing:
  (part_at blue_battery battery_placement)
extra:
  (part_at blue_battery table)

### picture_370  gold_in_topK=True
missing:
  (part_at blue_battery battery_placement)
extra:
  (part_at blue_battery table)

### picture_28  gold_in_topK=False
missing:
  (on blue_battery green_regulator)
extra:
  (on blue_battery red_battery)

### picture_377  gold_in_topK=True
missing:
  (on blue_battery red_pump)
extra:
  (part_at blue_battery table)

### picture_310  gold_in_topK=False
missing:
  (part_at blue_battery buffer_placement)
  (part_at blue_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at blue_regulator table)

### picture_323  gold_in_topK=True
missing:
  (part_at blue_battery buffer_placement)
extra:
  (part_at blue_battery table)

### picture_390  gold_in_topK=True
missing:
  (part_at red_battery table)
extra:
  (part_at red_battery battery_placement)

