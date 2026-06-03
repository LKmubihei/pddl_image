# ARIAC PDDL-Cache Verifier Diagnostic

This diagnostic loads one checkpoint and does not train a neural scorer.
The cache verifier is calibrated by train leave-one-out, then fixed for held-out evaluation.

## Setup

- checkpoint: `experiments/ariac_40_30_100_hplus640_d256_coords_structcf_baseline_20260602/k_30/placement/model.pt`
- feature_cache: `experiments/ariac_40_30_100_hplus640_d256_coords_structcf_baseline_20260602/ariac_dinov3_raw_s640_l1_last_coords_features.pt`
- train K from checkpoint: `30`
- test size from checkpoint metadata: `100`
- legal rerank topK: `10`
- cache mode: `knn_logit`
- bucket mode: `coarse`
- selected config: `k=3, beta=5, edge_lambda=0.2, state_lambda=0`

## Metrics

| decode | EM | F1 | precision | recall | legal |
| --- | ---: | ---: | ---: | ---: | ---: |
| train normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| train LOO cache | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| test normal | 0.6100 | 0.9201 | 0.9141 | 0.9261 | 1.0000 |
| test cache | 0.6100 | 0.9201 | 0.9141 | 0.9261 | 1.0000 |

## Placement Ranking

test placement top1/top3/top10: 0.8138 / 0.8739 / 0.9910
test top1 error counts: missed_stack=11, location_region=44, wrong_support_part=2, false_stack=5

## Top-K Gold Coverage

train gold legal state in top10: 30/30 = 1.0000
test gold legal state in top10: 79/100 = 0.7900

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

wrong_count: 39

### picture_99  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
  (part_at red_pump table)
extra:
  (part_at blue_battery table)
  (part_at blue_pump table)
  (part_at green_regulator table)
  (part_at red_pump pump_placement)

### picture_183  gold_in_topK=False
missing:
  (part_at blue_regulator regulator_placement)
  (part_at red_battery battery_placement)
extra:
  (part_at blue_regulator table)
  (part_at red_battery table)

### picture_385  gold_in_topK=True
missing:
  (part_at red_battery table)
extra:
  (part_at red_battery battery_placement)

### picture_322  gold_in_topK=True
missing:
  (part_at green_regulator buffer_placement)
extra:
  (part_at green_regulator table)

### picture_380  gold_in_topK=False
missing:
  (on blue_regulator red_pump)
  (part_at red_battery table)
extra:
  (part_at blue_regulator table)
  (part_at red_battery battery_placement)

### picture_302  gold_in_topK=True
missing:
  (part_at red_battery battery_placement)
extra:
  (part_at red_battery table)

### picture_376  gold_in_topK=False
missing:
  (on green_battery red_pump)
extra:
  (part_at green_battery table)

### picture_180  gold_in_topK=True
missing:
  (on blue_battery blue_pump)
extra:
  (part_at blue_battery table)

### picture_307  gold_in_topK=False
missing:
  (part_at blue_battery buffer_placement)
  (part_at red_pump table)
extra:
  (part_at blue_battery table)
  (part_at red_pump buffer_placement)

### picture_54  gold_in_topK=True
missing:
  (on blue_battery green_pump)
extra:
  (part_at blue_battery table)

### picture_381  gold_in_topK=True
missing:
  (on blue_regulator red_pump)
extra:
  (part_at blue_regulator table)

### picture_379  gold_in_topK=False
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
  (part_at blue_battery battery_placement)
  (part_at blue_pump pump_placement)
  (part_at green_battery buffer_placement)
  (part_at green_regulator regulator_placement)
  (part_at red_pump table)
extra:
  (part_at blue_battery table)
  (part_at blue_pump table)
  (part_at green_battery table)
  (part_at green_regulator table)
  (part_at red_pump pump_placement)

### picture_384  gold_in_topK=True
missing:
  (part_at red_battery table)
extra:
  (part_at red_battery battery_placement)

### picture_26  gold_in_topK=True
missing:
  (on green_regulator red_pump)
extra:
  (part_at green_regulator table)

### picture_336  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

### picture_201  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

### picture_27  gold_in_topK=False
missing:
  (on green_battery red_pump)
extra:
  (part_at green_battery table)

### picture_181  gold_in_topK=True
missing:
  (part_at red_battery battery_placement)
extra:
  (part_at red_battery table)

### picture_352  gold_in_topK=True
missing:
  (part_at red_battery table)
extra:
  (part_at red_battery battery_placement)

### picture_338  gold_in_topK=False
missing:
  (part_at blue_battery buffer_placement)
  (part_at red_battery table)
extra:
  (part_at blue_battery table)
  (part_at red_battery battery_placement)

### picture_63  gold_in_topK=False
missing:
  (on red_regulator blue_pump)
extra:
  (part_at red_regulator table)

### picture_320  gold_in_topK=True
missing:
  (part_at green_regulator buffer_placement)
extra:
  (part_at green_regulator table)

### picture_208  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

### picture_357  gold_in_topK=False
missing:
  (on blue_regulator green_pump)
  (part_at green_regulator table)
extra:
  (on green_regulator green_pump)
  (part_at blue_regulator table)

### picture_382  gold_in_topK=False
missing:
  (on blue_regulator green_battery)
extra:
  (part_at blue_regulator table)

### picture_371  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

### picture_341  gold_in_topK=True
missing:
  (part_at red_battery battery_placement)
extra:
  (part_at red_battery table)

### picture_375  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

... 9 more wrong samples omitted

## Wrong Samples - test cache

wrong_count: 39

### picture_99  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
  (part_at red_pump table)
extra:
  (part_at blue_battery table)
  (part_at blue_pump table)
  (part_at green_regulator table)
  (part_at red_pump pump_placement)

### picture_183  gold_in_topK=False
missing:
  (part_at blue_regulator regulator_placement)
  (part_at red_battery battery_placement)
extra:
  (part_at blue_regulator table)
  (part_at red_battery table)

### picture_385  gold_in_topK=True
missing:
  (part_at red_battery table)
extra:
  (part_at red_battery battery_placement)

### picture_322  gold_in_topK=True
missing:
  (part_at green_regulator buffer_placement)
extra:
  (part_at green_regulator table)

### picture_380  gold_in_topK=False
missing:
  (on blue_regulator red_pump)
  (part_at red_battery table)
extra:
  (part_at blue_regulator table)
  (part_at red_battery battery_placement)

### picture_302  gold_in_topK=True
missing:
  (part_at red_battery battery_placement)
extra:
  (part_at red_battery table)

### picture_376  gold_in_topK=False
missing:
  (on green_battery red_pump)
extra:
  (part_at green_battery table)

### picture_180  gold_in_topK=True
missing:
  (on blue_battery blue_pump)
extra:
  (part_at blue_battery table)

### picture_307  gold_in_topK=False
missing:
  (part_at blue_battery buffer_placement)
  (part_at red_pump table)
extra:
  (part_at blue_battery table)
  (part_at red_pump buffer_placement)

### picture_54  gold_in_topK=True
missing:
  (on blue_battery green_pump)
extra:
  (part_at blue_battery table)

### picture_381  gold_in_topK=True
missing:
  (on blue_regulator red_pump)
extra:
  (part_at blue_regulator table)

### picture_379  gold_in_topK=False
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
  (part_at blue_battery battery_placement)
  (part_at blue_pump pump_placement)
  (part_at green_battery buffer_placement)
  (part_at green_regulator regulator_placement)
  (part_at red_pump table)
extra:
  (part_at blue_battery table)
  (part_at blue_pump table)
  (part_at green_battery table)
  (part_at green_regulator table)
  (part_at red_pump pump_placement)

### picture_384  gold_in_topK=True
missing:
  (part_at red_battery table)
extra:
  (part_at red_battery battery_placement)

### picture_26  gold_in_topK=True
missing:
  (on green_regulator red_pump)
extra:
  (part_at green_regulator table)

### picture_336  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

### picture_201  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

### picture_27  gold_in_topK=False
missing:
  (on green_battery red_pump)
extra:
  (part_at green_battery table)

### picture_181  gold_in_topK=True
missing:
  (part_at red_battery battery_placement)
extra:
  (part_at red_battery table)

### picture_352  gold_in_topK=True
missing:
  (part_at red_battery table)
extra:
  (part_at red_battery battery_placement)

### picture_338  gold_in_topK=False
missing:
  (part_at blue_battery buffer_placement)
  (part_at red_battery table)
extra:
  (part_at blue_battery table)
  (part_at red_battery battery_placement)

### picture_63  gold_in_topK=False
missing:
  (on red_regulator blue_pump)
extra:
  (part_at red_regulator table)

### picture_320  gold_in_topK=True
missing:
  (part_at green_regulator buffer_placement)
extra:
  (part_at green_regulator table)

### picture_208  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

### picture_357  gold_in_topK=False
missing:
  (on blue_regulator green_pump)
  (part_at green_regulator table)
extra:
  (on green_regulator green_pump)
  (part_at blue_regulator table)

### picture_382  gold_in_topK=False
missing:
  (on blue_regulator green_battery)
extra:
  (part_at blue_regulator table)

### picture_371  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

### picture_341  gold_in_topK=True
missing:
  (part_at red_battery battery_placement)
extra:
  (part_at red_battery table)

### picture_375  gold_in_topK=False
missing:
  (part_at blue_battery battery_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_battery table)
  (part_at green_regulator table)

... 9 more wrong samples omitted
