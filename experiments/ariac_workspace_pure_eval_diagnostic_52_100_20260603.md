# ARIAC Workspace Pure Evaluation Diagnostic

This diagnostic loads one checkpoint and does not train.

## Setup

- checkpoint: `experiments/ariac_52_100_hplus640_d256_coords_structcf_20260602/k_52/placement/model.pt`
- feature_cache: `experiments/ariac_52_100_hplus640_d256_coords_structcf_20260602/ariac_dinov3_raw_s640_l1_last_coords_features.pt`
- train K from checkpoint: `52`
- test size from checkpoint metadata: `100`
- workspace topK: `10`
- workspace weight: `5.0`
- workspace boxes: `/home/pc/pddl_image/data/ariac/labels.csv`
- part bbox labels: `/home/pc/pddl_image/data/ariac/labels`

## Metrics

| decode | EM | F1 | precision | recall | legal |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_normal | 0.7900 | 0.9603 | 0.9552 | 0.9654 | 1.0000 |
| B_workspace_attention_center | 0.7800 | 0.9615 | 0.9564 | 0.9666 | 1.0000 |
| C_workspace_bbox_center | 0.7900 | 0.9638 | 0.9587 | 0.9690 | 1.0000 |

## Top-K Gold Coverage

gold legal state in top10: 92/100 = 0.9200

## BBox Center Coverage

assigned active part bbox centers: 149/333 = 0.4474
samples with all active part bbox centers: 45/100 = 0.4500
samples with all gold chain-root bbox centers: 45/100 = 0.4500

## Changed Images

### A_normal -> B_workspace_attention_center
- changed: 4
  picture_307(gold_topK=True), picture_338(gold_topK=False), picture_357(gold_topK=False), picture_323(gold_topK=True)
- bad_to_good: 0
- good_to_bad: 1
  picture_323(gold_topK=True)
- bad_to_bad: 3
  picture_307(gold_topK=True), picture_338(gold_topK=False), picture_357(gold_topK=False)

### A_normal -> C_workspace_bbox_center
- changed: 5
  picture_307(gold_topK=True), picture_338(gold_topK=False), picture_357(gold_topK=False), picture_310(gold_topK=False), picture_323(gold_topK=True)
- bad_to_good: 1
  picture_307(gold_topK=True)
- good_to_bad: 1
  picture_323(gold_topK=True)
- bad_to_bad: 3
  picture_338(gold_topK=False), picture_357(gold_topK=False), picture_310(gold_topK=False)

## Wrong Samples - A_normal

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


## Wrong Samples - B_workspace_attention_center

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
extra:
  (part_at blue_battery table)

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
  (part_at blue_battery table)
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
extra:
  (part_at blue_regulator table)

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


## Wrong Samples - C_workspace_bbox_center

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
  (part_at blue_battery table)
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
extra:
  (part_at blue_regulator table)

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
  (part_at blue_regulator regulator_placement)
extra:
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

