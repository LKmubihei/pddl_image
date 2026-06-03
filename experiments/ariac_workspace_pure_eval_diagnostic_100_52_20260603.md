# ARIAC Workspace Pure Evaluation Diagnostic

This diagnostic loads one checkpoint and does not train.

## Setup

- checkpoint: `experiments/ariac_k100_hplus640_baseline_20260603/k_100/placement/model.pt`
- feature_cache: `experiments/ariac_52_100_hplus640_d256_coords_structcf_20260602/ariac_dinov3_raw_s640_l1_last_coords_features.pt`
- train K from checkpoint: `100`
- test size from checkpoint metadata: `52`
- workspace topK: `10`
- workspace weight: `5.0`
- workspace boxes: `/home/pc/pddl_image/data/ariac/labels.csv`
- part bbox labels: `/home/pc/pddl_image/data/ariac/labels`

## Metrics

| decode | EM | F1 | precision | recall | legal |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_normal | 0.8462 | 0.9723 | 0.9678 | 0.9768 | 1.0000 |
| B_workspace_attention_center | 0.8462 | 0.9723 | 0.9678 | 0.9768 | 1.0000 |
| C_workspace_bbox_center | 0.8462 | 0.9723 | 0.9678 | 0.9768 | 1.0000 |

## Top-K Gold Coverage

gold legal state in top10: 49/52 = 0.9423

## BBox Center Coverage

assigned active part bbox centers: 95/171 = 0.5556
samples with all active part bbox centers: 28/52 = 0.5385
samples with all gold chain-root bbox centers: 28/52 = 0.5385

## Changed Images

### A_normal -> B_workspace_attention_center
- changed: 0
- bad_to_good: 0
- good_to_bad: 0
- bad_to_bad: 0

### A_normal -> C_workspace_bbox_center
- changed: 0
- bad_to_good: 0
- good_to_bad: 0
- bad_to_bad: 0

## Wrong Samples - A_normal

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


## Wrong Samples - B_workspace_attention_center

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


## Wrong Samples - C_workspace_bbox_center

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

