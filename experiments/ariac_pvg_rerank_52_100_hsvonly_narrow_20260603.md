# ARIAC PVG Rerank Diagnostic

This diagnostic does not train a new model. It reranks baseline top-K legal states with proposal geometry and atom likelihood.

## Setup

- checkpoint: `experiments/ariac_52_100_hplus640_d256_coords_structcf_20260602/k_52/placement/model.pt`
- feature_cache: `experiments/ariac_52_100_hplus640_d256_coords_structcf_20260602/ariac_dinov3_raw_s640_l1_last_coords_features.pt`
- workspace boxes: `/home/pc/pddl_image/data/ariac/labels.csv`
- part proposal labels: `/home/pc/pddl_image/data/ariac/labels`
- proposal sources: `['hsv']`
- train/test: `52/100`
- topK grid: `[10, 25]`

## Proposal Coverage

| split | label files | active centers | root centers | stack bbox pairs |
| --- | ---: | ---: | ---: | ---: |
| train | 0.6154 | 160/162 = 0.9877 | 150/152 = 0.9868 | 10/10 = 1.0000 |
`train` source counts: hsv=160
| test | 0.4900 | 326/333 = 0.9790 | 299/306 = 0.9771 | 26/27 = 0.9630 |
`test` source counts: hsv=326

## Oracle Coverage

train top10/top25: 52/52 = 1.0000 / 52/52 = 1.0000
test top10/top25: 92/100 = 0.9200 / 94/100 = 0.9400

## Metrics

| decode | train EM | test EM | test F1 | P | R | changed | bad->good | good->bad | bad->bad |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_normal | 1.0000 | 0.7900 | 0.9603 | 0.9552 | 0.9654 | 0 | 0 | 0 | 0 |
| B_atom | 1.0000 | 0.7900 | 0.9603 | 0.9552 | 0.9654 | 0 | 0 | 0 | 0 |
| C_geometry | 1.0000 | 0.7900 | 0.9603 | 0.9552 | 0.9654 | 0 | 0 | 0 | 0 |
| D_pvg | 1.0000 | 0.7900 | 0.9603 | 0.9552 | 0.9654 | 0 | 0 | 0 | 0 |

## Selected Configs

- `B_atom`: `topK=10, region=0, stack=0, atom=0.5, temp=1, tau=0, min_conf=0, changed=none`
- `C_geometry`: `topK=10, region=1, stack=1, atom=0, temp=1, tau=0, min_conf=0.5, changed=root_only`
- `D_pvg`: `topK=10, region=1, stack=1, atom=0, temp=1, tau=0, min_conf=0.5, changed=root_only`

## Error Counts

| decode | wrong edges | missed_stack | location_region | wrong_support_part | false_stack |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_normal | 28 | 9 | 18 | 1 | 0 |
| B_atom | 28 | 9 | 18 | 1 | 0 |
| C_geometry | 28 | 9 | 18 | 1 | 0 |
| D_pvg | 28 | 9 | 18 | 1 | 0 |

## Changed Images

### B_atom
changed-edge evidence: 0/0 = 0.0000
- changed: 0
- bad_to_good: 0
- good_to_bad: 0
- bad_to_bad: 0

### C_geometry
changed-edge evidence: 0/0 = 0.0000
- changed: 0
- bad_to_good: 0
- good_to_bad: 0
- bad_to_bad: 0

### D_pvg
changed-edge evidence: 0/0 = 0.0000
- changed: 0
- bad_to_good: 0
- good_to_bad: 0
- bad_to_bad: 0

## Wrong Samples - A_normal

wrong_count: 21

### picture_99  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_pump table)
  (part_at green_regulator table)

### picture_183  gold_in_topK=True
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

### picture_357  gold_in_topK=True
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


## Wrong Samples - B_atom

wrong_count: 21

### picture_99  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_pump table)
  (part_at green_regulator table)

### picture_183  gold_in_topK=True
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

### picture_357  gold_in_topK=True
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


## Wrong Samples - C_geometry

wrong_count: 21

### picture_99  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_pump table)
  (part_at green_regulator table)

### picture_183  gold_in_topK=True
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

### picture_357  gold_in_topK=True
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


## Wrong Samples - D_pvg

wrong_count: 21

### picture_99  gold_in_topK=False
missing:
  (part_at blue_pump pump_placement)
  (part_at green_regulator regulator_placement)
extra:
  (part_at blue_pump table)
  (part_at green_regulator table)

### picture_183  gold_in_topK=True
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

### picture_357  gold_in_topK=True
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

