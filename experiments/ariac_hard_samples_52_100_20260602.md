# ARIAC Hard Samples - 52 Train / 100 Test - 2026-06-02

Split:

```text
usable samples: 152
train: 52
test: 100
split_seed: 42
```

Models checked:

```text
no-geo structured
coords structured+CF
occupancy structured+CF
hybrid aux structured+CF
```

## Train Set

All four models fit the 52 training samples exactly:

```text
no-geo:    train EM = 1.0000
coords+CF: train EM = 1.0000
occupancy: train EM = 1.0000
hybrid:    train EM = 1.0000
```

So there are no actual train-set missing/extra atoms.

The following are the closest train-set near-misses under the best model
(`coords structured+CF`).  `margin` is the score gap between the gold legal
assignment and the best wrong legal assignment.

| sample | margin | if wrong, missing | if wrong, extra |
| --- | ---: | --- | --- |
| picture_355 | 10.1899 | `(on blue_battery red_pump)` | `(clear red_pump)`, `(part_at blue_battery table)` |
| picture_396 | 11.4941 | `(part_at red_pump pump_placement)` | `(part_at red_pump table)` |
| picture_41 | 11.8406 | `(on blue_battery red_battery)` | `(clear red_battery)`, `(part_at blue_battery table)` |
| picture_387 | 12.4140 | `(part_at green_regulator regulator_placement)` | `(part_at green_regulator buffer_placement)` |
| picture_391 | 12.8939 | `(part_at green_regulator regulator_placement)` | `(part_at green_regulator buffer_placement)` |
| picture_198 | 13.0057 | `(clear red_pump)`, `(part_at blue_battery table)` | `(on blue_battery red_pump)` |
| picture_30 | 13.1223 | `(on red_regulator blue_pump)` | `(clear blue_pump)`, `(part_at red_regulator table)` |
| picture_354 | 13.1554 | `(on green_regulator red_pump)` | `(clear red_pump)`, `(part_at green_regulator table)` |
| picture_334 | 13.3731 | `(part_at green_battery buffer_placement)` | `(part_at green_battery pump_placement)` |
| picture_205 | 14.2129 | `(clear green_regulator)`, `(part_at red_battery buffer_placement)` | `(on red_battery green_regulator)` |

## Test Set

Wrong counts:

```text
no-geo structured:          29 / 100 wrong
coords structured+CF:       23 / 100 wrong
occupancy structured+CF:    34 / 100 wrong
hybrid aux structured+CF:   33 / 100 wrong
```

The hardest test samples below are wrong for all four methods.  Missing/extra
atoms are shown for the best model, `coords structured+CF`.

| sample | missing | extra |
| --- | --- | --- |
| picture_100 | `(part_at blue_pump pump_placement)`, `(part_at green_battery buffer_placement)`, `(part_at green_regulator regulator_placement)`, `(part_at red_pump table)` | `(part_at blue_pump buffer_placement)`, `(part_at green_battery table)`, `(part_at green_regulator buffer_placement)`, `(part_at red_pump pump_placement)` |
| picture_99 | `(part_at blue_pump pump_placement)`, `(part_at green_regulator regulator_placement)`, `(part_at red_pump table)` | `(part_at blue_pump buffer_placement)`, `(part_at green_regulator buffer_placement)`, `(part_at red_pump pump_placement)` |
| picture_201 | `(clear red_pump)`, `(part_at blue_battery battery_placement)`, `(part_at green_regulator regulator_placement)` | `(on blue_battery red_pump)`, `(part_at green_regulator table)` |
| picture_370 | `(clear red_pump)`, `(part_at blue_battery battery_placement)`, `(part_at green_regulator regulator_placement)` | `(on blue_battery red_pump)`, `(part_at green_regulator table)` |
| picture_371 | `(clear red_pump)`, `(part_at blue_battery battery_placement)`, `(part_at green_regulator regulator_placement)` | `(on blue_battery red_pump)`, `(part_at green_regulator table)` |
| picture_375 | `(clear red_pump)`, `(part_at blue_battery battery_placement)`, `(part_at green_regulator regulator_placement)` | `(on blue_battery red_pump)`, `(part_at green_regulator table)` |
| picture_28 | `(clear red_battery)`, `(on blue_battery green_regulator)` | `(clear green_regulator)`, `(on blue_battery red_battery)` |
| picture_310 | `(part_at blue_battery buffer_placement)`, `(part_at blue_regulator regulator_placement)` | `(part_at blue_battery table)`, `(part_at blue_regulator table)` |
| picture_338 | `(part_at blue_battery buffer_placement)`, `(part_at red_battery table)` | `(part_at blue_battery battery_placement)`, `(part_at red_battery battery_placement)` |
| picture_26 | `(on green_regulator red_pump)` | `(clear red_pump)`, `(part_at green_regulator table)` |
| picture_27 | `(on green_battery red_pump)` | `(clear red_pump)`, `(part_at green_battery table)` |
| picture_357 | `(on blue_regulator green_pump)` | `(clear green_pump)`, `(part_at blue_regulator table)` |
| picture_382 | `(on blue_regulator green_battery)` | `(clear green_battery)`, `(part_at blue_regulator table)` |
| picture_54 | `(on blue_battery green_pump)` | `(clear green_pump)`, `(part_at blue_battery table)` |
| picture_63 | `(on red_regulator blue_pump)` | `(clear blue_pump)`, `(part_at red_regulator table)` |
| picture_70 | `(on blue_battery blue_pump)` | `(clear blue_pump)`, `(part_at blue_battery table)` |
| picture_183 | `(part_at blue_regulator regulator_placement)` | `(part_at blue_regulator table)` |
| picture_390 | `(part_at red_battery table)` | `(part_at red_battery battery_placement)` |

## Pattern

The hard errors are mostly:

```text
region/table swaps
region/region swaps
missed stack -> table
wrong support part among visually similar candidates
```

`coords structured+CF` helps compared with no-geo, but the remaining failures
need stronger visual grounding of regions and contact/support, not more legal
decoding.
