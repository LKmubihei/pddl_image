# ARIAC 100 Train / 52 Test Comparison - 2026-06-02

This run uses the updated no-duplicate ARIAC set:

```text
usable samples: 152
train: 100
test: 52
split_seed: 42
init_seed: 42
```

Note: `training/run_ariac_structured.py` previously capped explicit
`--test-size` by `n//5`, so `--test-size 52` still evaluated only 30 images.
The split logic was corrected so explicit `--test-size` is honored.

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

| model | EM | F1 | placement top1/top3/top10 | missed_stack | location_region | wrong_support_part |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| no-geo structured | 0.8269 | 0.9631 | 0.9123/0.9649/1.0000 | 6 | 7 | 2 |
| coords structured+CF | **0.8654** | **0.9700** | **0.9240**/0.9649/1.0000 | **4** | 7 | **1** |
| occupancy structured+CF | 0.8462 | 0.9642 | 0.9181/0.9649/1.0000 | 5 | 8 | 1 |
| hybrid aux structured+CF | 0.8462 | 0.9654 | 0.9123/0.9591/1.0000 | 4 | 9 | 2 |

## Error Sets

```text
no-geo structured wrong:
picture_99, picture_183, picture_376, picture_180, picture_54,
picture_70, picture_100, picture_26, picture_27

coords structured+CF wrong:
picture_99, picture_183, picture_54, picture_70, picture_100,
picture_26, picture_27

occupancy structured+CF wrong:
picture_99, picture_183, picture_376, picture_54, picture_70,
picture_100, picture_26, picture_27

hybrid aux structured+CF wrong:
picture_99, picture_183, picture_321, picture_54, picture_70,
picture_100, picture_26, picture_27
```

The common hard set across all four methods is:

```text
picture_99, picture_183, picture_54, picture_70,
picture_100, picture_26, picture_27
```

Compared with no-geo structured, coords structured+CF fixes:

```text
picture_180
picture_376
```

and adds no new wrong image.

## Interpretation

With a larger held-out set, the methods do separate:

```text
coords structured+CF improves EM by 0.0385 over no-geo structured
coords structured+CF improves F1 by 0.0069 over no-geo structured
```

The improvement is mainly stack/support ranking:

```text
missed_stack:        6 -> 4
wrong_support_part:  2 -> 1
```

However, region/table grounding remains the dominant unsolved issue:

```text
location_region stays at 7 for coords structured+CF
```

Occupancy and hybrid atom reranking do not beat coords structured+CF on this
split. They change the error mix but do not improve exact match.

## Remaining Errors for Best Model

Best model: `coords structured+CF`.

```text
picture_99:
  missing: blue_pump -> pump_placement,
           green_regulator -> regulator_placement
  extra:   blue_pump -> table,
           green_regulator -> buffer_placement

picture_183:
  missing: blue_regulator -> regulator_placement
  extra:   blue_regulator -> table

picture_54:
  missing: blue_battery -> green_pump
  extra:   clear green_pump,
           blue_battery -> table

picture_70:
  missing: blue_battery -> blue_pump
  extra:   clear blue_pump,
           blue_battery -> table

picture_100:
  multiple region/table swaps across battery/pump/buffer/table

picture_26:
  missing: green_regulator -> red_pump
  extra:   clear red_pump,
           green_regulator -> table

picture_27:
  missing: green_battery -> red_pump
  extra:   clear red_pump,
           green_battery -> table
```

## Takeaway

The 30-test split was too small to distinguish methods. On 52 test images,
`coords structured+CF` is the best of the four and improves without adding new
wrong examples. The remaining bottleneck is still visual grounding of regions
and contact/support, not PDDL legality.
