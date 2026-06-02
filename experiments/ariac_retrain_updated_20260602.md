# ARIAC Retrain After Label Updates - 2026-06-02

Duplicate-part samples are excluded. After the latest PDDL edits, the split has
152 usable samples, 30 test samples, and 122 training samples. The test split is
fixed with `split_seed=42`.

## Trained Models

| model | feature | config | init seed | test EM | test F1 | placement top1/top3/top10 | top1 error counts |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| d256_seed42 | DINOv3 raw 448 | object queries, relation=1, two-stage support, d=256, hidden=512 | 42 | 0.9000 | 0.9837 | 0.9588/0.9794/1.0000 | missed_stack=1, location_region=1 |
| d512_seed42 | DINOv3 raw 448 | object queries, relation=1, two-stage support, d=512, hidden=1024 | 42 | 0.8667 | 0.9797 | 0.9485/0.9691/0.9897 | missed_stack=1, location_region=2 |
| d512_seed7 | DINOv3 raw 448 | object queries, relation=1, two-stage support, d=512, hidden=1024 | 7 | 0.8667 | 0.9755 | 0.9485/0.9691/0.9897 | missed_stack=0, location_region=2 |

## Ensemble Results

| ensemble | mode | EM | F1 | precision | recall | global top1/top3/top10 | rank > 10 | mean/max rank |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| d256_seed42 + d512_seed42 | avg log-prob | 0.9333 | 0.9898 | 0.9878 | 0.9918 | 28/29/30 | 0 | 1.167/4 |
| d256_seed42 + d512_seed42 + d512_seed7 | avg log-prob | 0.9333 | 0.9898 | 0.9878 | 0.9918 | 28/29/30 | 0 | 1.200/5 |
| d256_seed42 | single | 0.9000 | 0.9837 | 0.9798 | 0.9878 | 27/30/30 | 0 | 1.167/3 |
| d256_seed42 + d512_seed7 | avg log-prob | 0.9000 | 0.9837 | 0.9798 | 0.9878 | 27/30/30 | 0 | 1.167/3 |
| d512_seed42 + d512_seed7 | avg log-prob | 0.9000 | 0.9837 | 0.9798 | 0.9878 | 27/29/30 | 0 | 1.333/7 |

Best current result: `d256_seed42 + d512_seed42` with avg-logprob support-score
ensemble.

## Remaining Best-Ensemble Errors

### picture_183

Image: `data/ariac/real_pictures/picture_183.png`

Gold placement:

```text
blue_regulator -> regulator_placement
red_battery -> battery_placement
red_pump -> table
```

Predicted placement:

```text
blue_regulator -> table
red_battery -> battery_placement
red_pump -> table
```

Missing atoms:

```text
(part_at blue_regulator regulator_placement)
```

Extra atoms:

```text
(part_at blue_regulator table)
```

### picture_54

Image: `data/ariac/real_pictures/picture_54.jpg`

Gold placement:

```text
blue_battery -> green_pump
green_pump -> table
red_pump -> table
```

Predicted placement:

```text
blue_battery -> table
green_pump -> table
red_pump -> table
```

Missing atoms:

```text
(on blue_battery green_pump)
```

Extra atoms:

```text
(clear green_pump)
(part_at blue_battery table)
```
