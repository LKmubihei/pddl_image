# ARIAC Label BBox Ablation, 2026-06-03

## Setup

New annotations were added under `data/ariac/labels` in YOLO format.
They contain object class boxes only:

```text
regulator
battery
pump
```

They do not include color labels. The experiment therefore aligns boxes to
PDDL object names with a deterministic color heuristic on the image crop.

Code changes:

- `--support-geometry-type label_bbox`
- `--ariac-label-dir data/ariac/labels`
- `--require-labels` for label-only subset tests

The bbox geometry is encoded as:

```text
[center_x, center_y, width, height, area, has_box]
```

and passed into the existing two-stage support head as explicit geometry.
DINOv3 remains frozen and cached H+640 dense features are reused.

## Label Coverage

After duplicate-active-part filtering:

```text
all non-duplicate samples: 152
samples with bbox labels: 81
label part assignment rate: 0.9686
```

For the original 52/100 split:

```text
train labels: 32 / 52
test labels: 49 / 100
```

## Results

| setting | test | geometry | hidden | EM | F1 | top1 / top3 / top10 | missed stack | location region | wrong support |
| --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| original baseline | 100 | none | 512 | 0.7900 | 0.9603 | 0.9129 / 0.9700 / 0.9940 | 7 | 17 | 4 |
| full split + bbox | 100 | label_bbox | 512 | 0.7200 | 0.9479 | 0.8859 / 0.9610 / 0.9940 | 12 | 22 | 2 |
| full split + bbox | 100 | label_bbox | 256 | 0.6700 | 0.9415 | 0.8619 / 0.9429 / 0.9970 | 13 | 30 | 0 |
| label-only baseline | 29 | none | 512 | 0.6207 | 0.9132 | 0.7979 / 0.8404 / 1.0000 | 4 | 15 | 0 |
| label-only + bbox | 29 | label_bbox | 512 | 0.6552 | 0.9151 | 0.8085 / 0.8936 / 1.0000 | 3 | 14 | 1 |
| label-only baseline | 29 | none | 256 | 0.6207 | 0.9110 | 0.7979 / 0.9043 / 0.9787 | 3 | 15 | 1 |
| label-only + bbox | 29 | label_bbox | 256 | 0.5862 | 0.9091 | 0.7872 / 0.8617 / 0.9894 | 4 | 16 | 0 |

## Interpretation

The labels contain useful signal, but the current integration is not robust.

On the fully labeled subset, `label_bbox` with hidden dim 512 gives a small
gain:

```text
EM:   0.6207 -> 0.6552
top3: 0.8404 -> 0.8936
missed_stack: 4 -> 3
location_region: 15 -> 14
```

However, on the original 52/100 split, only 49 test images have bbox labels and
the same geometry input hurts:

```text
EM: 0.7900 -> 0.7200
location_region: 17 -> 22
missed_stack: 7 -> 12
```

Lowering the support head hidden dimension to 256 does not fix the issue.

Conclusion:

```text
The bbox annotations are promising as explicit region evidence, but feeding
them directly into the existing support MLP is not the right final design.
They should be used as a region/proposal layer with object-region matching and
geometry/contact scoring, rather than as extra latent features for the current
slot-level scorer.
```

