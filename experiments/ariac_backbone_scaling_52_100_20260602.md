# ARIAC Backbone / Capacity Scaling - 52 Train / 100 Test - 2026-06-02

Split:

```text
usable samples: 152
train: 52
test: 100
split_seed: 42
init_seed: 42
```

All rows use:

```text
method: coords structured+CF
backbone: DINOv3 ViT-H+/16 frozen dense tokens
object extractor: PDDL object queries
support head: two-stage
structured_loss_weight: 0.2
counterfactual_margin_weight: 0.2
support_temperature: 1.5
```

## Results

| setting | EM | F1 | placement top1/top3/top10 | mean/max rank | missed_stack | location_region | false_stack |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| H+ 448, d_slot=256 | 0.7700 | 0.9512 | 0.8919/0.9369/0.9970 | 1.381/11 | 7 | 21 | 4 |
| H+ 448, d_slot=512 | 0.6600 | 0.9407 | 0.8498/0.9429/0.9970 | 1.459/11 | 9 | 33 | 5 |
| H+ 640, d_slot=256 | **0.7900** | **0.9603** | **0.9129/0.9700**/0.9940 | **1.267**/13 | 7 | **17** | **1** |

## Interpretation

### d_slot=512

Increasing model width from `d_slot=256` to `d_slot=512` is harmful in the
52-sample regime:

```text
EM: 0.7700 -> 0.6600
location_region: 21 -> 33
```

The training loss still reaches zero, so this is likely overfitting / poor
calibration rather than under-capacity.

### 640 Resolution

Increasing DINOv3 input resolution from 448 to 640 helps:

```text
tokens: 28 x 28 = 784  ->  40 x 40 = 1600
EM: 0.7700 -> 0.7900
F1: 0.9512 -> 0.9603
placement top1: 0.8919 -> 0.9129
placement top3: 0.9369 -> 0.9700
location_region: 21 -> 17
false_stack: 4 -> 1
```

This supports the diagnosis that spatial/detail resolution matters more than
simply increasing downstream slot width.

## Error Set Delta

Compared with H+ 448 d256, H+ 640 d256 fixes:

```text
picture_309
picture_323
picture_329
picture_356
```

It introduces:

```text
picture_377
picture_379
```

Net effect:

```text
4 fixed - 2 new = 2 fewer wrong images
```

H+ 640 remaining wrong images:

```text
picture_100, picture_183, picture_201, picture_26, picture_27,
picture_28, picture_307, picture_310, picture_338, picture_357,
picture_370, picture_371, picture_375, picture_377, picture_379,
picture_382, picture_390, picture_54, picture_63, picture_70, picture_99
```

## Takeaway

For small-sample grounding, scaling resolution is more promising than scaling
the downstream slot dimension. The current best among these runs is:

```text
H+ 640, d_slot=256, coords structured+CF
EM = 0.7900
F1 = 0.9603
```

This is still far from 0.9 EM, so the remaining bottleneck is not solved by
H+ resolution alone. The next likely useful step is local/contact grounding or
a stronger dense backbone/teacher comparison, not simply a wider scorer.
