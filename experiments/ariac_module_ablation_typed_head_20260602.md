# ARIAC Small Module Ablation: Typed Support Head - 2026-06-02

Goal:

```text
Improve the trainable grounding/scoring modules rather than the DINOv3 backbone.
```

Motivation:

The previous two-stage support scorer used one shared pair MLP for both:

```text
part -> support_part
part -> location/table
```

This mixes stack/contact scoring and region/table scoring.  A new
`typed_two_stage` head was added:

```text
part_scorer(pair_feat)      for part -> part candidates
location_scorer(pair_feat)  for part -> location/table candidates
stack_gate(part)            retained from two_stage
candidate_bias              learned per part/candidate prior
```

## Result

Split:

```text
train: 52
test: 100
backbone: DINOv3 H+ 640 frozen dense tokens
method: coords structured+CF
```

| support head | EM | F1 | placement top1/top3/top10 | mean/max rank | missed_stack | location_region |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| two_stage baseline | **0.7900** | **0.9603** | **0.9129/0.9700/0.9940** | **1.267**/13 | **7** | **17** |
| typed_two_stage | 0.7200 | 0.9502 | 0.8829/0.9399/0.9820 | 1.523/13 | 8 | 24 |

## Interpretation

The typed scorer is worse:

```text
EM: 0.7900 -> 0.7200
location_region: 17 -> 24
placement top3: 0.9700 -> 0.9399
```

The likely reason is small-sample overfitting.  The new head increases
parameters and separates data into two scorer paths, but there are only 52
training examples.  The model still fits train perfectly, while test ranking
gets worse.

## Takeaway

Small-module changes are worth testing, but this specific one is not the right
direction.  The next safer module improvements should add stronger inductive
bias without much extra parameter count:

```text
1. fixed or lightly learned location priors;
2. explicit support/contact margin losses;
3. low-parameter calibration per candidate type;
4. local query refinement with shared weights;
5. teacher/weak mask supervision for query attention.
```

The current best remains:

```text
DINOv3 H+ 640 + d_slot=256 + coords structured+CF
EM = 0.7900
F1 = 0.9603
```
