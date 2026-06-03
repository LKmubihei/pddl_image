# ARIAC Region Token Ablation, 2026-06-03

## Goal

Test whether the new `data/ariac/labels` bbox annotations help when used as an
explicit region/proposal layer, rather than as extra scalar geometry in the
existing support MLP.

## Implementation

New option:

```bash
--region-token-source label_bbox
--region-token-source label_bbox_hybrid
```

`label_bbox` converts the 640 DINO dense map into proposal region tokens:

```text
3 object classes * 5 bbox slots + 5 fixed PDDL location slots = 20 region tokens
```

Each region token is:

```text
ROI-pooled DINO dense feature + [class one-hot, cx, cy, w, h, area, present]
```

Then the existing PDDL object queries attend over these region tokens:

```text
PDDL object query -> region token retrieval -> object/location slot -> support scorer
```

`label_bbox_hybrid` keeps the dense patch tokens and prepends the region tokens:

```text
20 region tokens + 1600 dense tokens = 1620 visual tokens
```

## Results

| setting | test | region source | geometry | hidden | EM | F1 | top1 / top3 / top10 | missed stack | location region | wrong support |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| original full dense | 100 | none | none | 512 | 0.7900 | 0.9603 | 0.9129 / 0.9700 / 0.9940 | 7 | 17 | 4 |
| bbox geometry full | 100 | none | label_bbox | 512 | 0.7200 | 0.9479 | 0.8859 / 0.9610 / 0.9940 | 12 | 22 | 2 |
| region-only full | 100 | label_bbox | none | 256 | 0.5400 | 0.9032 | 0.7868 / 0.8859 / 0.9910 | 23 | 37 | 1 |
| region+dense full | 100 | label_bbox_hybrid | none | 256 | 0.7200 | 0.9472 | 0.8739 / 0.9339 / 0.9970 | 9 | 31 | 0 |
| label-only dense | 29 | none | none | 512 | 0.6207 | 0.9132 | 0.7979 / 0.8404 / 1.0000 | 4 | 15 | 0 |
| label-only bbox geometry | 29 | none | label_bbox | 512 | 0.6552 | 0.9151 | 0.8085 / 0.8936 / 1.0000 | 3 | 14 | 1 |
| label-only region-only | 29 | label_bbox | none | 512 | 0.6207 | 0.9237 | 0.8298 / 0.9362 / 1.0000 | 5 | 11 | 0 |
| label-only region-only | 29 | label_bbox | none | 256 | 0.6552 | 0.9215 | 0.8085 / 0.9043 / 0.9681 | 4 | 13 | 1 |
| label-only region-only e40 | 29 | label_bbox | none | 512 | 0.6207 | 0.9237 | 0.8298 / 0.8936 / 0.9787 | 5 | 11 | 0 |
| label-only region+dense | 29 | label_bbox_hybrid | none | 512 | 0.6207 | 0.9196 | 0.8191 / 0.9149 / 0.9894 | 4 | 12 | 1 |

## Interpretation

Region tokens do help the *ranking structure* on fully labeled samples:

```text
label-only dense top3:       0.8404
label-only region-only top3: 0.9362
location_region:             15 -> 11
```

This is the strongest evidence so far that explicit region/proposal tokens are
useful for region grounding.

However, exact match does not improve beyond the simpler bbox-geometry baseline:

```text
label-only bbox geometry EM: 0.6552
label-only region-only EM:   0.6552 best
```

On the original full split, region-only fails because only 81/152 samples have
labels. For unlabeled images the proposal bank has no part boxes, so replacing
dense tokens with region tokens discards too much visual evidence:

```text
full dense baseline EM: 0.7900
full region-only EM:   0.5400
```

The hybrid version partially recovers but is still below the dense baseline:

```text
full region+dense EM: 0.7200
```

## Conclusion

The region-token idea is directionally correct, but the current label set is
not sufficient as a final solution:

```text
1. labels cover only 81/152 non-duplicate samples;
2. labels have object class but no color/instance identity;
3. current location boxes are fixed rough priors, not annotated regions;
4. support/contact still lacks mask-level contact evidence;
5. the model has no direct object-region matching loss.
```

Best next step:

```text
Add an auxiliary object-region matching loss using pseudo assignments from
color+class boxes, and annotate/fix true placement region polygons. Then train
region retrieval first, before training placement decoding.
```

