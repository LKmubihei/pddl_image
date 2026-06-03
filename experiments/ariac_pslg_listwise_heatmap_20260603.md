# ARIAC PSLG / Listwise / Heatmap Experiments, 2026-06-03

## Goal

Test the proposed direction:

```text
PDDL-supervised latent grounding
+ decoder-in-the-loop legal-state ranking
```

The experiments use the same main split as the current baseline:

```text
152 non-duplicate samples
52 train
100 test
DINOv3 H+640 frozen dense features
```

## Implemented Variants

### 1. Listwise-heavy legal-state training

Added:

```bash
--support-ce-weight
--topk-legal-nll-weight
--topk-legal-nll-k
--topk-legal-nll-temperature
```

This trains over current top-K legal PDDL assignments plus gold assignment.

### 2. Heatmap query extractor

Added:

```bash
--object-extractor-type heatmap_queries
```

This replaces iterative object-query refinement with one-step query heatmap
pooling:

```text
PDDL object query -> softmax over dense DINO tokens -> heatmap-pooled slot
```

### 3. Latent grounding auxiliary loss

Added:

```bash
--latent-grounding-weight
--latent-grounding-loc-weight
--latent-grounding-on-weight
--latent-grounding-entropy-weight
```

It uses only PDDL placement facts:

```text
part_at(p, l): heatmap(p) should overlap heatmap(l)
on(p, q): center(p) should be above / horizontally near center(q)
```

No bbox labels are used.

## Results

| setting | extractor | loss change | EM | F1 | top1 / top3 / top10 | missed stack | location region | wrong support |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |
| baseline | object_queries | CE + 0.2 structured + 0.2 CF | 0.7900 | 0.9603 | 0.9129 / 0.9700 / 0.9940 | 7 | 17 | 4 |
| listwise-heavy | object_queries | 0.2 CE + 1.0 structured + 1.0 topK + 0.2 CF | 0.7400 | 0.9495 | 0.8829 / 0.9309 / 1.0000 | 7 | 23 | 3 |
| heatmap-only | heatmap_queries | baseline loss | 0.6100 | 0.9295 | 0.8378 / 0.9369 / 0.9970 | 10 | 39 | 2 |
| heatmap + listwise + grounding | heatmap_queries | 0.2 CE + 1.0 structured + 0.5 topK + 0.2 latent | 0.6300 | 0.9335 | 0.7988 / 0.9189 / 0.9760 | 7 | 37 | 2 |
| iterative + latent grounding | object_queries | baseline + 0.1 latent | 0.7900 | 0.9603 | 0.9159 / 0.9369 / 0.9940 | 10 | 13 | 2 |
| iterative + loc-only latent 0.05 | object_queries | baseline + 0.05 latent loc only | 0.7800 | 0.9537 | 0.8979 / 0.9489 / 0.9910 | 9 | 21 | - |
| iterative + loc-only latent 0.10 | object_queries | baseline + 0.10 latent loc only | 0.7000 | 0.9461 | 0.8799 / 0.9459 / 0.9850 | 12 | 24 | - |
| iterative + small topK | object_queries | baseline + 0.2 topK | 0.7900 | 0.9561 | 0.8889 / 0.9309 / 0.9940 | 6 | 19 | 5 |

## Legal-State Oracle Diagnostic

Using the original baseline support scores and constrained legal decoder:

```text
state oracle top1  EM = 0.7900  79/100
state oracle top3  EM = 0.8300  83/100
state oracle top5  EM = 0.8600  86/100
state oracle top10 EM = 0.9200  92/100
state oracle top25 EM = 0.9400  94/100
```

This is important: per-part top3/top10 is much higher than exact-state topK.
The legal state gold is often in top10, but not usually in top3.

## Interpretation

The proposed idea has useful signals but the first implementation does not
improve EM.

Findings:

```text
1. Aggressive listwise training hurts calibration.
   EM 0.7900 -> 0.7400.

2. One-step heatmap pooling is too weak.
   It loses the iterative object-query refinement and collapses location
   ranking badly: location_region 17 -> 39.

3. Latent grounding auxiliary has an unstable targeted effect.
   The full latent term keeps EM at 0.7900 and improves location_region
   17 -> 13, but worsens missed_stack 7 -> 10 and lowers top3. Removing
   the on/contact term did not preserve this benefit: loc-only 0.05 drops
   EM to 0.7800 and loc-only 0.10 drops EM to 0.7000.

4. Small topK listwise keeps EM but reduces F1/top3.
```

## Conclusion

The results do **not** support replacing the current object-query extractor
with naive heatmap pooling.

They do support a narrower claim:

```text
PDDL-derived latent grounding can reduce location_region errors, but current
on/contact geometry is too crude and hurts stack/contact ranking.
```

The follow-up loc-only experiments rule out the simplest fix:

```text
Keep iterative object queries and apply latent grounding only to location
overlap.
```

The positive `location_region 17 -> 13` signal appears coupled to the crude
on/contact term and does not survive as a clean location-only auxiliary.

Practical conclusion: PSLG is not yet a better main method than the current
baseline. The current weak heatmap/attention variables are not reliable enough
to supervise grounding directly from final PDDL labels. If this direction is
continued, the grounding variable needs a stronger source of alignment, such as
complete proposals, manual/fixed region maps, or a separate region/object
matching objective, instead of using uncalibrated query attention as geometry.
