# ARIAC Structured Legal-State Loss - 2026-06-02

This experiment upgrades placement training from only per-part support CE to
legal-state structured learning over the same PDDL placement space used by the
decoder.

Data state:

```text
valid samples: 186
duplicate-part excluded: 34
usable samples: 152
train/test: 122/30
split_seed: 42
```

## Implemented Objectives

### Legal-State Structured NLL

For each image, legal placement assignments are exactly enumerated. The model
optimizes:

```text
L_struct =
  logsumexp_{A in Legal(x)} Score(image, A)
  - logsumexp_{A in GoldVariants(x)} Score(image, A)
```

where:

```text
Score(image, A) = sum_p support_score(p -> A[p])
```

This directly trains the same legal PDDL assignment ranking that is used at
decode time.

### PDDL Counterfactual Margin

Legal but wrong counterfactual states are generated from the gold assignment:

```text
stack -> table
region -> table
region -> other region
support swap
```

The loss is:

```text
L_cf = max(0, margin + Score(A_negative) - Score(A_gold))
```

Counterfactual targets are filtered through the same legality checker as the
decoder.

## Single-Model Results

All rows use DINOv3 raw 448 features, object queries, one relation layer,
two-stage support head, `d_slot=256`, `support_hidden=512`, `init_seed=42`.

| model | objective | EM | F1 | placement top1/top3/top10 | top1 error counts |
| --- | --- | ---: | ---: | --- | --- |
| CE baseline | CE | 0.9000 | 0.9837 | 0.9588/0.9794/1.0000 | missed_stack=1, location_region=1 |
| structured | CE + 0.2 structured NLL | 0.9333 | 0.9898 | 0.9588/0.9897/1.0000 | missed_stack=1, location_region=1 |
| counterfactual | CE + 0.2 counterfactual margin | 0.9000 | 0.9837 | 0.9691/1.0000/1.0000 | missed_stack=0, location_region=1 |
| structured + counterfactual | CE + 0.2 structured NLL + 0.2 counterfactual margin | 0.9333 | 0.9878 | 0.9691/0.9897/1.0000 | missed_stack=0, location_region=0 |

## Ensemble Check

The best ensemble after adding structured/counterfactual objectives remains:

```text
EM = 0.9333
F1 = 0.9898
global top1/top3/top10 = 28/30/30
rank > 10 = 0
mean/max rank = 1.067/2
```

Many combinations tie at this level. The important change is that a single
structured model now reaches the same EM as the previous best ensemble.

## Interpretation

The correction works in the intended direction:

```text
old CE single model:        EM 0.9000
structured single model:   EM 0.9333
previous best ensemble:    EM 0.9333
```

Counterfactual margin alone did not raise exact match, but it improved
recoverability and the specific stack ranking behavior:

```text
placement top3: 0.9794 -> 1.0000
missed_stack:   1      -> 0
```

The remaining two wrong examples are unchanged:

```text
picture_183: blue_regulator region/table confusion
picture_54:  blue_battery on green_pump missed stack/contact
```

These are therefore likely visual grounding / geometry concept failures rather
than legal-state ranking failures.
