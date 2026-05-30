# PDDL-Conditioned Predicate-as-Query Grounding

## Positioning

This codebase implements a PDDL-conditioned predicate-as-query grounding
prototype.  The model compiles a PDDL domain into typed canonical atoms and
grounded action masks, maps visual features into object slots, encodes PDDL
predicate schemas as queries, and scores type-constrained ground atoms in a
canonical order.

The current strongest claim is intentionally narrow:

> PaQ evaluates PDDL-conditioned predicate grounding under cached frozen-DINOv3
> visual features, oracle object-type indexing, and explicitly separated
> transition-supervision sources.

The project should not claim strict online pixel-to-model end-to-end training
unless the runner forwards image tensors through DINOv3 during training.

## Relation to Prior Work

DINOv3 is used as a strong frozen visual representation.  The DINOv3 paper
reports large-scale self-supervised ViT training with high-quality dense
features across visual tasks, which makes it an appropriate feature backbone
for few-shot grounding experiments rather than a symbolic-reasoning baseline
by itself: https://arxiv.org/abs/2508.10104

CLIPort and PerAct are relevant robotics baselines for language-conditioned
manipulation.  CLIPort combines CLIP semantic features with Transporter-style
spatial precision, while PerAct uses a transformer over voxelized 3D
observations for multi-task manipulation:

- CLIPort: https://arxiv.org/abs/2109.12098
- PerAct: https://arxiv.org/abs/2209.05451

RT-2 is a representative vision-language-action line that transfers web-scale
VLM pretraining into robot control by representing actions as tokens.  It is a
strong high-level comparator for end-to-end VLA generalization, but it does not
provide the same PDDL-canonical predicate supervision interface:
https://arxiv.org/abs/2307.15818

VisualPredicator is the closest conceptual neighbor among neuro-symbolic
planning papers: it learns abstract predicates/world models for robot planning
and evaluates sample efficiency, OOD generalization, and interpretability:
https://arxiv.org/abs/2410.23156

## Experimental Semantics

All reported results must include these fields:

| Field | Meaning |
| --- | --- |
| `feature_source` | `synthetic_object_token`, `cached_dinov3`, or `online_dinov3` |
| `direct_object_tokens` | whether the feature tokens are already object-aligned |
| `transition_mask_source` | `state_diff`, `pddl`, `pddl_conservative`, or `pddl_sim` |
| `transition_supervision` | semantic label for the mask source |
| `object_type_source` | `oracle` or `predicted` |
| `threshold_source` | should be `validation` for formal numbers |

Interpretation:

- `state_diff` is oracle transition supervision and should be reported as an
  upper bound.
- `pddl` is static PDDL weak supervision.
- `pddl_conservative` is weak supervision with frame masks disabled.
- synthetic object-token experiments are structural sanity checks.
- cached DINOv3 experiments are real visual-feature grounding, not strict
  end-to-end pixel training.

## Main Experiment Matrix

Run the full real-feature matrix:

```bash
K_VALUES=20,50,100,200 \
N_EPOCHS=100 \
TRANSITION_WARMUP_EPOCHS=20 \
W_CONTRAST=0.0 \
CONDITIONS=static,random_pairs,adjacent,full \
GPU_ID=0 \
experiments/run_real_dinov3_matrix.sh
```

On multi-GPU machines, run the three mask sources in parallel:

```bash
GPU_IDS=0,1,2 \
K_VALUES=20,50,100,200 \
N_EPOCHS=100 \
TRANSITION_WARMUP_EPOCHS=20 \
W_CONTRAST=0.0 \
experiments/run_real_dinov3_matrix_parallel.sh
```

This produces:

- `real_dinov3_state_diff_*`: oracle transition upper bound.
- `real_dinov3_pddl_*`: primary PDDL weak-supervision result.
- `real_dinov3_pddl_conservative_*`: conservative diagnostic.
- `real_dinov3_matrix_*.csv` and `.md`: combined summary.

Quick validation:

```bash
MASK_SOURCE=pddl K_VALUES=20 N_EPOCHS=5 GPU_ID=0 \
experiments/run_real_dinov3_quick.sh
```

The 5-epoch quick command is only a path check.  Formal runs should use the
warmup schedule above; otherwise transition losses can enter before atom-level
grounding is established and produce high-recall/all-positive collapse.

For faster ablations before a full run, set `MAX_TRANSITION_SAMPLES=1024`.
Formal reported numbers should either use all transitions
(`MAX_TRANSITION_SAMPLES=0`) or explicitly report the subsample size.

## Architecture Variant

The scoring head has been upgraded from a purely arity-shared decoder to a
predicate-conditioned FiLM decoder.  For binary predicates it scores:

```text
score(p, o_i, o_j) = out( FiLM_p( MLP([o_i, o_j, o_i * o_j, |o_i - o_j|]) ) )
```

Nullary predicates now read object-set summaries through predicate-query
attention plus mean/max pooling.

This is the current SOTA-oriented variant inside the project because it keeps
the predicate-as-query structure while reducing interference between different
binary predicates.

## Reporting Template

Use this table shape for formal reporting:

| feature_source | transition_mask_source | transition_supervision | direct_object_tokens | object_type_source | K | condition | F1 | EM | pred_pos_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Claims should be phrased as:

> With cached frozen-DINOv3 features and oracle object-type indexing, PaQ
> improves/does not improve over static supervision under `pddl` transition
> masks.

Avoid:

> End-to-end visual grounding is solved.

Avoid:

> `state_diff` is PDDL weak supervision.

## Current Real-DINOv3 Results

The current verified real-feature result is a quick K=20 ablation with cached
DINOv3 features, oracle object types, the legacy scoring head, no transition
warmup, and 1024 transition samples per epoch.  It is not the final full-matrix
number, but it validates the upper-bound path:

| feature_source | mask_source | supervision | scorer | K | condition | F1 | EM | precision | recall | pred_pos_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cached_dinov3 | state_diff | oracle_state_diff | legacy | 20 | adjacent | 0.848 | 0.087 | 0.741 | 0.991 | 0.334 |
| cached_dinov3 | state_diff | oracle_state_diff | legacy | 20 | full | 0.891 | 0.326 | 0.807 | 0.993 | 0.308 |
| cached_dinov3 | pddl | static_pddl_weak | legacy | 20 | adjacent | 0.400 | 0.000 | 0.250 | 1.000 | 1.000 |
| cached_dinov3 | pddl_conservative | static_pddl_weak | legacy | 20 | adjacent | 0.400 | 0.000 | 0.250 | 1.000 | 1.000 |

Result files:

- `experiments/key_real_dinov3_results.md`
- `experiments/key_real_dinov3_results.csv`
- `experiments/legacy_state_diff_nowarm_k20_e30_sub1024/fewshot_structural_results.json`

Interpretation:

- The oracle transition path is functional and reaches strong real-feature
  grounding quickly under the legacy head.
- Static PDDL weak masks still collapse to the all-positive solution because
  the compiled Blocksworld action masks do not contain the conditional delete
  effects needed to constrain false atoms.
- The predicate-conditioned FiLM scorer is currently an exploratory variant,
  not the main reported SOTA setting; it needs additional schedule or loss
  tuning before replacing the legacy head.

## Next SOTA Steps

1. Complete the full real-DINOv3 matrix above.
2. Add predicted-type scoring evaluation alongside oracle-type scoring.
3. Add online-DINOv3 runner for strict pixel-to-model training.
4. Compare against non-PDDL heads using the same cached features and splits.
5. Report mean/std over multiple train seeds for the best setting.
