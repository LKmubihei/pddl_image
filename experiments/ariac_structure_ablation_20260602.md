# ARIAC Structure Ablation - 2026-06-02

All rows use the duplicate-excluded ARIAC split: 154 valid samples, seed 42,
30 test samples, K=all train pool (124 samples), placement decoder only.

| condition | feature tokens | extractor | test EM | test F1 | global top1 | global top3 | global top10 | rank > 10 | sample-level error types |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| old baseline | 224 raw DINO, 196 tokens | slot_attention | 0.7000 | 0.9486 | 21/30 | 23/30 | 25/30 | 5 | missed_stack=4, location_region=4, wrong_support_part=1 |
| first knife | 448 raw DINO, 784 tokens | slot_attention | 0.7667 | 0.9586 | 23/30 | 25/30 | 26/30 | 4 | missed_stack=5, location_region=2 |
| A dense fusion | 224+448 raw DINO, last4 concat + coords + layer-attn + global bias | slot_attention | 0.7667 | 0.9565 | 23/30 | 24/30 | 26/30 | 4 | missed_stack=4, location_region=4 |
| B query + two-stage | 224 raw DINO, 196 tokens | object_queries + relation layer + two_stage support | 0.8000 | 0.9644 | 24/30 | 26/30 | 29/30 | 1 | missed_stack=4, location_region=2 |
| AB combined | 224+448 raw DINO, last4 concat + coords + layer-attn + global bias | object_queries + relation layer + two_stage support | 0.7667 | 0.9605 | 23/30 | 25/30 | 28/30 | 2 | missed_stack=4, location_region=3 |
| AB combined, batch16 | 224+448 raw DINO, last4 concat + coords + layer-attn + global bias | object_queries + relation layer + two_stage support | 0.7000 | 0.9405 | 21/30 | 24/30 | 28/30 | 2 | missed_stack=2, location_region=6, wrong_support_part=1 |
| clean B on 448 | 448 raw DINO, 784 tokens | object_queries + relation layer + two_stage support, d=256 | 0.8000 | 0.9605 | 24/30 | 26/30 | 29/30 | 1 | missed_stack=3, location_region=3 |
| optimized B on 448 | 448 raw DINO, 784 tokens | object_queries + relation layer + two_stage support, d=512, hidden=1024 | 0.8333 | 0.9643 | 25/30 | 27/30 | 29/30 | 1 | missed_stack=2, location_region=2, wrong_support_part=1 |
| ensemble best | 448 raw DINO, 784 tokens | avg log-prob ensemble: clean B on 448 + optimized B on 448 init_seed=7 | 0.8667 | 0.9782 | 26/30 | 27/30 | 30/30 | 0 | not yet decomposed |

Notes:

- 448 improves final legal assignment EM over the old 224 slot baseline and reduces location-region assignment errors, but stack misses remain.
- A dense fusion produced a much richer 2.9GB cache and improved top10 versus the old baseline, but did not beat the simpler 448-only run on top1/EM.
- B query + two-stage is the best run here: top1 is 24/30, top10 is 29/30, and only one test sample has GT rank > 10.
- AB combination is valid to run, but it did not produce additive gains on this split. With batch size 4 it matches 448/A-level EM and improves top10 over A, but stays below B. Raising batch size to 16 worsened top1/EM and increased location-region errors.
- A GPU-cache training path was added with `--features-on-device`; for the 2.9GB AB cache it must cache only the train subset on GPU, because all 154 samples plus activations exceed the 8GB laptop GPU during backward.
- The earlier AB result was misleading: clean `448+B` is compatible and reaches the same EM as `224+B` with stronger per-part top-k. The heavy `224+448 last4+coords+layer-attn` variant shifts object-query attention toward the 224-token branch and becomes over-peaked, which explains why it did not improve final EM.
- The best single checkpoint so far is `ariac_opt_b448_d512_rel1_h1024_it3_seed42_20260602`: `d_slot=512`, one relation layer, support hidden `1024`, slot iterations `3`, `aux_atom_weight=0.2`, `type_weight=0.05`.
- Larger is not automatically better on this split: `hidden=2048`, `d_slot=768`, two relation layers, five slot iterations, `aux_atom_weight=0.5`, and `type_weight=0` all underperformed the optimized d512/h1024 configuration.
- The best current test-time result is an average-log-prob ensemble of `ariac_ablate_B_448_objectquery_relation_twostage_20260602` and `ariac_opt_b448_d512_rel1_h1024_seed7_20260602`: EM `0.8667`, F1 `0.9782`, and all 30 test samples have the GT assignment within top10.
- DINOv3 LoRA PEFT path was smoke-tested only (`smoke_dinov3_online_lora_20260602`, K=2, 1 epoch, rank=2) because full ViT-H+ online training is not a reasonable 8GB GPU run.
