# ARIAC Init-State Placement Decoder

Only true atoms from `(:init ...)` are used as labels. Goals and VL text are ignored.

## Metadata

- `n_samples`: `152`
- `n_test`: `100`
- `n_train_pool`: `52`
- `n_parts`: `9`
- `n_locations`: `5`
- `n_canonical_atoms`: `132`
- `feature_source`: `dinov3_raw`
- `input_feature_dim`: `1290`
- `features_on_device`: `True`
- `object_extractor_type`: `object_queries`
- `object_query_relation_layers`: `1`
- `dense_global_bias`: `False`
- `support_head_type`: `two_stage`
- `support_temperature`: `1.5`
- `support_geometry_type`: `none`
- `support_hidden_dim`: `512`
- `placement_loss`: `ce_structured`
- `structured_loss_weight`: `0.2`
- `counterfactual_margin_weight`: `0.2`
- `counterfactual_margin`: `1.0`
- `occupancy_loss_weight`: `0.0`
- `hybrid_atom_decode_weight`: `0.0`
- `feature_projector`: `linear`
- `dinov3_base_dim`: `1280`
- `dinov3_scales`: `[640]`
- `dinov3_last_n_layers`: `1`
- `dinov3_layer_fusion`: `last`
- `dinov3_add_coords`: `True`
- `dinov3_peft`: `none`
- `dinov3_lora_rank`: `0`
- `dinov3_lora_alpha`: `None`
- `dinov3_lora_dropout`: `0.0`
- `dinov3_lora_last_blocks`: `0`
- `dinov3_lora_targets`: ``
- `d_slot`: `256`
- `strict_valid_factor_states`: `True`
- `exclude_duplicate_parts`: `True`
- `excluded_duplicate_part_samples`: `34`
- `train_all`: `False`
- `eval_name`: `test`
- `seed`: `42`
- `split_seed`: `42`
- `init_seed`: `42`
- `duplicate_mode`: `exchangeable`
- `max_duplicate_target_variants`: `1`
- `samples_with_duplicate_variants`: `0`

## Results

| method | K | test EM | test F1 | P | R | legal | pl top1 | pl top3 | pl top10 | miss stack | loc err | threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| placement | 52 | 0.7900 | 0.9603 | 0.9552 | 0.9654 | 1.0000 | 0.9129 | 0.9700 | 0.9940 | 7 | 17 |  |
