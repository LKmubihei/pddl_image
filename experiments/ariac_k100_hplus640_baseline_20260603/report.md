# ARIAC Init-State Placement Decoder

Only true atoms from `(:init ...)` are used as labels. Goals and VL text are ignored.

## Metadata

- `n_samples`: `152`
- `n_test`: `52`
- `n_train_pool`: `100`
- `n_parts`: `9`
- `n_locations`: `5`
- `n_canonical_atoms`: `132`
- `feature_source`: `dinov3_raw`
- `input_feature_dim`: `1290`
- `features_on_device`: `True`
- `object_extractor_type`: `object_queries`
- `object_query_relation_layers`: `1`
- `object_query_local_refine`: `False`
- `object_query_local_top_k`: `4`
- `object_query_local_radius`: `2`
- `dense_global_bias`: `False`
- `support_head_type`: `two_stage`
- `support_temperature`: `1.5`
- `support_geometry_type`: `none`
- `support_ce_weight`: `1.0`
- `support_location_prior_weight`: `0.0`
- `support_location_prior_sigma`: `0.2`
- `support_patch_evidence_type`: `none`
- `support_patch_location_scale_init`: `0.5`
- `support_patch_table_scale_init`: `0.5`
- `support_patch_contact_scale_init`: `0.5`
- `support_patch_location_sigma`: `0.18`
- `support_patch_temperature`: `1.0`
- `support_patch_contact_top_k`: `16`
- `support_patch_contact_sigma_x`: `0.12`
- `support_patch_contact_sigma_y`: `0.12`
- `support_patch_contact_gap`: `0.06`
- `region_token_source`: `none`
- `region_token_max_boxes_per_class`: `5`
- `support_hidden_dim`: `512`
- `placement_loss`: `ce_structured`
- `structured_loss_weight`: `0.2`
- `counterfactual_margin_weight`: `0.2`
- `counterfactual_margin`: `1.0`
- `topk_legal_nll_weight`: `0.0`
- `topk_legal_nll_k`: `10`
- `topk_legal_nll_temperature`: `1.0`
- `dynamic_hard_negative_weight`: `0.0`
- `dynamic_hard_negative_margin`: `1.0`
- `dynamic_region_table_weight`: `2.0`
- `dynamic_stack_table_weight`: `2.0`
- `dynamic_wrong_support_weight`: `1.5`
- `occupancy_loss_weight`: `0.0`
- `latent_grounding_weight`: `0.0`
- `latent_grounding_loc_weight`: `1.0`
- `latent_grounding_on_weight`: `1.0`
- `latent_grounding_entropy_weight`: `0.0`
- `hybrid_atom_decode_weight`: `0.0`
- `legal_reranker`: `none`
- `legal_reranker_steps`: `200`
- `legal_reranker_top_k`: `25`
- `legal_reranker_lr`: `0.05`
- `legal_reranker_l2`: `0.05`
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
- `require_labels`: `False`
- `excluded_unlabeled_samples`: `0`
- `ariac_label_dir`: `/home/pc/pddl_image/data/ariac/labels`
- `label_bbox_coverage`: `0`
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
| placement | 100 | 0.8462 | 0.9723 | 0.9678 | 0.9768 | 1.0000 | 0.9415 | 0.9766 | 1.0000 | 3 | 4 |  |
