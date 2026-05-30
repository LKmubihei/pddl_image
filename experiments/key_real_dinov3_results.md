| feature_source | transition_mask_source | transition_supervision | direct_object_tokens | object_type_source | scoring_head_type | K | condition | f1 | exact_match | precision | recall | pred_pos_rate | label_pos_rate | best_threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cached_dinov3 | pddl | static_pddl_weak | False | oracle | legacy | 20 | adjacent | 0.4000 | 0.0000 | 0.2500 | 1.0000 | 1.0000 | 0.2500 | 0.0500 |
| cached_dinov3 | pddl_conservative | static_pddl_weak | False | oracle | legacy | 20 | adjacent | 0.4000 | 0.0000 | 0.2500 | 1.0000 | 1.0000 | 0.2500 | 0.0500 |
| cached_dinov3 | state_diff | oracle_state_diff | False | oracle | legacy | 20 | adjacent | 0.8481 | 0.0873 | 0.7414 | 0.9906 | 0.3340 | 0.2500 | 0.0500 |
| cached_dinov3 | state_diff | oracle_state_diff | False | oracle | legacy | 20 | full | 0.8905 | 0.3263 | 0.8069 | 0.9934 | 0.3078 | 0.2500 | 0.0500 |
