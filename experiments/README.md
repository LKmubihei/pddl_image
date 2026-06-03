# experiments 目录说明

最后更新：2026-06-03

当前目录只保留：

```text
1. 可复现主线结果目录
2. K=100 baseline 目录
3. few-shot curve 目录
4. 文字报告 / negative ablation 总结
```

已删除 label bbox、region token、PSLG、workspace rerank 重训练等失败方案的训练目录和中间模型。当前总占用约 1.3GB，主要来自一份 H+640 DINOv3 dense feature cache。

## 保留的结果目录

### `ariac_52_100_hplus640_d256_coords_structcf_20260602/`

当前主线结果目录。

```text
train K = 52
test = 100
backbone = frozen DINOv3 H+ at 640
dense tokens = 40x40
coords = enabled
d_slot = 256
object extractor = PDDL object queries
relation layers = 1
support head = two_stage
loss = CE + structured legal-state loss + counterfactual margin
decoder = legal PDDL placement decoder
```

结果：

```text
EM 0.7900
F1 0.9603
placement top1/top3/top10 = 0.9129 / 0.9700 / 0.9940
missed_stack = 7
location_region = 17
```

该目录保留唯一真实 H+640 特征缓存：

```text
ariac_dinov3_raw_s640_l1_last_coords_features.pt
```

### `ariac_k100_hplus640_baseline_20260603/`

100 train / 52 test 的 H+640 baseline 目录，用于更多训练样本下的对照和 workspace pure-eval 诊断。

结果：

```text
EM 0.8462
F1 0.9723
placement top1/top3/top10 = 0.9415 / 0.9766 / 1.0000
missed_stack = 3
location_region = 4
```

该目录的 feature cache 是软链接，指向主线 H+640 cache。

### `ariac_40_30_100_hplus640_d256_coords_structcf_baseline_20260602/`

few-shot curve 目录，记录同一 100-test split 下 K=40 和 K=30 的 baseline。

| train K | EM | F1 | placement top1/top3/top10 | missed_stack | location_region |
| ---: | ---: | ---: | --- | ---: | ---: |
| 40 | 0.6300 | 0.9190 | 0.8258 / 0.9099 / 0.9730 | 13 | 39 |
| 30 | 0.6100 | 0.9201 | 0.8138 / 0.8739 / 0.9910 | 11 | 44 |

用途：展示训练样本减少时，region/table grounding 和 support ranking 明显退化。

## 保留的报告

### 主线与历史对照

```text
ariac_52_100_comparison_20260602.md
ariac_100_52_comparison_20260602.md
ariac_backbone_scaling_52_100_20260602.md
ariac_hard_samples_52_100_20260602.md
ariac_retrain_updated_20260602.md
ariac_structure_ablation_20260602.md
ariac_structured_loss_20260602.md
```

这些报告记录主线结构、backbone/resolution scaling、structured+CF、hard samples 和历史 ensemble/retrain 结果。

### 负向 ablation / 失败方向总结

```text
ariac_geometry_hybrid_followup_20260602.md
ariac_module_ablation_typed_head_20260602.md
ariac_label_bbox_ablation_20260603.md
ariac_region_token_ablation_20260603.md
ariac_pslg_listwise_heatmap_20260603.md
ariac_pslg_k100_52_20260603.md
ariac_workspace_oracle_20260603.md
ariac_workspace_pure_eval_diagnostic_52_100_20260603.md
ariac_workspace_pure_eval_diagnostic_100_52_20260603.md
```

这些报告只保留结论，不保留训练目录和中间模型。核心结论：

```text
1. typed_two_stage / larger scorer 在少样本下负收益。
2. label bbox / region token 有信息，但 coverage 不完整，不能作为主线。
3. PSLG / heatmap / heavy listwise 当前不稳定，不能替代 object_queries baseline。
4. labels.csv 作为 static workspace calibration 合理，但简单 top-K rerank 不能稳定提升 EM。
5. workspace pure eval 显示：bbox-center rerank 在 52/100 修好 1 张又弄坏 1 张，在 100/52 不改变 top1。
```

## 已删除的失败训练目录

已删除以下类型的中间目录和模型：

```text
ariac_label_bbox_full_*
ariac_label_only_*
ariac_label_region_*
ariac_pslg_heatmap_*
ariac_pslg_iter_*
ariac_pslg_listwise_*
ariac_k100_pslg_*
ariac_workspace_current_baseline_*
ariac_workspace_initloc_*
ariac_workspace_w*_top10_*
```

如果需要复现这些失败方向，使用保留的报告和主线 H+640 feature cache 重新运行即可。
