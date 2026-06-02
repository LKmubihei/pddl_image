# experiments 目录说明

最后更新：2026-06-02

这个目录已清理为当前 ARIAC PDDL image grounding 项目需要保留的主线结果和文字报告。已删除 smoke、中间失败实验、旧 AEPaQ/fair/primitive/support-decoder 结果、strict K=50 结果，以及 K=50 小模块/patch evidence 报告。

当前总占用约 1.3GB，主要来自 H+640 DINOv3 dense feature cache。

## 保留的结果目录

### `ariac_52_100_hplus640_d256_coords_structcf_20260602/`

当前主结果目录。

含义：

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

关键结果：

```text
EM 0.7900
F1 0.9603
placement top1/top3/top10 = 0.9129 / 0.9700 / 0.9940
missed_stack = 7
location_region = 17
```

该目录也保留主 H+640 特征缓存：

```text
ariac_dinov3_raw_s640_l1_last_coords_features.pt
```

这是最大文件，也是后续复现实验最有价值的缓存。

### `ariac_40_30_100_hplus640_d256_coords_structcf_baseline_20260602/`

few-shot curve 的低样本结果目录。

含义：

```text
same 100-test split as K=52
same H+640 d256 coords structured+CF baseline
train K = 40 and 30
```

关键结果：

| train K | EM | F1 | placement top1/top3/top10 | missed_stack | location_region |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 | 0.6300 | 0.9190 | 0.8258 / 0.9099 / 0.9730 | 13 | 39 |
| 30 | 0.6100 | 0.9201 | 0.8138 / 0.8739 / 0.9910 | 11 | 44 |

用途：

```text
展示训练样本减少时，region/table grounding 和 support ranking 明显退化。
```

## 保留的报告文件

### `ariac_52_100_comparison_20260602.md`

52 train / 100 test 设置下的结构化方法比较报告。

用途：

```text
记录 no-geo、coords structured+CF、occupancy/hybrid 等方法的对比。
```

### `ariac_100_52_comparison_20260602.md`

100 train / 52 test 设置下的参考比较报告。

用途：

```text
作为更多训练样本时的参考结果，不和 100-test few-shot curve 直接混比。
```

### `ariac_backbone_scaling_52_100_20260602.md`

backbone / resolution scaling 报告。

用途：

```text
记录 H+448 vs H+640、d_slot=256 vs d_slot=512 等结果。
核心结论是 H+640 有小幅稳定收益，d_slot=512 负收益。
```

### `ariac_geometry_hybrid_followup_20260602.md`

geometry / occupancy / hybrid decode 方向的跟进记录。

用途：

```text
说明 geometry/attention-derived concepts 等方向为何没有成为主方法。
```

### `ariac_hard_samples_52_100_20260602.md`

hard sample 分析报告。

用途：

```text
记录难样本错在哪里，多预测/少预测了哪些 atoms。
```

### `ariac_module_ablation_typed_head_20260602.md`

typed two-stage head ablation 报告。

用途：

```text
记录 typed_two_stage 为什么失败。
结论是拆更复杂 scorer 在少样本下会过拟合，location_region 变差。
```

### `ariac_retrain_updated_20260602.md`

样本修正后重新训练的记录。

用途：

```text
记录数据样本修正后模型重训和 ensemble 相关历史结果。
当前不作为主方法依据。
```

### `ariac_structure_ablation_20260602.md`

结构 ablation 总结。

用途：

```text
记录 object queries、two_stage、structured loss、counterfactual margin 等结构选择。
```

### `ariac_structured_loss_20260602.md`

structured legal-state loss 实验报告。

用途：

```text
记录从 per-part CE 到 legal-state structured loss / CF margin 的改进过程。
```

## 已删除内容

为了避免误读和节省空间，以下类型已删除：

```text
smoke_* 临时验证目录
strict K=50 baseline 目录
K=50 小模块与 patch evidence 报告
calibrated/dynamic/reranker/location-prior/local-refine 中间实验目录
typed_two_stage 失败目录
旧 448/224+448 ablation 缓存
旧 ARIAC 20260601 试错目录与 log
旧 AEPaQ / fair / primitive / support decoder / oracle-state-diff 实验
```

如果后续需要重新跑 K=50，可用 K=52 目录中的 H+640 feature cache 复现，不需要保留旧 K=50 目录。
