# TC-ADV

`TC-ADV` 是论文第二个核心工作的科研工程仓库，定位为 `LMCA-TIC` 生成器的对抗验证扩展层。本章实现严格对应论文第 4 章 "基于双通道对抗验证的时间一致性纠偏方法"。

## 仓库布局

```
src/tc_adv/
├── discriminators/      # TRM (4.2.3) / ECM (4.2.4) / 融合 (4.2.4)
├── training/            # 动态边界损失 (4.2.5) / G-D 交替训练 (4.2.6)
├── experiments/
│   ├── runner.py        # 单次训练 / 评测 / suite
│   ├── tvr_evaluator.py # TVR 离线规则库 (4.3.2 表 4-3)
│   ├── sweeps.py        # 多 seed / γ / T / σ / 多步 sweeps（含配对 t 检验）
│   ├── error_analysis.py# 验证集高置信度错误分类 (图 4-2)
│   ├── longtail_analysis.py # 长尾实体分析 (图 4-9)
│   ├── transferability.py   # 基础 TVR vs ΔTVR 散点 (图 4-5)
│   ├── complexity.py    # 参数 / 训练时间 / 推理延迟 (表 4-10)
│   └── plotting.py      # 图 4-2 / 4-3 / 4-4 / 4-5 / 4-7 / 4-8 / 4-9
└── ...

scripts/
├── train.py / evaluate.py / run_suite.py   # 简易包装
├── run_chapter4_sweeps.py                  # 一键跑全部 sweep（按 ModelScope 单卡 A10 缩规模）
├── aggregate_seeds.py                      # 5-seed 配对 t 检验（表 4-4 / 表 4-5）
└── plot_chapter4_figures.py                # 一键画全部图

configs/
├── bridge/                          # 桥接到 LMCA-TIC 的 generator 配置
│   ├── full_icews18_lmca.yaml       # 论文原 8B / DDP 配置（A10 24GB 单卡跑不动）
│   ├── full_gdelt_lmca.yaml
│   ├── a10_icews18_lmca.yaml        # **A10 单卡缩比例配置（Qwen2.5-0.5B）**
│   └── a10_gdelt_lmca.yaml
└── experiments/
    ├── full_icews14.yaml / full_icews18.yaml / full_gdelt.yaml   # 论文规模
    ├── a10_icews14.yaml / a10_icews18.yaml / a10_gdelt.yaml      # 单卡缩比例
    └── ablation_*.yaml
```

## 核心 CLI

```bash
python3 -m pip install -e .

# 单次训练 / 评测
tc-adv train     --config configs/experiments/a10_icews14.yaml
tc-adv evaluate  --config configs/experiments/a10_icews14.yaml

# 第 4 章实验
tc-adv seed-sweep        --config configs/experiments/a10_icews18.yaml --seeds 42 1337 2024 7 9
tc-adv gamma-sweep       --config configs/experiments/a10_icews14.yaml --gammas 0.1 0.3 0.5 0.7 0.9
tc-adv temperature-sweep --config configs/experiments/a10_icews14.yaml --temperatures 0.1 0.3 0.5 0.7 0.9
tc-adv noise-sweep       --config configs/experiments/a10_icews18.yaml --sigmas 0 0.5 1.0 2.0
tc-adv multistep-eval    --config configs/experiments/a10_icews14.yaml --max-steps 5

# 分析与画图
tc-adv error-types       --diagnostics outputs/tcadv_a10_icews14/test_diagnostics.json --threshold 0.7
tc-adv complexity        --config configs/experiments/a10_icews18.yaml
tc-adv evaluate-tvr      --config configs/experiments/a10_icews14.yaml --predictions outputs/tcadv_a10_icews14/test_predictions.jsonl
```

## 章节 → 代码映射

| 论文章节 | 公式 / 表 / 图 | 代码位置 |
|------|------|------|
| 4.2.1 候选连续松弛 | Eq. (4-1) ~ (4-3) | `src/tc_adv/training/objectives.py:gumbel_softmax` |
| 4.2.3 TRM | Eq. (4-4) ~ (4-8) | `src/tc_adv/discriminators/trm.py` |
| 4.2.4 ECM | Eq. (4-9) ~ (4-13) | `src/tc_adv/discriminators/ecm.py` |
| 4.2.5 动态边界损失 | Eq. (4-14) ~ (4-17) | `src/tc_adv/training/objectives.py:relu_margin_loss` |
| 4.2.6 对抗训练 | Eq. (4-18) ~ (4-19) | `src/tc_adv/training/trainer.py` |
| 4.2.7 多步推演 | Eq. (4-20) ~ (4-22) | `trainer.evaluate_multi_step` |
| 4.3.2 TVR 离线规则库 | 表 4-3 | `src/tc_adv/experiments/tvr_evaluator.py` |
| 4.3.3 主对比 | 表 4-4 | `tc-adv seed-sweep` + `scripts/aggregate_seeds.py` |
| 4.3.4 组件消融 | 表 4-5 | `configs/experiments/ablation_*.yaml` |
| 4.3.4 双通道分工 | 表 4-6 | `scripts/aggregate_seeds.py`（基于 4-5 数据）|
| 4.3.4 错误类型分布 | 图 4-2 | `tc-adv error-types` + `plot_chapter4_figures.py` |
| 4.3.4 γ 敏感性 | 表 4-7 | `tc-adv gamma-sweep` |
| 4.3.5 多步预测 | 表 4-8 / 图 4-3 | `tc-adv multistep-eval` |
| 4.3.5 时间戳扰动 | 表 4-9 / 图 4-4 | `tc-adv noise-sweep` |
| 4.3.6 可迁移性 | 图 4-5 | `src/tc_adv/experiments/transferability.py` |
| 4.3.7 复杂度 | 表 4-10 | `tc-adv complexity` |
| 4.3.7 训练动力学 | 图 4-7 | `train_history.jsonl` + `plot_chapter4_figures.py --train-history` |
| 4.3.7 温度参数 T | 图 4-8 | `tc-adv temperature-sweep` |
| 4.3.7 长尾实体 | 图 4-9 | `python -m tc_adv.experiments.longtail_analysis` |

## 单 A10（24GB）跑通流程示例

```bash
# 0) 确认 LMCA-TIC 兄弟仓库与本仓都已 editable 安装
cd ../LMCA-TIC && python -m pip install -e . && cd -
python -m pip install -e .

# 1) 缩比例数据准备（需要数据集已下载到服务器；脚本会自动 normalize 到 data/processed/*）
tc-adv prepare-data --events <events.tsv> --entities <entities.tsv> --output-root data/local/icews18

# 2) 主对比：跑 5 seeds
tc-adv seed-sweep --config configs/experiments/a10_icews18.yaml --seeds 42 1337 2024 7 9
tc-adv seed-sweep --config configs/experiments/a10_gdelt.yaml   --seeds 42 1337 2024 7 9

# 3) 消融：γ 扫 / 温度扫
tc-adv gamma-sweep       --config configs/experiments/a10_icews14.yaml
tc-adv temperature-sweep --config configs/experiments/a10_icews14.yaml

# 4) 鲁棒性：噪声扫 + 多步
tc-adv noise-sweep    --config configs/experiments/a10_icews18.yaml
tc-adv multistep-eval --config configs/experiments/a10_icews14.yaml --max-steps 5

# 5) 错误分类 + 复杂度
tc-adv error-types --diagnostics outputs/tcadv_a10_icews14/test_diagnostics.json
tc-adv complexity  --config configs/experiments/a10_icews18.yaml --output outputs/complexity_icews18.json

# 6) 一键画图
python scripts/plot_chapter4_figures.py --output-dir outputs/figures
```

## 备注：与论文的差异

- 单卡 A10 24GB 配置使用 `Qwen2.5-0.5B-Instruct` 替代论文 `Qwen3-8B`，`micro_batch_size=16` 替代 64，`max_epochs=12` 替代 100。绝对数值与论文表 4-4 不可严格匹配，但论文中"接入 TC-ADV 后 TVR 下降 / MRR 上升"的趋势仍可复现。
- 表 4-4 中 RotatE / BoxTE / RE-GCN / CyGNet / GLTW 的"基础"列直接引用各原始论文报数；TC-ADV 仅在 LMCA-TIC 上跑 `+TC-ADV` 配对（4.3.6 节迁移性分析按用户许可后再补）。
