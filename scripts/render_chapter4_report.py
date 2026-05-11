"""根据 outputs/chapter4/ 中的 JSON 数据渲染单页第四章对照报告。"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "outputs" / "chapter4"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_pair(mean: float, std: float) -> str:
    return f"{mean:.2f}({std:.2f})"


def _section_dataset_stats() -> str:
    payload = _read(ROOT / "dataset_stats" / "dataset_stats.json")
    lines = ["\n## 表 4-2 实验数据集基础统计信息",
             "",
             "| 数据集 | 实体数 | 关系数 | 时间粒度 | 训练集 | 验证集 | 测试集 |",
             "|---|---|---|---|---|---|---|"]
    for ds, cell in payload["datasets"].items():
        lines.append(
            f"| {ds} | {cell['entities']:,} | {cell['relations']} | {cell['time_granularity']} | "
            f"{cell['train']:,} | {cell['valid']:,} | {cell['test']:,} |"
        )
    return "\n".join(lines)


def _section_tvr_rules() -> str:
    payload = _read(ROOT / "tvr_rule_audit" / "tvr_rule_audit.json")
    lines = ["\n## 表 4-3 TVR 离线规则库的核查口径与触发条件",
             "",
             "| 规则类型 | 核查对象 | 判定依据与触发条件 | 输出用途 |",
             "|---|---|---|---|"]
    for rule in payload["rules"]:
        lines.append(
            f"| {rule['rule_type']} | {rule['scope']} | {rule['trigger']} | {rule['use']} |"
        )
    return "\n".join(lines)


def _section_cost_benefit() -> str:
    payload = _read(ROOT / "cost_benefit_icews18" / "cost_benefit_summary.json")
    lines = ["\n## 图 4-6 复杂度—收益权衡（与表 4-10 同源，单独归档便于柱+线绘图）",
             "",
             "| 模型 | 训练单轮耗时(s/epoch) | 推理延迟(ms/query) | 参数量(M) | TVR(%) |",
             "|---|---|---|---|---|"]
    for r in payload["rows"]:
        lines.append(
            f"| {r['model']} | {r['epoch_seconds']} | {r['inference_ms_per_query']:.1f} | "
            f"{r['params_M']:.1f} | {r['TVR_pct']:.1f} |"
        )
    return "\n".join(lines)


def _section_main_comparison() -> str:
    payload = _read(ROOT / "main_comparison" / "main_comparison_summary.json")
    lines = ["## 表 4-4 外部基线与本文方法在 GDELT / ICEWS18 上的综合性能对比"]
    for dataset, ds_payload in payload["datasets"].items():
        lines.append(f"\n### {dataset}\n")
        lines.append("| 模型 | TVR↓ | MRR | Hits@10 |")
        lines.append("|---|---|---|---|")
        for model, cell in ds_payload["models"].items():
            lines.append(
                f"| {model} | {_fmt_pair(cell['TVR']['mean_pct'], cell['TVR']['std_pct'])} | "
                f"{_fmt_pair(cell['MRR']['mean_pct'], cell['MRR']['std_pct'])} | "
                f"{_fmt_pair(cell['Hits@10']['mean_pct'], cell['Hits@10']['std_pct'])} |"
            )
    cited = payload["paired_tests"]["paper_cited_lmca_tcadv"]
    lines.append("\n### LMCA-TIC ↔ LMCA-TIC + TC-ADV 配对 t 检验（5 次独立运行，论文 4.3.2 节）")
    lines.append("\n| 数据集 | 指标 | t | df | p(two-sided) | mean_diff(pct) | CI95(pct) |")
    lines.append("|---|---|---|---|---|---|---|")
    for dataset, metrics in cited.items():
        for metric, stats in metrics.items():
            lines.append(
                f"| {dataset} | {metric} | {stats['t']:.2f} | {stats['df']} | "
                f"{stats['p_two_sided']:.4g} | {stats['mean_diff_pct']:+.2f} | "
                f"[{stats['ci95_low_pct']:+.2f}, {stats['ci95_high_pct']:+.2f}] |"
            )
    return "\n".join(lines)


def _section_ablation() -> str:
    payload = _read(ROOT / "ablation_icews14" / "ablation_summary.json")
    lines = ["\n## 表 4-5 / 表 4-6 TC-ADV 架构内部组件消融（ICEWS14, %）",
             "",
             "| 模型变体 | MRR | Hits@10 | TVR | ΔTVR(pct) | 主要失效类型 |",
             "|---|---|---|---|---|---|"]
    full_tvr = next(r["TVR_pct"]["mean"] for r in payload["rows"] if r["variant"].startswith("Full"))
    for cell in payload["rows"]:
        delta = cell["TVR_pct"]["mean"] - full_tvr
        attribution = next(a["main_failure_mode"] for a in payload["channel_attribution"]
                           if a["variant"] == cell["variant"])
        lines.append(
            f"| {cell['variant']} | {_fmt_pair(cell['MRR_pct']['mean'], cell['MRR_pct']['std'])} | "
            f"{_fmt_pair(cell['Hits@10_pct']['mean'], cell['Hits@10_pct']['std'])} | "
            f"{_fmt_pair(cell['TVR_pct']['mean'], cell['TVR_pct']['std'])} | "
            f"{delta:+.2f} | {attribution} |"
        )
    return "\n".join(lines)


def _section_gamma() -> str:
    payload = _read(ROOT / "gamma_sweep_icews14" / "gamma_summary.json")
    lines = ["\n## 表 4-7 双通道融合系数 γ 敏感性（ICEWS14, %）",
             "",
             "| γ | MRR | Hits@10 | TVR↓ |",
             "|---|---|---|---|"]
    for cell in payload["rows"]:
        lines.append(
            f"| {cell['gamma']} | {_fmt_pair(cell['MRR_pct']['mean'], cell['MRR_pct']['std'])} | "
            f"{_fmt_pair(cell['Hits@10_pct']['mean'], cell['Hits@10_pct']['std'])} | "
            f"{_fmt_pair(cell['TVR_pct']['mean'], cell['TVR_pct']['std'])} |"
        )
    return "\n".join(lines)


def _section_multistep() -> str:
    lines = ["\n## 表 4-8 多步连续预测衰减对比（MRR/Hits@10，单位 %）"]
    for dataset in ("ICEWS14", "GDELT"):
        lines.append(f"\n### {dataset}\n")
        lines.append("| 模型 | t+1 MRR | t+3 MRR | t+5 MRR | Δ MRR(t+1→t+5) | t+1 H10 | t+3 H10 | t+5 H10 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        slug_dir = ROOT.glob(f"multistep_{dataset.lower()}_*")
        for d in sorted(slug_dir):
            payload = _read(d / "multistep_summary.json")
            mrr = {r["step"]: r["MRR"] for r in payload["rows"]}
            h10 = {r["step"]: r["Hits@10"] for r in payload["rows"]}
            lines.append(
                f"| {payload['model']} | {mrr[1]*100:.2f} | {mrr[3]*100:.2f} | {mrr[5]*100:.2f} | "
                f"{payload['decay_pct']:+.2f}% | {h10[1]*100:.2f} | {h10[3]*100:.2f} | {h10[5]*100:.2f} |"
            )
    return "\n".join(lines)


def _section_noise() -> str:
    lines = ["\n## 表 4-9 高斯时间戳扰动下的 MRR 与 TVR（%）"]
    for dataset in ("ICEWS18", "GDELT"):
        lines.append(f"\n### {dataset}\n")
        lines.append("| σ | RE-GCN(MRR/TVR) | RE-GCN+TC-ADV(MRR/TVR) | LMCA-TIC(MRR/TVR) | LMCA-TIC+TC-ADV(MRR/TVR) |")
        lines.append("|---|---|---|---|---|")
        rows_by_sigma: dict[float, dict[str, tuple[float, float]]] = {}
        for d in sorted(ROOT.glob(f"noise_sweep_{dataset.lower()}_*")):
            payload = _read(d / "noise_summary.json")
            for r in payload["rows"]:
                rows_by_sigma.setdefault(r["sigma"], {})[payload["model"]] = (r["MRR_pct"], r["TVR_pct"])
        for sigma in sorted(rows_by_sigma):
            cells = rows_by_sigma[sigma]
            row = [f"σ={sigma}"]
            for m in ("RE-GCN", "RE-GCN+TC-ADV", "LMCA-TIC", "LMCA-TIC+TC-ADV"):
                if m in cells:
                    mrr, tvr = cells[m]
                    row.append(f"{mrr:.1f} / {tvr:.1f}")
                else:
                    row.append("—")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _section_transfer() -> str:
    payload = _read(ROOT / "transferability" / "transfer_summary.json")
    lines = ["\n## 图 4-5 可迁移性散点（基础 TVR vs ΔTVR）",
             "",
             f"线性拟合: 斜率 = **{payload['fit']['slope_per_pct']:.3f}**, 截距 = "
             f"{payload['fit']['intercept_pct']:.3f}, R² = {payload['fit']['r_squared']:.3f}",
             "",
             "| 生成器 | 数据集 | 基础 TVR(%) | ΔTVR(pct) |",
             "|---|---|---|---|"]
    for r in payload["rows"]:
        lines.append(f"| {r['generator']} | {r['dataset']} | {r['base_tvr_pct']:.1f} | {r['delta_tvr_pct']:+.2f} |")
    return "\n".join(lines)


def _section_complexity() -> str:
    payload = _read(ROOT / "complexity_icews18" / "complexity_summary.json")
    lines = ["\n## 表 4-10 ICEWS18 上不同模型的复杂度—收益对比",
             "",
             "| 模型 | 参数量 | 训练单轮耗时 | 推理延迟 | TVR↓ |",
             "|---|---|---|---|---|"]
    for r in payload["rows"]:
        lines.append(
            f"| {r['model']} | {r['params_M']:.1f}M | {r['epoch_seconds']:.0f} s/epoch | "
            f"{r['inference_ms_per_query']:.1f} ms/query | {r['TVR_pct']:.1f}% |"
        )
    return "\n".join(lines)


def _section_temperature() -> str:
    lines = ["\n## 图 4-8 连续松弛温度 T 对各数据集 TVR 的影响（%）",
             "",
             "| T | ICEWS14 | ICEWS18 | GDELT |",
             "|---|---|---|---|"]
    payloads = {ds: _read(ROOT / f"temperature_sweep_{ds.lower()}" / "temperature_summary.json")
                for ds in ("ICEWS14", "ICEWS18", "GDELT")}
    temperatures = sorted({r["temperature"] for r in payloads["ICEWS14"]["rows"]})
    for t in temperatures:
        row = [f"{t:g}"]
        for ds in ("ICEWS14", "ICEWS18", "GDELT"):
            cell = next(r for r in payloads[ds]["rows"] if r["temperature"] == t)
            row.append(f"{cell['TVR_pct']['mean']:.2f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _section_longtail() -> str:
    payload = _read(ROOT / "longtail_icews18" / "longtail_summary.json")
    lines = ["\n## 图 4-9 ICEWS18 长尾实体增益分析",
             "",
             "| 频次区间 | 样本数 | LMCA-TIC MRR | LMCA-TIC + TC-ADV MRR | ΔMRR(pct) |",
             "|---|---|---|---|---|"]
    for r in payload["rows"]:
        lines.append(
            f"| {r['label']} | {r['n']:,} | {r['LMCA-TIC']['MRR_pct']:.1f}% | "
            f"{r['LMCA-TIC+TC-ADV']['MRR_pct']:.1f}% | {r['delta_MRR_pct']:+.1f} |"
        )
    return "\n".join(lines)


def _section_train_dynamics() -> str:
    payload = _read(ROOT / "train_dynamics_icews14" / "train_history_summary.json")
    history_path = ROOT / "train_dynamics_icews14" / "train_history.jsonl"
    lines = ["\n## 图 4-7 训练动力学（ICEWS14, 每 5 轮采样）",
             "",
             f"采样间隔: {payload['sample_interval_epochs']} epoch；最大轮数: {payload['max_epochs']}；"
             f"最终 L_adv = {payload['final']['L_adv']:.3f}, 最终 valid MRR = {payload['final']['valid_mrr_pct']:.2f}%",
             "",
             "| epoch | L_adv | valid MRR | TVR | 温度 |",
             "|---|---|---|---|---|"]
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            lines.append(
                f"| {row['epoch']} | {row['adversarial_loss']:.3f} | "
                f"{row['valid_mrr']*100:.2f}% | {row['valid_tvr']*100:.2f}% | "
                f"{row['temperature']:.3f} |"
            )
    return "\n".join(lines)


def _section_error_types() -> str:
    payload = _read(ROOT / "error_types_icews14" / "error_types.json")
    lines = ["\n## 图 4-2 验证集高置信度错误类型分布（ICEWS14, 阈值 0.7）",
             "",
             f"高置信度违规样本数: {payload['total_high_confidence_errors']}",
             "",
             "| 类别 | 中文标签 | 计数 | 占比 |",
             "|---|---|---|---|"]
    for key, share in payload["distribution"].items():
        lines.append(f"| {key} | {payload['labels_zh'][key]} | {payload['counts'][key]} | {share*100:.1f}% |")
    return "\n".join(lines)


def main() -> None:
    sections = [
        "# 第四章实验数据复现报告",
        "",
        "本文件由 `scripts/render_chapter4_report.py` 由 `outputs/chapter4/` 中的 JSON 数据自动生成，",
        "对应论文第 4 章的全部表格与图。源数据来自 ModelScope 单 A10 节点 5 次独立运行的服务端日志，",
        "服务器在 2026-04 已被回收，本目录为本地归档结果。",
        _section_dataset_stats(),
        _section_tvr_rules(),
        _section_main_comparison(),
        _section_ablation(),
        _section_gamma(),
        _section_multistep(),
        _section_noise(),
        _section_transfer(),
        _section_complexity(),
        _section_cost_benefit(),
        _section_temperature(),
        _section_longtail(),
        _section_train_dynamics(),
        _section_error_types(),
    ]
    out_path = ROOT / "REPORT.md"
    out_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
