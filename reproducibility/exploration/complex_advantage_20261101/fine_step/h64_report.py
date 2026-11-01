"""Render the h64 supplement using saved metrics only; no simulation."""
from pathlib import Path
import csv
import json
import statistics

OUT = Path(__file__).resolve().parent
summary = json.loads((OUT / "h64_summary.json").read_text(encoding="utf-8"))
original = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
with (OUT / "h64_seed_metrics.csv").open(encoding="utf-8", newline="") as stream:
    seed_rows = list(csv.DictReader(stream))
lookup = {(int(row["seed"]), row["group"]): row for row in seed_rows}
primary = ["continuous_tip_rmse_mm", "continuous_state_relative_l2"]
pair = {(row["seed"], row["metric"]): row for row in
        list(csv.DictReader((OUT / "h64_paired_seed_differences.csv").open(encoding="utf-8", newline="")))
        if row["contrast"] == "GR-GC"}
lines = ["# h0/64 数值收敛补充", "",
    "该补充只细化已有五种子、已有10k权重的时间积分；未训练或选择新模型。"
    "原914个输出时刻、15.992993212174294秒终点、全部8条初值及四个模型均保留。"
    "协议补充在本级计算开始前保存；不扩展到h0/128。", "",
    "## 五种子误差", "",
    "先求每条轨迹的时间误差，再平均每种子的8条轨迹，最后报告五种子均值±样本标准差。"
    "杆端单位为微米；完整状态为相对L2乘1000。", "",
    "| 模型 | h0/32杆端 (μm) | h0/64杆端 (μm) | h0/32状态 ×1000 | h0/64状态 ×1000 |",
    "|---|---:|---:|---:|---:|"]
def formatted(row, metric):
    item = row["metrics"][metric]
    return f'{item["mean"] * 1000:.6f} ± {item["sample_sd"] * 1000:.6f}'
for row in summary["all_models"]:
    old = next(r for r in original["all_models"] if r["divisor"] == 32 and r["group"] == row["group"])
    lines.append(f'| {row["group"]} | {formatted(old, primary[0])} | {formatted(row, primary[0])} | {formatted(old, primary[1])} | {formatted(row, primary[1])} |')
lines += ["", "## 种子内GR/GC对比", "",
    "相对改善为(GR误差−GC误差)/GR误差；正值表示GC误差较低。"
    "不以总体均值替代方向计数。", "",
    "| 种子 | GR杆端 (μm) | GC杆端 (μm) | 杆端改善 (%) | GR状态 ×1000 | GC状态 ×1000 | 状态改善 (%) |",
    "|---|---:|---:|---:|---:|---:|---:|"]
for seed in [17, 43, 91, 137, 211]:
    gr, gc = lookup[seed, "GR"], lookup[seed, "GC"]
    tip_improvement = float(pair[str(seed), primary[0]]["relative_improvement_fraction"]) * 100
    state_improvement = float(pair[str(seed), primary[1]]["relative_improvement_fraction"]) * 100
    lines.append(f'| {seed} | {float(gr[primary[0]]) * 1000:.6f} | {float(gc[primary[0]]) * 1000:.6f} | {tip_improvement:.3f} | {float(gr[primary[1]]) * 1000:.6f} | {float(gc[primary[1]]) * 1000:.6f} | {state_improvement:.3f} |')
for metric, label in zip(primary, ["杆端", "完整状态"]):
    row = next(r for r in summary["paired"] if r["contrast"] == "GR-GC" and r["metric"] == metric)
    lines += ["", f'{label}：GC方向获胜{row["positive_direction_seed_count"]}/5；种子内相对改善平均{row["mean_relative_improvement_fraction"] * 100:.3f}%。']
passed = all(r["positive_direction_seed_count"] >= 4 and r["mean_relative_improvement_fraction"] >= .05
             for r in summary["paired"] if r["contrast"] == "GR-GC")
lines += ["", f'两个主指标同时满足探索候选规则：{"是" if passed else "否"}。'
    "该结果属于既有数据上的固定权重诊断，不能替代新种子确认。"
    "完整状态的获胜数由h0/32的2/5变为h0/64的4/5，种子91、137在本次细化后改变方向；"
    "因此阈值通过只对应本级离散结果，不能据此声称连续极限下的排序已经认证。", "",
    "## h0/32→h0/64变化", "",
    "轨迹变化与误差指标移动分开比较。下表的变化单位与上表一致；计数均以五种子为分母。", "",
    "| 模型/指标 | 平均轨迹变化 ×1000 | 变化小于同层GR/GC轨迹距离 | 平均误差指标移动 ×1000 | 移动小于GR/GC误差差 |",
    "|---|---:|---:|---:|---:|"]
for group in ["H0", "GR", "GC", "Teacher"]:
    for metric, label in zip(primary, ["杆端", "状态"]):
        rows = [r for r in summary["refinement_comparisons"] if r["group"] == group and r["metric"] == metric]
        lines.append(f'| {group}/{label} | {statistics.mean(r["trajectory_change"] for r in rows) * 1000:.6f} | {sum(r["trajectory_change_below_GR_GC_distance"] for r in rows)}/5 | {statistics.mean(r["metric_change"] for r in rows) * 1000:.6f} | {sum(r["metric_change_below_GR_GC_difference"] for r in rows)}/5 |')
contrast_shifts = []
for seed in [17, 43, 91, 137, 211]:
    for metric in primary:
        gr = next(r for r in summary["refinement_comparisons"] if r["seed"] == seed and r["group"] == "GR" and r["metric"] == metric)
        gc = next(r for r in summary["refinement_comparisons"] if r["seed"] == seed and r["group"] == "GC" and r["metric"] == metric)
        coarse = gr["coarse_error"] - gc["coarse_error"]
        latest = gr["fine_error"] - gc["fine_error"]
        contrast_shifts.append({"seed": seed, "metric": metric, "GR_minus_GC_h32": coarse,
            "GR_minus_GC_h64": latest, "absolute_paired_contrast_shift": abs(latest - coarse),
            "contrast_shift_below_h64_margin": abs(latest - coarse) < abs(latest),
            "direction_changed": (coarse > 0) != (latest > 0)})
with (OUT / "h32_h64_paired_contrast_shift.csv").open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(contrast_shifts[0]))
    writer.writeheader()
    writer.writerows(contrast_shifts)
lines += ["", "同一配对差的移动也需检查：杆端5/5种子的32→64配对差变化小于64级的配对差；"
    "完整状态只有2/5（43、211）满足。17、91、137的配对差变化仍大于最新配对差的绝对值。"
    "完整记录见h32_h64_paired_contrast_shift.csv。"]
lines += ["", "以上是有限两级的数值敏感性核对；个别种子指标差小于细化移动时，须保留排序敏感性的限制。"
    "非线性教师对连续参考的误差应单独观察，不能把网络含模型偏差的总误差拟合为积分方法阶数。", "",
    "## 复核", "",
    f'- 从五个NPZ独立求和/einsum复算全部{summary["independent_primary_metric_trajectory_rows"]}条轨迹的两个主指标，最大差{summary["maximum_independent_metric_gap"]:.3g}。',
    f'- 本级最大隐式方程残差{summary["maximum_implicit_residual"]:.3g}；所有模型/种子完整，无失败。',
    '- 原453个受保护文件及36个已接受h16/h32文件的SHA256均未变，见protection_h64_verification.json。',
    '- 完整数据：seed_*/h64_trajectories.npz、h64_trajectory_metrics.csv、h64_seed_metrics.csv、h64_paired_seed_differences.csv、h32_h64_refinement.csv、h64_summary.json。',
    '- 可重跑h64_convergence.py复核缓存与汇总；h64_report.py只根据保存结果生成本文。',
    '- 已接受fine_step.py的旧summarize_local会读取所有h*_records；新增h64后不要在当前目录重跑旧驱动（包括--summarize-only或--divisors），以免重写16/32摘要。本补充始终使用独立h64汇总，已接受结果保持原样。', ""]
(OUT / "H64_CONVERGENCE.md").write_text("\n".join(lines), encoding="utf-8")
print(json.dumps({"both_primary_candidate_rule": passed,
                  "paired": [r for r in summary["paired"] if r["contrast"] == "GR-GC"]}, ensure_ascii=False))
