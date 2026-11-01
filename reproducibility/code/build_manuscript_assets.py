"""Build manuscript tables/figures only from saved, completed measurements."""
from pathlib import Path
import os
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = ROOT / "manuscript"
TABLES, FIGURES = PAPER / "tables", PAPER / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
def cjk_font_family():
    """Use an installed CJK family, or an explicitly selected font file."""
    configured = os.environ.get("IJRR_CJK_FONT")
    if configured and Path(configured).is_file():
        font_manager.fontManager.addfont(configured)
        return font_manager.FontProperties(fname=configured).get_name()
    windows_fonts = Path(os.environ.get("WINDIR", os.environ.get("SystemRoot", "C:/Windows"))) / "Fonts"
    for name in ["msyh.ttc", "simhei.ttf", "simsun.ttc"]:
        candidate = windows_fonts / name
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
    available = {font.name for font in font_manager.fontManager.ttflist}
    choices = [configured] if configured else [
        "Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "WenQuanYi Zen Hei", "SimHei", "SimSun", "PingFang SC", "Heiti SC",
    ]
    for family in choices:
        if family in available:
            return family
    raise RuntimeError("No supported CJK font found; set IJRR_CJK_FONT to an installed CJK family or font file.")


plt.rcParams.update({"font.family": cjk_font_family(), "font.size": 10,
                     "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
GROUPS = ["R", "C", "GR", "GC"]
SEEDS = [17, 43, 91, 137, 211]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pm(obj, decimals=3):
    return f"${obj['mean']:.{decimals}f} \\pm {obj['sample_sd']:.{decimals}f}$"


def scientific(value, significant_digits=4):
    mantissa, exponent = f"{value:.{significant_digits - 1}e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def write_table(name, caption, label, columns, headings, rows, note=""):
    text = "\n".join([
        r"\begin{table*}[t]", r"\centering\small",
        r"\caption{" + caption + "}", r"\label{" + label + "}",
        r"\begin{tabular}{" + columns + "}", r"\toprule",
        " & ".join(headings) + r" \\", r"\midrule",
        *[" & ".join(row) + r" \\" for row in rows],
        r"\bottomrule", r"\end{tabular}",
        (r"\par\smallskip\begin{minipage}{0.96\textwidth}\footnotesize " + note
         + r"\end{minipage}") if note else "", r"\end{table*}", ""])
    (TABLES / f"{name}.tex").write_text(text, encoding="utf-8")


def save_figure(fig, name):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=250, facecolor="white")
    plt.close(fig)


def plot_neural_comparison(runs=None):
    """Render only the neural figure from the frozen per-run measurements."""
    if runs is None:
        runs = read_csv(RESULTS / "neural_learning_10000/all_runs.csv")
    assert {(int(r["seed"]), r["group"]) for r in runs} == {(s, g) for s in SEEDS for g in GROUPS}
    with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), layout="constrained")
        for ax, metric, title in zip(axes, ["teacher_tip_rmse_mm_mean", "tip_rmse_mm_mean"],
                                     ["(a) Discrete teacher", "(b) Continuous reference"]):
            for i, (g, color) in enumerate(zip(GROUPS, COLORS)):
                values = [float(next(r for r in runs if r["group"] == g and int(r["seed"]) == s)[metric]) for s in SEEDS]
                ax.scatter(i + np.linspace(-.13, .13, 5), values, color=color, s=33, alpha=.9, zorder=3)
                ax.plot([i-.21, i+.21], [np.mean(values)]*2, color="black", lw=1.8)
            ax.set(xticks=range(4), xticklabels=GROUPS, title=title, ylabel="Tip time-domain RMSE (mm)")
            ax.grid(axis="y", alpha=.18)
        axes[0].set_yscale("log")
        axes[1].set_ylim(bottom=0)
        save_figure(fig, "neural_comparison")


def main():
    neural = json.loads((RESULTS / "neural_learning_10000/summary.json").read_text(encoding="utf-8-sig"))
    assert neural["completed"] == 20 and not neural["failures"]
    assert neural["complete_paired_seeds"] == SEEDS
    runs = read_csv(RESULTS / "neural_learning_10000/all_runs.csv")
    assert {(int(r["seed"]), r["group"]) for r in runs} == {(s, g) for s in SEEDS for g in GROUPS}
    models = neural["models"]
    rows = []
    for g in GROUPS:
        m = models[g]
        scaled = {k: 1e3 * v for k, v in m["teacher_tip_rmse_mm_mean"].items()}
        rows.append([g, pm(scaled), pm(m["tip_rmse_mm_mean"], 5),
                     pm(m["state_relative_l2_mean"], 4), pm(m["training_seconds"], 2)])
    write_table("neural", "端到端网络的独立测试结果（五种子均值与样本标准差）。", "tab:neural",
                "lrrrr", ["模型", r"对教师RMSE ($\mu$m)", "对连续参考RMSE (mm)",
                           r"状态相对$L^2$误差", "训练时间 (s)"], rows,
                r"每种子先对8条轨迹分别计算时间域末端位置RMSE，再取轨迹平均；预测时长15.993 s。"
                r"状态误差相对于连续参考。R/C无先验，GR/GC固定已知二次能量。每组673个实参数、10000次更新；CPU单线程耗时仅代表本机。")
    pairs = { (r["metric"], r["contrast"]): r for r in neural["paired_contrasts"] }
    contrasts = ["R-GR", "C-GC", "R-C", "GR-GC", "(GR-GC)-(R-C)"]
    write_table("paired", "端到端模型的五项种子内配对差。", "tab:paired", "lrr",
                ["配对差", "对教师RMSE差 (mm)", "对连续参考RMSE差 (mm)"],
                [[f"${c}$", pm(pairs["teacher_tip_rmse_mm_mean", c], 6),
                  pm(pairs["tip_rmse_mm_mean", c], 6)] for c in contrasts],
                r"报告5个配对差的均值$\pm$样本标准差。正值表示前一组误差更大；交互项按式\eqref{eq:five-contrasts}定义。"
                r"这些描述性统计不构成显著性检验。")

    plot_neural_comparison(runs)

    convergence = read_csv(RESULTS / "convergence.csv")
    labels = {"linear_conservative": "常质量／保守", "linear_damped": "常质量／阻尼",
              "variable_mass_conservative": "变质量／保守", "variable_mass_damped": "变质量／阻尼"}
    rows = []
    for case, label in labels.items():
        r = next(r for r in convergence if r["case"] == case and float(r["h"]) == .025)
        rows.append([label, f"${float(r['state_error']):.6g}$", f"${float(r['observed_order']):.6f}$"])
    write_table("convergence", "固定时域$T=10$上的状态误差及观察阶。", "tab:convergence", "lrr",
                ["算例", "$h=0.025$的终点状态误差", "$h=0.05$至$0.025$的阶"], rows)

    ref = read_csv(RESULTS / "learning/refinement/metrics.csv")
    rows = []
    for factor in [1, 2, 4]:
        a = next(r for r in ref if r["model"] == "teacher" and int(r["refinement_factor"]) == factor)
        b = next(r for r in ref if r["model"] == "GC" and int(r["refinement_factor"]) == factor)
        rows.append(["$h$" if factor == 1 else f"$h/{factor}$", f"{float(a['tip_rmse_mm']):.8f}",
                     f"{float(b['tip_rmse_mm']):.8f}",
                     "---" if factor == 1 else f"{float(b['tip_rmse_observed_order']):.4f}",
                     scientific(float(b['same_step_learning_tip_rmse_mm']))])
    write_table("refinement", "40秒固定权重步长加密的代表性结果（seed17）。", "tab:refinement", "lrrrr",
                ["步长", "教师对连续参考 (mm)", "GC对连续参考 (mm)", "GC观察阶", "GC对同步长教师 (mm)"], rows,
                "这里的GC为匹配解析特征读出。误差对8条轨迹及全部时刻联合计算RMSE；与端到端表的聚合方式不同。其余三组的完整结果随数据提供。")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), layout="constrained")
    for group, color, marker in zip(["teacher"]+GROUPS, ["#333333"]+COLORS, ["x", "o", "s", "^", "D"]):
        rr = sorted([r for r in ref if r["model"] == group], key=lambda r: int(r["refinement_factor"]))
        h = np.array([float(r["step_tau"]) for r in rr])
        axes[0].loglog(h, [float(r["tip_rmse_mm"]) for r in rr], marker=marker, ms=6, lw=1.2, color=color, label=group)
        if group != "teacher":
            axes[1].loglog(h, [float(r["same_step_learning_tip_rmse_mm"]) for r in rr], marker=marker, ms=5, lw=1.2, color=color, label=group)
    for ax in axes:
        ax.set_xlabel(r"无量纲步长 $h_\tau$")
        ax.set_ylabel("末端位置时间域 RMSE (mm)")
        ax.grid(which="major", alpha=.18)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("a  相对连续参考（曲线近乎重合）")
    axes[1].set_title("b  相对同步长教师")
    axes[1].legend(loc="center", bbox_to_anchor=(.68, .5), frameon=False, fontsize=8)
    save_figure(fig, "step_refinement")

    kam = json.loads((RESULTS / "kam/results.json").read_text(encoding="utf-8-sig"))
    references = kam["continuous_references"]
    cases = kam["cases"] if "cases" in kam else kam["case_metrics"]
    rows = []
    for r in references:
        a = r["amplitude"]
        order = next(x["observed_frequency_error_order"] for x in kam["step_sensitivity"] if x["amplitude"] == a and x["fine_h"] == .025)
        rows.append([f"{a:.1f}", f"{r['action']:.6f}", f"{r['omega']:.6f}", f"{r['numerical_twist']:.6f}", f"{order:.6f}"])
    write_table("kam", "连续能级的作用量、频率及数值非退化性诊断。", "table:kam", "rrrrr",
                ["初幅$A$", "作用量$I$", "角频率$\\omega$", "$d\\omega/dI$", "频率误差观察阶"], rows,
                "观察阶由$h=0.05$与$0.025$得到。正的采样twist不等同于区间内严格非退化下界。")
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0), layout="constrained")
    for amplitude, color in zip([.2, .8, 1.2], COLORS):
        case = next(r for r in cases if r["amplitude"] == amplitude and r["h"] == .025)
        with np.load(RESULTS / "kam" / case["trajectory_file"]) as data:
            state = data["state"]
            axes[0, 0].plot(state[:, 0], state[:, 1], lw=.8, color=color, label=f"A={amplitude:.1f}")
            omega, power = data["spectrum_omega"], data["spectrum_power"]
            keep = (omega > 0) & (omega < 6)
            axes[0, 1].semilogy(omega[keep], np.maximum(power[keep]/power.max(), 1e-18), color=color, lw=.8, label=f"A={amplitude:.1f}")
        rr = sorted([r for r in cases if r["amplitude"] == amplitude], key=lambda r: r["h"])
        axes[1, 1].loglog([r["h"] for r in rr], [r["omega_error_absolute"] for r in rr], "o-", color=color, ms=4, label=f"A={amplitude:.1f}")
    axes[0, 0].set(xlabel="q", ylabel="p", title="a  保守标量相轨道")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 1].set(xlabel=r"角频率 $\omega$", ylabel="归一化谱功率", ylim=(1e-12, 2), title="b  基频及谐波")
    axes[1, 0].plot([r["action"] for r in references], [r["omega"] for r in references], "o-", color=COLORS[0])
    axes[1, 0].set(xlabel="连续作用量 I", ylabel=r"连续角频率 $\omega$", title="c  采样频率随作用量变化")
    axes[1, 1].set(xlabel="步长 h", ylabel=r"$|\omega_h-\omega|$", title="d  频率误差的步长敏感性")
    for i, ax in enumerate(axes.flat):
        ax.grid(alpha=.18)
        if i != 2:
            ax.legend(frameon=False, fontsize=8)
    save_figure(fig, "kam_diagnostics")
    print("Generated five tables and three figures from completed results.")


if __name__ == "__main__":
    main()
