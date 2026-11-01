"""Plot saved diagnostic measurements without generating or altering results."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "manuscript" / "figures"


def read_csv(name):
    with (RESULTS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    convergence = read_csv("convergence.csv")
    structure = read_csv("structure_defects.csv")
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none",
                         "axes.unicode_minus": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), layout="constrained")
    names = {"linear_conservative": "Constant mass / conservative", "linear_damped": "Constant mass / damped",
             "variable_mass_conservative": "Variable mass / conservative", "variable_mass_damped": "Variable mass / damped"}
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for (case, label), color, marker in zip(names.items(), colors, ["o", "s", "^", "D"]):
        rows = [row for row in convergence if row["case"] == case]
        h = np.array([float(r["h"]) for r in rows])
        err = np.array([float(r["state_error"]) for r in rows])
        axes[0].loglog(h, err, marker=marker, color=color, label=label, linewidth=1.5, markersize=4)
    axes[0].loglog(h, .3 * h ** 2, "--", color="#555555", label=r"$O(h^2)$", linewidth=1)
    axes[0].set_title("(a) Fixed-horizon convergence")
    axes[0].set_xlabel(r"Step $h$")
    axes[0].set_ylabel("Endpoint state error")
    axes[0].legend(fontsize=8, frameon=False)
    rows = [row for row in structure if float(row["damping"]) == .2]
    h = np.array([float(r["h"]) for r in rows])
    measured = np.array([float(r["continuous_birkhoff_defect_relative"]) for r in rows])
    exact = np.array([float(r["exact_flow_birkhoff_defect_relative"]) for r in rows])
    axes[1].loglog(h, measured, "o-", color="#D55E00", label="Eq. (36)")
    axes[1].loglog(h, exact, "s--", color="#0072B2", label="Exact flow")
    axes[1].set_title("(b) Damped Birkhoff defect")
    axes[1].set_xlabel(r"Step $h$")
    axes[1].set_ylabel("Relative structure defect")
    axes[1].legend(fontsize=9, frameon=False)
    for ax in axes:
        ax.grid(True, which="major", alpha=.18)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "equation36_diagnostics.png", dpi=220, facecolor="white")
    fig.savefig(OUT / "equation36_diagnostics.pdf", facecolor="white")
    fig.savefig(OUT / "equation36_diagnostics.svg", facecolor="white")
    plt.close(fig)
    print(str(OUT / "equation36_diagnostics.png"))


if __name__ == "__main__":
    main()
