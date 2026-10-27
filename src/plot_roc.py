"""Grafico ROC log-log degli attacchi (lo standard di Carlini et al. 2022)."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results"
rep = json.loads((OUT / "attack_report.json").read_text())

COL = {"Threshold": "#888888", "Shadow": "#2b6cb0", "LiRA": "#c0392b"}
fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
for ax, name in zip(axes, ["normal", "overfit"]):
    t = rep["targets"][name]
    for atk, roc in t["roc"].items():
        fpr, tpr = np.array(roc["fpr"]), np.array(roc["tpr"])
        a = t["attacks"][atk]["auc"]
        ax.plot(fpr, tpr, color=COL.get(atk), lw=1.8, label=f"{atk} (AUC={a:.3f})")
    ax.plot([1e-4, 1], [1e-4, 1], "k--", lw=1, label="random")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-3, 1); ax.set_ylim(1e-3, 1)
    gap = None
    mc = t.get("metrics_check")
    if mc:
        gap = round(mc["members"]["accuracy"] - mc["non_members"]["accuracy"], 3)
    ax.set_title(f"target_{name}" + (f"  (gap accuracy = {gap})" if gap is not None else ""))
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.grid(True, which="both", alpha=0.3); ax.legend(loc="upper left", fontsize=8)
fig.suptitle("Membership Inference — ROC log-log (reo-confesso)", fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "roc_loglog.png", dpi=150)
print("salvato", OUT / "roc_loglog.png")
