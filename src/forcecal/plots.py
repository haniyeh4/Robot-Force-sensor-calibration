import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import FIGURES_DIR  # noqa: E402


def pred_vs_meas(measured, predicted, labels, title, fname, xlabel="Sample", unit="N", lines=False):
    fig, axes = plt.subplots(len(labels), 1, figsize=(9, 2.6 * len(labels)), sharex=True)
    style = dict(lw=1) if lines else dict(marker="o", ms=3, lw=0.8)
    for k, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(measured[:, k], label="Measured", color="0.2", **style)
        ax.plot(predicted[:, k], label="Predicted", color="tab:orange", ls="--", **style)
        ax.set_ylabel(f"{lab} [{unit}]")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    axes[-1].set_xlabel(xlabel)
    _save(fig, fname)


def errors(err, labels, title, fname, xlabel="Sample", unit="N", lines=False):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for k, lab in enumerate(labels):
        if lines:
            ax.plot(err[:, k], lw=0.8, label=lab)
        else:
            ax.plot(err[:, k], marker="o", ms=3, lw=0.8, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"Predicted - measured [{unit}]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    _save(fig, fname)


def _save(fig, fname):
    FIGURES_DIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname, dpi=200)
    plt.close(fig)
    print(f"wrote figures/{fname}")
