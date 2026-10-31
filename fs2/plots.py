"""Training / validation loss-curve plots from the CSV logs written by fs2.trainer."""
import csv
import os

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

COMP = ["mel", "mel_postnet", "pitch", "energy", "duration"]


def _read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]} if rows else {}


def plot_losses(log_dir, out_dir, language):
    os.makedirs(out_dir, exist_ok=True)
    for name, title in (("train", "Training"), ("val", "Validation")):
        d = _read(os.path.join(log_dir, name + "_loss.csv"))
        if not d:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax[0].plot(d["step"], d["total"], color="k", lw=2)
        ax[0].set_title("{} total loss ({})".format(title, language))
        ax[0].set_xlabel("step")
        ax[0].set_ylabel("loss")
        ax[0].grid(alpha=0.3)
        for c in COMP:
            ax[1].plot(d["step"], d[c], label=c)
        ax[1].set_yscale("log")
        ax[1].set_title("{} loss components".format(title))
        ax[1].set_xlabel("step")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_loss.png" if name == "train" else "validation_loss.png"), dpi=140)
        plt.close(fig)
