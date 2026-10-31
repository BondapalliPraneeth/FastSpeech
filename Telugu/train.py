"""Train Telugu FastSpeech 2.   python train.py [--restore_step N]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fs2.common import load_config  # noqa: E402
from fs2.plots import plot_losses  # noqa: E402
from fs2.trainer import train  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--restore_step", type=int, default=0)
    a = ap.parse_args()
    cfgs = load_config(a.config)
    train(cfgs, a.restore_step)
    plot_losses(cfgs[2]["path"]["log_path"], cfgs[2]["path"]["result_path"], "Telugu")
