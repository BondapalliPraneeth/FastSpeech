"""Synthesise the assignment's English test sentences + variance-adaptor plots.

  python inference.py [--ckpt best|<step>] [--speaker <speaker id from speakers.json>]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import frontend  # noqa: E402
from fs2.common import load_config  # noqa: E402
from fs2.plots import plot_losses  # noqa: E402
from fs2.synth import synthesize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--speaker", default=None)
    ap.add_argument("--text", default=None, help="synthesise a custom sentence instead of the test set")
    a = ap.parse_args()
    cfgs = load_config(a.config)
    sents = [("custom", a.text)] if a.text else json.load(open(os.path.join(HERE, "test_sentences.json"), encoding="utf-8"))
    out = cfgs[2]["path"]["result_path"]
    synthesize(cfgs, sents, frontend.text_to_symbols, out, a.ckpt, a.speaker)
    plot_losses(cfgs[2]["path"]["log_path"], out, "English")
