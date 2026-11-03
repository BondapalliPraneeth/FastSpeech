"""English preprocessing: Svarah -> wav/lab -> MFA (pretrained english_us_arpa) -> mel/pitch/energy/duration.

  python preprocess.py prepare     # download + subsample + split + write raw corpus and MFA dictionary
  python preprocess.py align       # Montreal Forced Aligner (needs `mfa` on PATH, e.g. conda-forge)
  python preprocess.py features    # TextGrid -> durations, mel, pitch, energy, symbols.json, train/val.txt
  python preprocess.py all
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import yaml  # noqa: E402

import frontend  # noqa: E402
from fs2 import corpus, pipeline  # noqa: E402
from fs2.common import load_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_SYMBOLS = ["sil", "sp", "spn"] + frontend.ARPA


def clean_transcript(text):
    t = re.sub(r"<[^>]*>|\[[^\]]*\]|\([^)]*\)", " ", text)   # annotation tags / noise markers
    return frontend.normalize(t).lower()


def prepare(pre):
    cfg = dict(pre["corpus"])
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    recs = corpus.load_records(cfg["hf_id"], cfg.get("split"), cfg)
    corpus.prepare(recs, clean_transcript, cfg, pre["path"]["raw_path"], pre["path"]["preprocessed_path"], sr, "svarah")
    pipeline.write_dictionary(pre["path"]["raw_path"], pre["path"]["mfa_dict_path"],
                              lambda w: frontend.word_pron(w))


def align(pre, jobs):
    out = os.path.join(pre["path"]["preprocessed_path"], "TextGrid")
    pipeline.run(["mfa", "model", "download", "acoustic", "english_us_arpa"])
    # positional arguments first: `mfa align` ignores unknown options, so a leading "--beam" would be taken as the corpus dir
    pipeline.run(["mfa", "align", pre["path"]["raw_path"], pre["path"]["mfa_dict_path"], "english_us_arpa", out,
                  "--clean", "-j", str(jobs), "--beam", "100", "--retry_beam", "400"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "align", "features", "all"])
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("-j", "--jobs", type=int, default=4)
    a = ap.parse_args()
    pre, _, _ = load_config(a.config)
    os.makedirs(pre["path"]["preprocessed_path"], exist_ok=True)
    if a.stage in ("prepare", "all"):
        prepare(pre)
    if a.stage in ("align", "all"):
        align(pre, a.jobs)
    if a.stage in ("features", "all"):
        pipeline.extract_features(pre, BASE_SYMBOLS)
