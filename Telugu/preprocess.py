"""Telugu preprocessing: IndicTTS Telugu -> wav/lab -> MFA trained from scratch (grapheme dictionary) -> features.

  python preprocess.py prepare | align | features | all

There is no pretrained Telugu MFA acoustic model, so `align` runs `mfa train` on the selected ~6 h of
speech (train+val) with a grapheme "pronunciation" dictionary and uses the resulting alignments.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import frontend  # noqa: E402
from fs2 import corpus, pipeline  # noqa: E402
from fs2.common import load_config  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
# every assigned code point of the Telugu Unicode block (so rarely-seen letters still have an id)
TELUGU_BLOCK = [chr(c) for c in range(0x0C00, 0x0C80) if chr(c).isprintable() and chr(c) not in "‌‍"]
BASE_SYMBOLS = ["sil", "sp", "spn"] + [c for c in TELUGU_BLOCK if c.strip()]


def prepare(pre):
    cfg = dict(pre["corpus"])
    gender = cfg.pop("gender", None)
    if gender is not None:
        cfg["filter"] = lambda r: r["extra"].get("gender") == gender
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    recs = corpus.load_records(cfg["hf_id"], cfg.get("split"), cfg)
    corpus.prepare(recs, frontend.normalize, cfg, pre["path"]["raw_path"], pre["path"]["preprocessed_path"],
                   sr, "te", default_speaker="te_spk{}".format(gender))
    pipeline.write_dictionary(pre["path"]["raw_path"], pre["path"]["mfa_dict_path"], frontend.word_pron)


def align(pre, jobs):
    out = os.path.join(pre["path"]["preprocessed_path"], "TextGrid")
    model_zip = os.path.join(pre["path"]["preprocessed_path"], "telugu_mfa_acoustic.zip")
    pipeline.run(["mfa", "train", pre["path"]["raw_path"], pre["path"]["mfa_dict_path"], out,
                  "--clean", "-j", str(jobs), "--output_model_path", model_zip])


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
