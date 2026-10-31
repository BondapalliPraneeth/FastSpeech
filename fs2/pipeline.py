"""Language-independent glue used by each language's preprocess.py / train.py / inference.py."""
import glob
import os
import subprocess

from fs2.features import Features


def write_dictionary(raw_dir, dict_path, word_pron):
    """MFA pronunciation dictionary (word<TAB>space separated symbols) for every word in the .lab files."""
    words = set()
    for lab in glob.glob(os.path.join(raw_dir, "*", "*.lab")):
        with open(lab, encoding="utf-8") as f:
            words.update(f.read().split())
    with open(dict_path, "w", encoding="utf-8") as f:
        for w in sorted(words):
            pron = word_pron(w)
            if pron:
                f.write("{}\t{}\n".format(w, " ".join(pron)))
    print("dictionary: {} words -> {}".format(len(words), dict_path))


def run(cmd):
    print("$", " ".join(cmd))
    subprocess.check_call(cmd)


def extract_features(pre_cfg, base_symbols):
    Features(pre_cfg, base_symbols).run()
