"""Corpus preparation shared by both languages: HF dataset -> 22.05 kHz wavs + .lab + disjoint train/val lists.

Subsampling strategy (identical for both languages, parameters live in config.yaml -> preprocess.corpus):
  1. optional filters (speaker/gender, duration window, transcript sanity filter)
  2. utterances are grouped by normalised transcript, so identical sentences never fall into both splits
  3. groups are shuffled with a fixed seed and consumed until `train_hours` is reached (train),
     then until `val_hours` is reached (validation); everything else is discarded.
"""
import io
import os
import random
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm


def _first(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def load_records(hf_id, split, cfg):
    """Yield dicts(audio_bytes, text, speaker, extra) from a Hugging Face dataset without decoding audio."""
    from datasets import Audio, load_dataset

    ds = load_dataset(hf_id, split=split) if split else load_dataset(hf_id)
    if not split:  # DatasetDict -> concatenate all splits (Svarah only ships a "test" split)
        from datasets import concatenate_datasets
        ds = concatenate_datasets([ds[k] for k in ds.keys()])
    print("columns:", ds.column_names)
    audio_col = cfg.get("audio_column") or _first(ds.column_names, ["audio", "audio_filepath", "wav"])
    text_col = cfg.get("text_column") or _first(ds.column_names, ["text", "transcript", "sentence", "normalized"])
    spk_col = cfg.get("speaker_column") or _first(ds.column_names, ["speaker_id", "speaker", "spk_id", "client_id"])
    ds = ds.cast_column(audio_col, Audio(decode=False))
    print("audio_col={} text_col={} speaker_col={}".format(audio_col, text_col, spk_col))
    for i, ex in enumerate(ds):
        a = ex[audio_col]
        if a.get("bytes") is None and a.get("path"):
            with open(a["path"], "rb") as f:
                data = f.read()
        else:
            data = a["bytes"]
        yield {
            "idx": i, "audio": data, "text": ex[text_col],
            "speaker": str(ex[spk_col]) if spk_col else None,
            "extra": {k: v for k, v in ex.items() if k not in (audio_col,)},
        }


def decode(data, sr):
    wav, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(1)
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav


def prepare(records, normalize_text, cfg, raw_dir, lists_dir, sr, prefix, default_speaker="spk"):
    """records: iterable from load_records. Writes raw_dir/<speaker>/<id>.wav|.lab and train/val lists."""
    rng = random.Random(cfg.get("seed", 1234))
    lo, hi = cfg.get("min_dur", 1.0), cfg.get("max_dur", 12.0)
    keep = cfg.get("filter", lambda r: True)
    cands = []
    for r in tqdm(records, desc="scan"):
        if not keep(r):
            continue
        text = normalize_text(r["text"])
        if not text or len(text.split()) < cfg.get("min_words", 1):
            continue
        info = sf.info(io.BytesIO(r["audio"]))
        dur = info.frames / info.samplerate
        if not lo <= dur <= hi:
            continue
        cands.append({**r, "text_norm": text, "dur": dur, "speaker": r["speaker"] or default_speaker})

    max_spk = cfg.get("max_speakers")
    if max_spk:  # keep the speakers with the most data
        tot = defaultdict(float)
        for c in cands:
            tot[c["speaker"]] += c["dur"]
        top = {s for s, _ in sorted(tot.items(), key=lambda kv: -kv[1])[:max_spk]}
        cands = [c for c in cands if c["speaker"] in top]
    print("candidates: {} utts, {:.2f} h".format(len(cands), sum(c["dur"] for c in cands) / 3600))

    groups = defaultdict(list)
    for c in cands:
        groups[c["text_norm"]].append(c)
    keys = sorted(groups)
    rng.shuffle(keys)
    split = {"train": [], "val": []}
    budget = {"train": cfg["train_hours"] * 3600, "val": cfg["val_hours"] * 3600}
    cur, used = "train", 0.0
    for k in keys:
        if used >= budget[cur]:
            if cur == "val":
                break
            cur, used = "val", 0.0
        for c in groups[k]:
            split[cur].append(c)
            used += c["dur"]

    os.makedirs(lists_dir, exist_ok=True)
    lists = {"train": [], "val": []}
    for name, items in split.items():
        for c in tqdm(items, desc="write " + name):
            spk = "".join(ch if ch.isalnum() else "_" for ch in c["speaker"])
            base = "{}_{:06d}".format(prefix, c["idx"])
            d = os.path.join(raw_dir, spk)
            os.makedirs(d, exist_ok=True)
            sf.write(os.path.join(d, base + ".wav"), decode(c["audio"], sr), sr, subtype="PCM_16")
            with open(os.path.join(d, base + ".lab"), "w", encoding="utf-8") as f:
                f.write(c["text_norm"] + "\n")
            lists[name].append("{}|{}".format(spk, base))
    for name, l in lists.items():
        with open(os.path.join(lists_dir, name + "_list.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(l) + "\n")
        print("{}: {} utts, {:.2f} h".format(name, len(l), sum(c["dur"] for c in split[name]) / 3600))
    tr_t = {c["text_norm"] for c in split["train"]}
    assert not any(c["text_norm"] in tr_t for c in split["val"]), "transcript overlap between train and val"
    return split
