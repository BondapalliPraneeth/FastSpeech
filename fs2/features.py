"""Feature extraction from MFA TextGrids: durations, mel, pitch (F0), energy.

Adapted from ming024/FastSpeech2 preprocessor/preprocessor.py. Changes:
  * train / validation membership is decided up-front (disjoint lists) instead of a random split
  * normalisation statistics (pitch/energy mean+std) are estimated on TRAIN utterances only
  * a symbols.json vocabulary is written so the same code serves phones (English) and graphemes (Telugu)
  * fixed a bug in build_from_path where stats of a skipped utterance were re-used
"""
import json
import os

import librosa
import numpy as np
import pyworld as pw
import tgt
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import fs2.audio as Audio

SIL = ["sil", "sp", "spn"]


class Features:
    def __init__(self, config, base_symbols):
        self.cfg = config
        p = config["preprocessing"]
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.sr = p["audio"]["sampling_rate"]
        self.hop = p["stft"]["hop_length"]
        self.pitch_avg = p["pitch"]["feature"] == "phoneme_level"
        self.energy_avg = p["energy"]["feature"] == "phoneme_level"
        self.pitch_norm = p["pitch"]["normalization"]
        self.energy_norm = p["energy"]["normalization"]
        self.base_symbols = base_symbols
        self.STFT = Audio.stft.TacotronSTFT(
            p["stft"]["filter_length"], p["stft"]["hop_length"], p["stft"]["win_length"],
            p["mel"]["n_mel_channels"], p["audio"]["sampling_rate"],
            p["mel"]["mel_fmin"], p["mel"]["mel_fmax"],
        )

    def run(self):
        for d in ("mel", "pitch", "energy", "duration"):
            os.makedirs(os.path.join(self.out_dir, d), exist_ok=True)
        splits = {}
        for name in ("train", "val"):
            with open(os.path.join(self.out_dir, name + "_list.txt"), encoding="utf-8") as f:
                splits[name] = [l.strip().split("|") for l in f if l.strip()]  # speaker|basename
        assert not ({b for _, b in splits["train"]} & {b for _, b in splits["val"]}), "train/val overlap!"

        speakers = {s: i for i, s in enumerate(sorted({s for sp in splits.values() for s, _ in sp}))}
        pitch_sc, energy_sc = StandardScaler(), StandardScaler()
        meta = {"train": [], "val": []}
        frames = {"train": 0, "val": 0}
        used_phones = set()
        for name in ("train", "val"):
            for spk, base in tqdm(splits[name], desc="features/" + name):
                tg = os.path.join(self.out_dir, "TextGrid", spk, base + ".TextGrid")
                if not os.path.exists(tg):
                    continue  # MFA failed to align this file
                ret = self.process_utterance(spk, base)
                if ret is None:
                    continue
                line, pitch, energy, n, phones = ret
                meta[name].append(line)
                frames[name] += n
                used_phones.update(phones)
                if name == "train":
                    if len(pitch):
                        pitch_sc.partial_fit(pitch.reshape(-1, 1))
                    if len(energy):
                        energy_sc.partial_fit(energy.reshape(-1, 1))

        p_mean, p_std = (pitch_sc.mean_[0], pitch_sc.scale_[0]) if self.pitch_norm else (0, 1)
        e_mean, e_std = (energy_sc.mean_[0], energy_sc.scale_[0]) if self.energy_norm else (0, 1)
        p_min, p_max = self.normalize(os.path.join(self.out_dir, "pitch"), p_mean, p_std)
        e_min, e_max = self.normalize(os.path.join(self.out_dir, "energy"), e_mean, e_std)

        symbols = sorted(set(self.base_symbols) | used_phones)
        with open(os.path.join(self.out_dir, "symbols.json"), "w", encoding="utf-8") as f:
            json.dump(symbols, f, ensure_ascii=False)
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            json.dump(speakers, f)
        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            json.dump({"pitch": [float(p_min), float(p_max), float(p_mean), float(p_std)],
                       "energy": [float(e_min), float(e_max), float(e_mean), float(e_std)]}, f)
        for name in ("train", "val"):
            with open(os.path.join(self.out_dir, name + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(meta[name]) + "\n")
            print("{}: {} utts, {:.2f} h".format(name, len(meta[name]),
                                                 frames[name] * self.hop / self.sr / 3600))

    def process_utterance(self, speaker, basename):
        wav_path = os.path.join(self.in_dir, speaker, basename + ".wav")
        text_path = os.path.join(self.in_dir, speaker, basename + ".lab")
        tg_path = os.path.join(self.out_dir, "TextGrid", speaker, basename + ".TextGrid")
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end = self.get_alignment(textgrid.get_tier_by_name("phones"))
        if start >= end or not phone:
            return None

        wav, _ = librosa.load(wav_path, sr=self.sr)
        wav = wav[int(self.sr * start): int(self.sr * end)].astype(np.float32)
        with open(text_path, encoding="utf-8") as f:
            raw_text = f.readline().strip("\n")

        w64 = wav.astype(np.float64)
        pitch, t = pw.dio(w64, self.sr, frame_period=self.hop / self.sr * 1000)
        pitch = pw.stonemask(w64, pitch, t, self.sr)
        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        mel, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel, energy = mel[:, : sum(duration)], energy[: sum(duration)]

        if self.pitch_avg:
            nz = np.where(pitch != 0)[0]
            fn = interp1d(nz, pitch[nz], fill_value=(pitch[nz[0]], pitch[nz[-1]]), bounds_error=False)
            pitch = fn(np.arange(len(pitch)))
            pos = 0
            for i, d in enumerate(duration):
                pitch[i] = np.mean(pitch[pos: pos + d]) if d > 0 else 0
                pos += d
            pitch = pitch[: len(duration)]
        if self.energy_avg:
            pos = 0
            for i, d in enumerate(duration):
                energy[i] = np.mean(energy[pos: pos + d]) if d > 0 else 0
                pos += d
            energy = energy[: len(duration)]

        for tag, arr in (("duration", duration), ("pitch", pitch), ("energy", energy), ("mel", mel.T)):
            np.save(os.path.join(self.out_dir, tag, "{}-{}-{}.npy".format(speaker, tag, basename)), arr)
        line = "|".join([basename, speaker, "{" + " ".join(phone) + "}", raw_text])
        return line, self.remove_outlier(pitch), self.remove_outlier(energy), mel.shape[1], phone

    def get_alignment(self, tier):
        phones, durations = [], []
        start_time = end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            if not phones:
                if p in SIL:
                    continue
                start_time = s
            phones.append(p)
            if p not in SIL:
                end_time, end_idx = e, len(phones)
            durations.append(int(np.round(e * self.sr / self.hop) - np.round(s * self.sr / self.hop)))
        return phones[:end_idx], durations[:end_idx], start_time, end_time

    @staticmethod
    def remove_outlier(values):
        values = np.array(values)
        p25, p75 = np.percentile(values, 25), np.percentile(values, 75)
        lo, hi = p25 - 1.5 * (p75 - p25), p75 + 1.5 * (p75 - p25)
        return values[(values > lo) & (values < hi)]

    @staticmethod
    def normalize(in_dir, mean, std):
        mx, mn = -np.inf, np.inf
        for fn in os.listdir(in_dir):
            fp = os.path.join(in_dir, fn)
            v = (np.load(fp) - mean) / std
            np.save(fp, v)
            mx, mn = max(mx, v.max()), min(mn, v.min())
        return mn, mx
