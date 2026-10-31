"""Inference + variance-adaptor visualisation (duration / pitch / energy)."""
import json
import os
import unicodedata

import matplotlib
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib import pyplot as plt
from scipy.io import wavfile

from fs2.common import get_model, get_vocoder, vocoder_infer
from fs2.tokenizer import Tokenizer

matplotlib.use("Agg")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TELUGU_FONTS = ["Noto Sans Telugu", "Nirmala UI", "Gautami", "Lohit Telugu", "Pothana2000", "Vani"]


def pick_font():
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in TELUGU_FONTS:
        if name in have:
            plt.rcParams["font.family"] = name
            return name
    return None


def _denorm(x, mean, std):
    return x * std + mean


@torch.no_grad()
def synthesize(configs, sentences, frontend, out_dir, ckpt_step=None, speaker=None,
               p_control=1.0, e_control=1.0, d_control=1.0):
    """sentences: list of (id, text). frontend(text) -> list of phone/grapheme symbols.

    Writes <out_dir>/test_generated_audios/<id>.wav and
           <out_dir>/variance_adaptor_plots/{duration,pitch,energy}/<id>.png
    """
    pre, model_cfg, train_cfg = configs
    pick_font()
    ckpt_dir = train_cfg["path"]["ckpt_path"]
    if ckpt_step is None:
        ckpt_step = "best"
    # get_model expects int steps; best.pth.tar is loaded manually
    model = get_model(configs, device, 0, train=False)
    ck = torch.load(os.path.join(ckpt_dir, "{}.pth.tar".format(ckpt_step)), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    vocoder = get_vocoder(device)
    tok = Tokenizer(pre["path"]["preprocessed_path"])
    with open(os.path.join(pre["path"]["preprocessed_path"], "stats.json")) as f:
        stats = json.load(f)
    with open(os.path.join(pre["path"]["preprocessed_path"], "speakers.json")) as f:
        spk_map = json.load(f)
    spk_id = spk_map[speaker] if speaker in spk_map else 0
    p_mean, p_std = stats["pitch"][2:]
    e_mean, e_std = stats["energy"][2:]
    sr = pre["preprocessing"]["audio"]["sampling_rate"]
    hop = pre["preprocessing"]["stft"]["hop_length"]

    audio_dir = os.path.join(out_dir, "test_generated_audios")
    plot_dirs = {k: os.path.join(out_dir, "variance_adaptor_plots", k) for k in ("duration", "pitch", "energy")}
    for d in [audio_dir, *plot_dirs.values()]:
        os.makedirs(d, exist_ok=True)

    report = {}
    for sid, text in sentences:
        symbols = frontend(text)
        unk = [s for s in symbols if s not in tok.sym2id]
        ids = [tok.sym2id[s] for s in symbols if s in tok.sym2id]
        x = torch.LongTensor([ids]).to(device)
        out = model(torch.LongTensor([spk_id]).to(device), x, torch.LongTensor([len(ids)]).to(device),
                    len(ids), p_control=p_control, e_control=e_control, d_control=d_control)
        mel, post, p_pred, e_pred, logd, d_round = out[:6]
        mel_len = int(out[9][0])
        wav = vocoder_infer(post.transpose(1, 2), vocoder, pre, lengths=[mel_len * hop])[0]
        wavfile.write(os.path.join(audio_dir, sid + ".wav"), sr, wav)

        labels = [s for s in symbols if s in tok.sym2id]
        shown = ["◌" + s if unicodedata.category(s[0]).startswith("M") else s for s in labels]  # dotted circle for combining marks
        dur = d_round[0].cpu().numpy().astype(int)
        pitch_hz = _denorm(p_pred[0].cpu().numpy(), p_mean, p_std)
        energy = _denorm(e_pred[0].cpu().numpy(), e_mean, e_std)
        _plot_duration(shown, dur, hop, sr, text, sid, os.path.join(plot_dirs["duration"], sid + ".png"))
        _plot_contour(shown, dur, pitch_hz, "F0 (Hz)", "tomato", text, sid, os.path.join(plot_dirs["pitch"], sid + ".png"))
        _plot_contour(shown, dur, energy, "Energy (L2 norm of mel-STFT)", "darkviolet", text, sid, os.path.join(plot_dirs["energy"], sid + ".png"))
        report[sid] = {
            "text": text, "n_tokens": len(labels), "unknown_symbols": unk,
            "duration_s": round(mel_len * hop / sr, 3),
            "tokens": labels, "durations_frames": dur.tolist(),
            "pitch_hz": [round(float(v), 1) for v in pitch_hz],
            "energy": [round(float(v), 3) for v in energy],
        }
        print(sid, "->", round(mel_len * hop / sr, 2), "s", "| unknown:", unk)
    with open(os.path.join(out_dir, "synthesis_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report


def _title(sid, text):
    return "{}: {}".format(sid, text if len(text) < 90 else text[:87] + "...")


def _plot_duration(shown, dur, hop, sr, text, sid, path):
    ms = dur * hop / sr * 1000
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.28), 4))
    colors = ["#bbbbbb" if l in ("sil", "sp", "spn") else "#2a7ab9" for l in labels]
    ax.bar(range(len(labels)), ms, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_ylabel("Predicted duration (ms)")
    ax.set_title(_title(sid, text) + "  | total {:.2f}s".format(ms.sum() / 1000), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_contour(shown, dur, values, ylabel, color, text, sid, path):
    """Phoneme-level predictions drawn as a step contour on the frame axis (time-aligned)."""
    bounds = np.concatenate([[0], np.cumsum(dur)])
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.28), 4))
    ax.step(bounds, np.append(values, values[-1]), where="post", color=color)
    ax.plot((bounds[:-1] + bounds[1:]) / 2, values, "o", ms=3, color=color)
    for b in bounds:
        ax.axvline(b, color="#dddddd", lw=0.5, zorder=0)
    centers = (bounds[:-1] + bounds[1:]) / 2
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_xlim(0, bounds[-1])
    ax.set_ylabel(ylabel)
    ax.set_xlabel("token (width = predicted duration)")
    ax.set_title(_title(sid, text), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
