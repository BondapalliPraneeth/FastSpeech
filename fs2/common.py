"""Config loading, model / vocoder construction (adapted from ming024/FastSpeech2 utils/model.py)."""
import json
import os
import zipfile

import torch
import yaml

from fs2 import hifigan
from fs2.model import FastSpeech2, ScheduledOptim
from fs2.tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path):
    """Single config.yaml per language -> (preprocess, model, train) dicts.

    Relative paths inside the config are resolved against the config file's directory,
    so scripts work from any working directory.
    """
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pre, model, train = cfg["preprocess"], cfg["model"], cfg["train"]
    for sect in (pre["path"], train["path"]):
        for k, v in sect.items():
            if isinstance(v, str) and v.startswith(("./", "../")):
                sect[k] = os.path.normpath(os.path.join(base, v))
    # vocabulary size is language specific (ARPAbet phones for English, graphemes for Telugu)
    sym_file = os.path.join(pre["path"]["preprocessed_path"], "symbols.json")
    if os.path.exists(sym_file):
        model["n_vocab"] = Tokenizer(pre["path"]["preprocessed_path"]).vocab_size
    return pre, model, train


def latest_step(ckpt_dir):
    steps = [int(f.split(".")[0]) for f in os.listdir(ckpt_dir) if f.endswith(".pth.tar")]
    return max(steps) if steps else 0


def get_model(configs, device, restore_step=0, train=False):
    pre, model_cfg, train_cfg = configs
    model = FastSpeech2(pre, model_cfg).to(device)
    ckpt = None
    if restore_step:
        ckpt = torch.load(
            os.path.join(train_cfg["path"]["ckpt_path"], "{}.pth.tar".format(restore_step)),
            map_location=device,
        )
        model.load_state_dict(ckpt["model"])
    if train:
        optim = ScheduledOptim(model, train_cfg, model_cfg, restore_step)
        if ckpt is not None:
            optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, optim
    model.eval()
    return model


def get_param_num(model):
    return sum(p.numel() for p in model.parameters())


def get_vocoder(device):
    """Universal HiFi-GAN (22.05 kHz, 80 mel, fmax 8 kHz) shipped with the reference repo."""
    hdir = os.path.join(HERE, "hifigan")
    with open(os.path.join(hdir, "config.json")) as f:
        cfg = hifigan.AttrDict(json.load(f))
    ckpt_path = os.path.join(hdir, "generator_universal.pth.tar")
    if not os.path.exists(ckpt_path):
        with zipfile.ZipFile(ckpt_path + ".zip") as z:
            z.extractall(hdir)
    vocoder = hifigan.Generator(cfg)
    vocoder.load_state_dict(torch.load(ckpt_path, map_location="cpu")["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    return vocoder.to(device)


def vocoder_infer(mels, vocoder, pre_cfg, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)
    wavs = (wavs.cpu().numpy() * pre_cfg["preprocessing"]["audio"]["max_wav_value"]).astype("int16")
    wavs = list(wavs)
    if lengths is not None:
        wavs = [w[: lengths[i]] for i, w in enumerate(wavs)]
    return wavs
