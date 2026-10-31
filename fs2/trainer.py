"""Training loop (adapted from ming024/FastSpeech2 train.py / evaluate.py).

Changes: single config.yaml, CSV loss logging (train + validation) and loss-curve PNGs required by the
assignment, no DataParallel / vocoder synthesis during training, best-validation checkpoint tracking.
"""
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fs2.common import get_model, get_param_num, latest_step
from fs2.dataset import Dataset
from fs2.model import FastSpeech2Loss
from fs2.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def evaluate(model, configs, loss_fn):
    pre, model_cfg, train_cfg = configs
    ds = Dataset("val.txt", pre, train_cfg, sort=False, drop_last=False)
    loader = DataLoader(ds, batch_size=train_cfg["optimizer"]["batch_size"] * 2,
                        shuffle=False, collate_fn=ds.collate_fn)
    sums = [0.0] * 6
    model.eval()
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                out = model(*batch[2:])
                losses = loss_fn(batch, out)
            for i in range(6):
                sums[i] += losses[i].item() * len(batch[0])
    model.train()
    return [s / len(ds) for s in sums]


def train(configs, restore_step=0):
    pre, model_cfg, train_cfg = configs
    ocfg, scfg = train_cfg["optimizer"], train_cfg["step"]
    for p in train_cfg["path"].values():
        os.makedirs(p, exist_ok=True)
    log_dir = train_cfg["path"]["log_path"]

    ds = Dataset("train.txt", pre, train_cfg, sort=True, drop_last=True)
    group = 4  # batches are length-sorted within groups of 4 batches
    loader = DataLoader(ds, batch_size=ocfg["batch_size"] * group, shuffle=True, collate_fn=ds.collate_fn)

    model, optim = get_model(configs, device, restore_step, train=True)
    loss_fn = FastSpeech2Loss(pre, model_cfg).to(device)
    print("device:", device, "| params: {:.2f}M | train utts: {}".format(get_param_num(model) / 1e6, len(ds)))

    new_logs = restore_step == 0
    tr_f = open(os.path.join(log_dir, "train_loss.csv"), "w" if new_logs else "a", newline="")
    va_f = open(os.path.join(log_dir, "val_loss.csv"), "w" if new_logs else "a", newline="")
    tr_w, va_w = csv.writer(tr_f), csv.writer(va_f)
    if new_logs:
        tr_w.writerow(["step"] + NAMES)
        va_w.writerow(["step"] + NAMES)

    step = restore_step + 1
    total, log_step, val_step, save_step = scfg["total_step"], scfg["log_step"], scfg["val_step"], scfg["save_step"]
    acc = ocfg["grad_acc_step"]
    best = float("inf")
    bar = tqdm(total=total, initial=restore_step, desc="train")
    run = [0.0] * 6
    n_run = 0
    t0 = time.time()
    while step <= total:
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                out = model(*batch[2:])
                losses = loss_fn(batch, out)
                (losses[0] / acc).backward()
                if step % acc == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), ocfg["grad_clip_thresh"])
                    optim.step_and_update_lr()
                    optim.zero_grad()

                for i in range(6):
                    run[i] += losses[i].item()
                n_run += 1
                if step % log_step == 0:
                    avg = [r / n_run for r in run]  # mean over the last log_step steps (less noisy)
                    tr_w.writerow([step] + ["%.5f" % a for a in avg])
                    tr_f.flush()
                    bar.write("step {} | total {:.3f} mel {:.3f} post {:.3f} pitch {:.3f} energy {:.3f} dur {:.3f} | {:.1f}s".format(
                        step, *avg, time.time() - t0))
                    run, n_run = [0.0] * 6, 0
                if step % val_step == 0:
                    v = evaluate(model, configs, loss_fn)
                    va_w.writerow([step] + ["%.5f" % a for a in v])
                    va_f.flush()
                    bar.write("VAL step {} | total {:.3f} mel {:.3f} post {:.3f} pitch {:.3f} energy {:.3f} dur {:.3f}".format(step, *v))
                    if v[0] < best:
                        best = v[0]
                        torch.save({"model": model.state_dict(), "optimizer": optim._optimizer.state_dict()},
                                   os.path.join(train_cfg["path"]["ckpt_path"], "best.pth.tar"))
                if step % save_step == 0 or step == total:
                    torch.save({"model": model.state_dict(), "optimizer": optim._optimizer.state_dict()},
                               os.path.join(train_cfg["path"]["ckpt_path"], "{}.pth.tar".format(step)))
                bar.update(1)
                if step == total:
                    tr_f.close()
                    va_f.close()
                    return
                step += 1
