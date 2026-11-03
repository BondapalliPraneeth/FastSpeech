# FastSpeech 2 Text-to-Speech: English (Svarah) and Telugu (IndicTTS)

Two separately trained [FastSpeech 2](https://arxiv.org/abs/2006.04558) models, one per language, built on
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence, kept in `fs2/LICENSE_FastSpeech2`).
Each model is trained on about 5 h of speech and validated on a separate 1 h.

> **Status:** code, configs and the Colab notebook are complete and smoke-tested. The full training runs, checkpoints,
> `Results/` and `Report/Technical_Report.pdf` are produced by running the notebook on a GPU. They are added to this
> repo after that run, so the corresponding folders may still be empty.

## Repository layout

```
English/                 config.yaml  preprocess.py  train.py  inference.py  frontend.py  test_sentences.json  checkpoints/
Telugu/                  (same files)
fs2/                     shared code adapted from the reference repo (model, transformer, hifigan, audio) + new modules
Results/<Language>/      training_loss.png  validation_loss.png  test_generated_audios/
                         variance_adaptor_plots/{duration,pitch,energy}/
Report/                  Technical_Report.pdf
FastSpeech2_Colab.ipynb  end-to-end pipeline for a Colab GPU
requirements.txt  PROGRESS.md
```

## Quick start (Google Colab, T4 GPU)

1. Open `FastSpeech2_Colab.ipynb`, set `Runtime -> GPU`, and set `REPO` in the first cell to this repository's URL.
2. Accept the terms of the gated dataset [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) on Hugging Face, then log in when the notebook asks.
3. Run all cells. Montreal Forced Aligner is installed through condacolab, which restarts the kernel once.
   Afterwards re-run from the `%cd` cell and skip the install cell.
4. Expect about 1-2 h of training per language on a T4. Download the zip produced in the last cell.

## Manual commands

```bash
pip install -r requirements.txt
mamba install -c conda-forge montreal-forced-aligner     # MFA is not pip-installable
python -c "import nltk; [nltk.download(p) for p in ('averaged_perceptron_tagger_eng','cmudict','punkt')]"

cd English
python preprocess.py all -j 4        # or: prepare | align | features
python train.py                      # add --restore_step N to resume
python inference.py                  # writes ../Results/English/...

cd ../Telugu
python preprocess.py all -j 4
python train.py
python inference.py
```

`inference.py --text "..."` synthesises a custom sentence, and `--ckpt <step>` selects a specific checkpoint
(default is `best`, the lowest validation loss).

Telugu plots need a Telugu font (`sudo apt-get install fonts-noto-core` on Colab/Linux; Nirmala UI is used on Windows).

## Data and preprocessing

| | English | Telugu |
|---|---|---|
| Dataset | `ai4bharat/Svarah` (gated, Indian-accented English; no speaker-id column, so treated as one speaker id) | `SPRINGLab/IndicTTS_Telugu` (HF mirror of IndicTTS Telugu: 8,576 utts, 36.7 h, 2 speakers, 48 kHz) |
| Selection | utterances 1.5-12 s, at least 3 words, annotation tags removed | one speaker (`gender=1`), utterances 1-12 s, at least 2 words |
| Why that speaker | all speakers, single speaker id (no speaker labels in Svarah) | speaker 1 has 6.9 h of 1-12 s clips; speaker 0 only 2.3 h |
| Subsampling | 5 h train, 1 h val | 5 h train, 1 h val |
| Alignment | MFA, pretrained `english_us_arpa` | MFA trained from scratch (no pretrained Telugu model) |
| Tokens | stress-marked ARPAbet phones (CMUdict + g2p_en) | Unicode graphemes (code points) |
| Audio | 22.05 kHz, 80 mel bins, hop 256, F0 via WORLD (phoneme-level pitch/energy), universal HiFi-GAN vocoder |

**Split:** utterances are grouped by normalised transcript, the groups are shuffled with seed 1234, and groups are
consumed until 5 h (train) and then 1 h (val) are reached. Identical sentences never appear in both splits, and
`preprocess.py` asserts this. Pitch/energy normalisation statistics are computed on the training split only.

All parameters are in `config.yaml` under `preprocess.corpus`.

## Changes relative to the reference implementation

Reused unchanged in structure: FFT-block encoder/decoder, variance adaptor (duration, pitch, energy predictors),
length regulator, PostNet, loss, optimiser schedule, HiFi-GAN vocoder.

Shared (`fs2/`):
- Absolute imports; the encoder vocabulary size comes from the config (`model.n_vocab`) instead of the English `text.symbols`.
- `tokenizer.py` replaces the English-only `text/` package with a per-language `symbols.json`.
- `features.py` replaces `preprocessor.py`: explicit disjoint train/val lists, train-only statistics, and a fix for a bug
  where a skipped utterance's statistics were re-used.
- Fixed a reference bug where the energy embedding used `p_control` instead of `e_control`.
- librosa >= 0.10 API fixes and a CPU-safe STFT.
- `corpus.py` (Hugging Face data to wavs plus subsampling and splitting), `trainer.py` (CSV loss logs, best-validation
  checkpoint), `plots.py` (loss curves), `synth.py` (inference, duration/pitch/energy plots).
- One `config.yaml` per language replaces the three yaml files.

English-specific: number, time and year normalisation (`num2words`), POS-based homograph handling so that *conduct*
(noun) and *conduct* (verb) get different pronunciations, `max_seq_len` raised to 1500.

Telugu-specific: NFC normalisation with ZWJ/ZWNJ removed, Telugu number/ordinal/time verbalisation (e.g. `2026` as
రెండు వేల ఇరవై ఆరు, `12వ` as పన్నెండవ), grapheme tokens covering the whole Telugu Unicode block, MFA trained
from scratch with a grapheme dictionary, single speaker.

## Test sentences

The sentences are in `English/test_sentences.json` and `Telugu/test_sentences.json` (IDs E1, E2a, E2b, E3, E4 and
T1, T2a, T2b, T3, T4). English text is verbatim, including the `rhyth m` split in E3. The text layer of the assignment
PDF garbles Telugu conjuncts (for example `అడ విలో`, `నిశ్శబ్ంద గా`), so the Telugu file uses the correctly shaped
text as shown in the rendered PDF.

## Outputs

- `Results/<Language>/training_loss.png`, `validation_loss.png` (from `logs/train_loss.csv` and `val_loss.csv`).
- `Results/<Language>/test_generated_audios/<ID>.wav`.
- `Results/<Language>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`, plus `synthesis_report.json` with the
  raw predicted durations, F0 and energy per token.

## Checkpoints

Checkpoints are written to `English/checkpoints/` and `Telugu/checkpoints/` (`best.pth.tar` and periodic steps).
They exceed GitHub's 100 MB limit, so they are tracked with Git LFS (`git lfs track "*.pth.tar"`) or hosted externally.
The download link, if any, will be listed here: _to be added after training_.

## Known limitations

- Telugu uses graphemes rather than phonemes, so schwa/geminate/sandhi effects are learned from data only.
- Characters that never occur in the training text have untrained embeddings.
- Word boundaries are not marked; pauses are only predicted at punctuation (`sp`).
- The universal HiFi-GAN vocoder was not fine-tuned on either dataset.
- Training logs, loss analysis and per-sentence observations are in the technical report once training is complete.
