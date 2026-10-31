# Progress / handoff (saved 2026-10-31)

## Done
- Shared code `fs2/` (adapted from ming024/FastSpeech2), per-language `English/`, `Telugu/` (config.yaml, preprocess.py, train.py, inference.py, frontend.py, test_sentences.json)
- README.md, requirements.txt, FastSpeech2_Colab.ipynb, .gitignore
- Smoke-tested Telugu pipeline locally (fake alignments, 40 steps): features, training, inference, loss plots, variance plots all work.
- Telugu raw data downloaded to data/telugu_raw (SPRINGLab/IndicTTS_Telugu). Telugu speaker = gender 1 (6.9 h of 1-12 s clips).

## NOT done (needs Colab GPU + user's HF login for gated ai4bharat/Svarah)
1. Push repo to GitHub, edit REPO in the notebook.
2. Run notebook: MFA (condacolab), preprocess, train (~1-2 h/lang on T4), inference.
3. Copy Results/, checkpoints/ back into project (Git LFS or Drive link for checkpoints).
4. Write analysis + Report/Technical_Report.pdf from the real results (losses, pronunciation, duration/pitch/energy plots, EN vs TE comparison).

## Untested risks
- MFA install/run on Colab (never run locally); English path (Svarah column names are auto-detected, schema unseen).
- Telugu test sentences use corrected text (PDF text layer is garbled) - mention in report.
