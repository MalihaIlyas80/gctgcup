# GC-TGCUP — Full Procedure (GitHub + Kaggle GPU)

Goal: beat TG-CUP. GraphCodeBERT is mandatory and fine-tuned. CPU is too slow, so
training runs on Kaggle GPU. The big `cup2_dataset` stays on Kaggle (already
uploaded); GitHub holds only the small code repo.

---

## What changed in the code (why results will improve)

1. **Closed vocabulary (biggest fix).** Old code loaded `mix_vocab.json` (~100k
   tokens). With a few thousand samples the output softmax never trained → BLEU
   collapsed. Now the vocab is built from training comments (~8–15k tokens), so
   generation is actually learnable.
2. **Pointer-generator copy head.** The decoder can now COPY tokens from the old
   comment + code-edit sequence. Since a new comment ≈ old comment + small edits,
   this directly lifts BLEU / GLEU / METEOR / SARI.
3. **Weight tying** between input and output embeddings (better small-data generation).
4. **Token-space evaluation.** Predictions and references are compared in the same
   tokenized space, so exact-match / BLEU are fair (no spacing mismatch).
5. **GPU config** (`configs/kaggle_gpu.yaml`): GraphCodeBERT **unfrozen**, hidden 256,
   4+4 layers, GGNN 6 steps, more data.

> Note: your downloaded `best.pt` was trained on the OLD architecture and will NOT
> load anymore. The Kaggle run trains a fresh `best.pt` — that is expected.

---

## STEP A — Push the code to GitHub (run on YOUR PC, in `E:\GCB`)

`.gitignore` already excludes `cup2_dataset/`, `data/processed/`, and `*.pt`, so the
repo stays small (only code).

```powershell
cd E:\GCB
git add -A
git commit -m "Add copy mechanism, closed vocab, GPU config to beat TG-CUP"
git push origin main
```

If the repo/remote is not set yet:

```powershell
cd E:\GCB
git init
git add -A
git commit -m "GC-TGCUP code"
git branch -M main
git remote add origin https://github.com/MalihaIlyas80/gctgcup.git
git push -u origin main
```

(If `git push` is rejected because the remote already has the old `best.pt`/data,
that's fine — those paths are now gitignored. Just commit and push the code.)

---

## STEP B — Run on Kaggle (GPU)

1. Open your notebook on Kaggle.
2. Settings: **Accelerator = GPU T4 x2**, **Internet = ON**.
3. Add Data → your dataset that contains `cup2_dataset` (the one you already uploaded).
4. Upload/open `kaggle_train.ipynb` (it is in the repo after STEP A) and **Run All**.

The notebook does exactly this (you can also paste the cells manually):

```python
# 1) clone code
!rm -rf /kaggle/working/gctgcup
!git clone https://github.com/MalihaIlyas80/gctgcup.git /kaggle/working/gctgcup
%cd /kaggle/working/gctgcup

# 2) deps
!pip install -q transformers sacrebleu nltk scikit-learn pyyaml tqdm javalang

# 3) symlink the uploaded dataset (auto-detect path)
import os, glob
CUP = glob.glob('/kaggle/input/**/cup2_dataset', recursive=True)[0]
link = '/kaggle/working/gctgcup/cup2_dataset'
if os.path.lexists(link): os.remove(link)
os.symlink(CUP, link)
os.environ['HF_HOME'] = '/kaggle/temp/hf'; os.makedirs('/kaggle/temp/hf', exist_ok=True)

# 4) data prep
!python scripts/prepare_data.py --max-samples 20000 --vocab-max-size 30000

# 5) train (GraphCodeBERT fine-tuned)
!python scripts/train.py --config configs/kaggle_gpu.yaml

# 6) evaluate vs TG-CUP
!python scripts/evaluate.py --config configs/kaggle_gpu.yaml --checkpoint checkpoints/best.pt
```

---

## STEP C — Get results back

The last notebook cell copies `best.pt`, `evaluation_report.json`, and
`training_history.json` to `/kaggle/working/output`. Download them from the right
sidebar. Put `best.pt` back into `E:\GCB\checkpoints\` only if you want to evaluate
locally.

---

## Don't want to re-run repeatedly?

- Training **resumes from `checkpoints/best.pt`** automatically, so you can stop and
  continue across Kaggle sessions (keep checkpoints by committing the notebook output
  as a new dataset version, or re-upload `best.pt`).
- For a quick smoke test first (5–10 min), run prepare with `--max-samples 1000` and
  train with fewer epochs by editing `update_epochs` in `configs/kaggle_gpu.yaml`.
- For the final, TG-CUP-beating run, raise `--max-samples` to 40000+ and keep
  `update_epochs: 25`.

---

## Tuning knobs (in `configs/kaggle_gpu.yaml`)

| Knob | Effect |
|------|--------|
| `data.vocab_max_size` | Larger = covers more words, slower softmax. 30k is plenty. |
| `model.det_threshold` | Raise (0.55–0.6) for higher detection precision (more copies). |
| `training.update_epochs` | More epochs = better generation until early-stop. |
| `training.pos_weight` | Raise if detector misses outdated comments (low recall). |
| `data.max_samples` (via `--max-samples`) | More data = better generation; main lever to beat TG-CUP. |
