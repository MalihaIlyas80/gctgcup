# GC-TGCUP: GraphCodeBERT-Enhanced Two-Stage Comment Updating

Thesis framework that **extends TG-CUP** to address identified research gaps and beat its performance.

## Gaps Addressed vs TG-CUP

| Gap | Solution in GC-TGCUP |
|-----|---------------------|
| No detection stage | **Stage 1**: Outdated comment detector (GraphCodeBERT + cross-attention) |
| Semantic weakness | **GraphCodeBERT** encodes old/new code semantics (mandatory) |
| Long comment full-rewrite | **Local Edit Decoder** with copy-from-old-comment mechanism |
| Structural information | **AST-Difference Graph + GGNN** (preserved from TG-CUP) |
| Poor NCIU handling | Explicit NCIU tagging + structural + semantic fusion |

## Architecture

```
Stage 1 (Detection):  old_code + new_code + old_comment → outdated? (yes/no)
Stage 2 (Update):     if outdated → Transformer decoder
                        ← old_comment + edit_sequence (Transformer encoder)
                        ← GraphCodeBERT code semantics
                        ← AST-Diff Graph (GGNN)
                      if long comment → Local Edit (pointer-generator)
```

## Quick Start (1000-sample test run)

Run these commands on your machine:

```bash
pip install -r requirements.txt

# Step 1 – clean & split data
python scripts/prepare_data.py --max-samples 1000

# Step 2 – train (GPU recommended)
python scripts/train.py --epochs 20

# Step 3 – evaluate vs TG-CUP paper numbers
python scripts/evaluate.py

# Step 4 – compare vs local TG-CUP baseline replica
python scripts/compare_baselines.py
```

Or all at once:

```bash
python run_pipeline.py
```

Quick smoke test (50 samples, 2 epochs):

```bash
python scripts/quick_test.py
```

## Evaluation Metrics (TG-CUP paper Section 4.2)

**Update stage:** Accuracy, Recall@5, GLEU, METEOR, SARI, BLEU  
**Detection stage:** Accuracy, Precision, Recall, F1  
**Subgroups:** NCIU accuracy, Long-comment accuracy

Results are compared against TG-CUP baseline (Table 4 of the paper).

## Project Structure

```
GCB/
├── configs/default.yaml       # hyperparameters
├── cup2_dataset/              # raw CUP2 dataset
├── data/processed/            # cleaned 1000-sample splits
├── src/
│   ├── data/                  # cleaning, AST-diff, dataset
│   ├── models/                # GraphCodeBERT, GGNN, detector, GC-TGCUP
│   ├── training/              # trainer
│   └── evaluation/            # metrics
├── scripts/                   # prepare_data, train, evaluate
└── checkpoints/               # saved models + reports
```

## Data Cleaning

Professional pipeline (`src/data/cleaning.py`):
- URL / email removal (TG-CUP Section 2.2)
- Unicode normalization, control-char stripping
- Duplicate removal, validity filtering
- NCIU heuristic tagging (TG-CUP RQ4)
- Stratified train/valid/test split

## Reference

Chen et al., "TG-CUP: A Transformer and GNN-Based Multi-Modal Comment Updating Method", ACM TOSEM 2025.
