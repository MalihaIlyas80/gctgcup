"""
Training loop for GC-TGCUP two-stage framework.
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import collate_fn
from src.evaluation.metrics import compute_all_metrics


class Trainer:
  def __init__(
    self,
    model,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    checkpoint_dir: str = "checkpoints",
    patience: int = 5,
    vocab=None,
    max_valid_batches: int = 10,
    pos_weight: Optional[torch.Tensor] = None,
    grad_accumulation_steps: int = 4,
  ):
    self.model = model.to(device)
    self.train_loader = train_loader
    self.valid_loader = valid_loader
    self.optimizer = optimizer
    self.device = device
    self.checkpoint_dir = checkpoint_dir
    self.patience = patience
    self.vocab = vocab
    self.max_valid_batches = max_valid_batches
    # move pos_weight to device
    self.pos_weight = pos_weight.to(device) if pos_weight is not None else None
    self.grad_accumulation_steps = grad_accumulation_steps
    os.makedirs(checkpoint_dir, exist_ok=True)
    self.best_val_loss = float("inf")
    self.epochs_no_improve = 0
    self.scheduler = None

  def train_epoch(self) -> float:
    self.model.train()
    total_loss = 0.0
    n = 0
    self.optimizer.zero_grad()
    for step, batch in enumerate(tqdm(self.train_loader, desc="Train", leave=False)):
      batch = self._to_device(batch)
      out = self.model(batch, pos_weight=self.pos_weight)
      loss = out["loss"] / self.grad_accumulation_steps
      loss.backward()
      total_loss += out["loss"].item()
      n += 1
      if (step + 1) % self.grad_accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        if self.scheduler is not None:
          self.scheduler.step()
        self.optimizer.zero_grad()
    # final leftover
    if n % self.grad_accumulation_steps != 0:
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
      self.optimizer.step()
      if self.scheduler is not None:
        self.scheduler.step()
      self.optimizer.zero_grad()
    return total_loss / max(n, 1)

  @torch.no_grad()
  def validate(self) -> Dict:
    self.model.eval()
    total_loss = 0.0
    n = 0
    det_preds, det_labels = [], []
    predictions, references, sources = [], [], []
    beam_candidates_all = []
    is_nciu_list, is_long_list = [], []

    for bi, batch in enumerate(tqdm(self.valid_loader, desc="Valid", leave=False)):
      if bi >= self.max_valid_batches:
        break
      batch = self._to_device(batch)
      out = self.model(batch, pos_weight=self.pos_weight)
      total_loss += out["loss"].item()
      n += 1

      det_logits = out["det_logits"]
      preds = (torch.sigmoid(det_logits) >= 0.45).long().cpu().tolist()
      labels = batch["labels"].long().cpu().tolist()
      det_preds.extend(preds)
      det_labels.extend(labels)

      gen_ids, no_upd_texts, beam_cands = self.model.generate(
        batch["src_ids"], batch["edit_ids"],
        batch["src_methods"], batch["dst_methods"],
        batch["graphs"],
        comments=batch["src_descs"],
        src_descs=batch["src_descs"],
        return_beam_candidates=True,
      )

      for ids, no_upd, cands, ref, src in zip(
        gen_ids, no_upd_texts, beam_cands, batch["dst_descs"], batch["src_descs"]
      ):
        if no_upd is not None:
          # No update predicted: return original comment directly (no tokenization loss)
          pred_text = no_upd
          beam_texts = [no_upd] * 5
        else:
          pred_text = " ".join(self.vocab.decode(ids)) if self.vocab else " ".join(str(t) for t in ids)
          beam_texts = [
            " ".join(self.vocab.decode(c)) if self.vocab else " ".join(str(t) for t in c)
            for c in cands
          ]
        predictions.append(pred_text)
        references.append(ref)
        sources.append(src)
        beam_candidates_all.append(beam_texts)

      is_nciu_list.extend(batch["is_nciu"].cpu().tolist())
      is_long_list.extend(batch["is_long"].cpu().tolist())

    metrics = compute_all_metrics(
      predictions, references, sources,
      beam_candidates=beam_candidates_all,
      det_preds=det_preds, det_labels=det_labels,
      is_nciu=is_nciu_list, is_long=is_long_list,
    )
    metrics.per_sample = {"val_loss": total_loss / max(n, 1)}
    return metrics

  def _to_device(self, batch: Dict) -> Dict:
    out = {}
    for k, v in batch.items():
      if isinstance(v, torch.Tensor):
        out[k] = v.to(self.device)
      else:
        out[k] = v
    return out

  def fit(self, epochs: int) -> Dict:
    # resume from best checkpoint if it exists
    best_path = os.path.join(self.checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
      print("Resuming from best.pt ...")
      self.load_checkpoint("best.pt")

    steps_per_epoch = max(len(self.train_loader) // self.grad_accumulation_steps, 1)
    self.scheduler = OneCycleLR(
      self.optimizer,
      max_lr=[pg["lr"] for pg in self.optimizer.param_groups],
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      pct_start=0.1,
    )
    history = []
    for epoch in range(1, epochs + 1):
      t0 = time.time()
      train_loss = self.train_epoch()
      metrics = self.validate()
      val_loss = metrics.per_sample["val_loss"]
      elapsed = time.time() - t0

      print(
        f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"Acc={metrics.accuracy:.2f}% | Det-F1={metrics.det_f1:.2f}% | "
        f"SARI={metrics.sari:.2f}% | time={elapsed:.1f}s"
      )
      history.append({"epoch": epoch, "train_loss": train_loss, **metrics.to_dict()})

      if val_loss < self.best_val_loss:
        self.best_val_loss = val_loss
        self.epochs_no_improve = 0
        self.save_checkpoint("best.pt")
      else:
        self.epochs_no_improve += 1
        if self.epochs_no_improve >= self.patience:
          print(f"Early stopping at epoch {epoch}")
          break

    return {"history": history, "best_val_loss": self.best_val_loss}

  def save_checkpoint(self, name: str) -> None:
    path = os.path.join(self.checkpoint_dir, name)
    torch.save({
      "model_state": self.model.state_dict(),
      "optimizer_state": self.optimizer.state_dict(),
    }, path)

  def load_checkpoint(self, name: str) -> None:
    path = os.path.join(self.checkpoint_dir, name)
    ckpt = torch.load(path, map_location=self.device)
    self.model.load_state_dict(ckpt["model_state"])
