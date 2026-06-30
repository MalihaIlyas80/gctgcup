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
    det_threshold: float = 0.5,
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
    self.det_threshold = det_threshold
    self.pos_weight = pos_weight.to(device) if pos_weight is not None else None
    self.grad_accumulation_steps = grad_accumulation_steps
    os.makedirs(checkpoint_dir, exist_ok=True)
    self.best_val_loss = float("inf")
    self.best_val_score = float("-inf")
    self.epochs_no_improve = 0
    self.scheduler = None
    self.use_amp = device.type == "cuda"
    self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

  def _optimizer_step(self):
    self.scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    if self.scheduler is not None:
      self.scheduler.step()
    self.optimizer.zero_grad()

  def train_epoch(self, phase: str = "joint") -> float:
    self.model.train()
    total_loss = 0.0
    n = 0
    self.optimizer.zero_grad()
    for step, batch in enumerate(tqdm(self.train_loader, desc="Train", leave=False)):
      batch = self._to_device(batch)
      with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
        out = self.model(batch, pos_weight=self.pos_weight, phase=phase)
        loss = out["loss"] / self.grad_accumulation_steps
      self.scaler.scale(loss).backward()
      total_loss += out["loss"].item()
      n += 1
      if (step + 1) % self.grad_accumulation_steps == 0:
        self._optimizer_step()
    if n % self.grad_accumulation_steps != 0:
      self._optimizer_step()
    return total_loss / max(n, 1)

  def _val_score(self, metrics, phase: str) -> float:
    if phase == "detection":
      return metrics.det_f1
    return metrics.accuracy * 2.0 + metrics.bleu * 0.2 + metrics.sari * 0.2

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
      with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
        out = self.model(batch, pos_weight=self.pos_weight)
      total_loss += out["loss"].item()
      n += 1

      det_logits = out["det_logits"]
      preds = (torch.sigmoid(det_logits) >= self.det_threshold).long().cpu().tolist()
      labels = batch["labels"].long().cpu().tolist()
      det_preds.extend(preds)
      det_labels.extend(labels)

      src_tok_texts = [" ".join(t) for t in batch["src_tokens_list"]]
      ref_tok_texts = [" ".join(t) for t in batch["dst_tokens_list"]]

      gen_ids, no_upd_texts, beam_cands, surface_texts, beam_surfaces = self.model.generate(
        batch["src_ids"], batch["edit_ids"],
        batch["src_methods"], batch["dst_methods"],
        batch["graphs"],
        beam_size=1,
        det_threshold=self.det_threshold,
        comments=batch["src_descs"],
        src_descs=src_tok_texts,
        src_tokens_list=batch["src_tokens_list"],
        id2token=self.vocab.id2token if self.vocab else {},
        return_beam_candidates=True,
        force_update=True,
      )

      for ids, no_upd, cands, surf, surf_cands, ref, src in zip(
        gen_ids, no_upd_texts, beam_cands, surface_texts, beam_surfaces,
        ref_tok_texts, src_tok_texts,
      ):
        if no_upd is not None:
          pred_text = no_upd
          beam_texts = [no_upd] * 5
        else:
          pred_text = surf or " ".join(self.vocab.decode(ids))
          beam_texts = surf_cands if surf_cands else [" ".join(self.vocab.decode(c)) for c in cands]
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
      outdated=det_labels,
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

  def _reset_early_stop(self) -> None:
    self.epochs_no_improve = 0
    self.best_val_score = float("-inf")
    self.best_val_loss = float("inf")

  def fit(self, epochs: int, phase: str = "joint", resume: bool = True) -> Dict:
    if phase == "detection":
      self.model.set_training_phase("detection")
    elif phase == "update":
      self.model.set_training_phase("update")
    else:
      self.model.set_training_phase("joint")

    best_path = os.path.join(self.checkpoint_dir, "best.pt")
    if resume and phase == "joint" and os.path.exists(best_path):
      print("Resuming from best.pt ...")
      self.load_checkpoint("best.pt")

    steps_per_epoch = max(len(self.train_loader) // self.grad_accumulation_steps, 1)
    self.scheduler = OneCycleLR(
      self.optimizer,
      max_lr=[pg["lr"] for pg in self.optimizer.param_groups],
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      pct_start=0.2,
    )
    history = []
    for epoch in range(1, epochs + 1):
      t0 = time.time()
      train_loss = self.train_epoch(phase=phase)
      metrics = self.validate()
      val_loss = metrics.per_sample["val_loss"]
      elapsed = time.time() - t0
      val_score = self._val_score(metrics, phase)

      print(
        f"Epoch {epoch}/{epochs} [{phase}] | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
        f"Acc={metrics.accuracy:.2f}% | SARI={metrics.sari:.2f}% | BLEU={metrics.bleu:.2f}% | "
        f"Det-F1={metrics.det_f1:.2f}% | score={val_score:.2f} | time={elapsed:.1f}s"
      )
      history.append({"epoch": epoch, "phase": phase, "train_loss": train_loss, **metrics.to_dict()})

      if val_score > self.best_val_score:
        self.best_val_score = val_score
        self.best_val_loss = val_loss
        self.epochs_no_improve = 0
        self.save_checkpoint("best.pt")
        print(f"  new best (score={val_score:.2f}) -> saved best.pt")
      else:
        self.epochs_no_improve += 1
        if self.epochs_no_improve >= self.patience:
          print(f"Early stopping at epoch {epoch}")
          break

    return {
      "history": history,
      "best_val_loss": self.best_val_loss,
      "best_val_score": self.best_val_score,
      "phase": phase,
    }

  def fit_two_stage(self, detection_epochs: int, update_epochs: int) -> Dict:
    """Stage 1: train detection. Stage 2: train update decoder (detector frozen)."""
    print("\n" + "=" * 70)
    print(f" STAGE 1 ? Detection ({detection_epochs} epochs max)")
    print("=" * 70)
    self._reset_early_stop()
    det_result = self.fit(detection_epochs, phase="detection", resume=False)
    best_path = os.path.join(self.checkpoint_dir, "best.pt")
    if os.path.exists(best_path):
      print("Loading best detection checkpoint for stage 2 ...")
      self.load_checkpoint("best.pt")

    print("\n" + "=" * 70)
    print(f" STAGE 2 ? Update generation ({update_epochs} epochs max)")
    print("=" * 70)
    self._reset_early_stop()
    upd_result = self.fit(update_epochs, phase="update", resume=False)

    return {
      "detection": det_result,
      "update": upd_result,
      "best_val_score": upd_result["best_val_score"],
      "best_val_loss": upd_result["best_val_loss"],
      "history": det_result["history"] + upd_result["history"],
    }

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
