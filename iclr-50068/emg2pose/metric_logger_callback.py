# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import pytorch_lightning as pl


class EpochMetricsLogger(pl.Callback):
    """Persist key epoch-level metrics to a CSV file after every validation epoch."""

    def __init__(
        self,
        filename: str = "epoch_metrics.csv",
        metric_keys: List[str] | None = None,
    ) -> None:
        """
        Args:
            filename:    output CSV filename (written to the logger's log_dir).
            metric_keys: explicit list of metric names to log, read from
                         trainer.callback_metrics each epoch.  When None the
                         callback falls back to the hardcoded regression columns
                         (train_loss / train_mae / val_mae / …).
                         Pass this from YAML to get classifier-specific columns.
        """
        super().__init__()
        self.filename = filename
        self.metric_keys: List[str] | None = metric_keys
        self.filepath: Path | None = None
        self.fieldnames: List[str] = []
        self._lr_lookup: List[Tuple[str, int, int]] = []
        self._train_summary_buffer: Dict[str, float] | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self._configure_lr_lookup(trainer)

        if self.metric_keys is not None:
            # Dynamic columns: whatever the caller listed in the YAML
            self.fieldnames = (
                ["epoch"]
                + list(self.metric_keys)
                + ["grad_norm"]
                + [name for name, _, _ in self._lr_lookup]
            )
        else:
            # Regression defaults (backward-compatible)
            self.fieldnames = [
                "epoch",
                "train_loss",
                "train_mae",
                "train_vel",
                "train_consistency",
                "train_landmark_distance",
                "val_loss",
                "val_mae",
                "val_vel",
                "val_consistency",
                "val_landmark_distance",
                "grad_norm",
            ] + [name for name, _, _ in self._lr_lookup]

        self.filepath = self._resolve_log_path(trainer)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        if not self.filepath.exists():
            with self.filepath.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._train_summary_buffer = getattr(pl_module, "train_epoch_summary", None)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        val_summary = getattr(pl_module, "val_epoch_summary", None)
        train_summary = self._train_summary_buffer or getattr(
            pl_module, "train_epoch_summary", None
        )

        cb = trainer.callback_metrics

        def _f(v):
            """Convert tensor/any to plain Python float, or None."""
            if v is None:
                return None
            return v.item() if hasattr(v, "item") else float(v)

        def _get(summary, key, cb_key):
            """Read from module summary; fall back to callback_metrics."""
            if summary:
                v = summary.get(key)
                if v is not None:
                    return _f(v)
            return _f(cb.get(cb_key))

        if self.metric_keys is not None:
            # Dynamic mode: read every listed key directly from callback_metrics
            # Skip epoch if the primary val metric has not been logged yet
            first_val_key = next(
                (k for k in self.metric_keys if k.startswith("val_")), None)
            if first_val_key and cb.get(first_val_key) is None:
                return
            record = {"epoch": trainer.current_epoch}
            for key in self.metric_keys:
                record[key] = _f(cb.get(key))
            record["grad_norm"] = _f(cb.get("grad_norm_epoch", cb.get("grad_norm")))
            record.update(self._collect_lr_values(trainer))
        else:
            # Regression defaults (backward-compatible)
            val_loss = _get(val_summary, "loss", "val_loss")
            if val_loss is None:
                return
            record = {
                "epoch": trainer.current_epoch,
                "train_loss": _get(train_summary, "loss", "train_loss"),
                "train_mae": _get(train_summary, "mae", "train_mae"),
                "train_vel": _get(train_summary, "vel", "train_vel"),
                "train_consistency": _f(cb.get("train_consistency_epoch", cb.get("train_consistency"))),
                "train_landmark_distance": _get(train_summary, "landmark_distance", "train_landmark_distance"),
                "val_loss": val_loss,
                "val_mae": _get(val_summary, "mae", "val_mae"),
                "val_vel": _get(val_summary, "vel", "val_vel"),
                "val_consistency": _f(cb.get("val_consistency_epoch", cb.get("val_consistency"))),
                "val_landmark_distance": _get(val_summary, "landmark_distance", "val_landmark_distance"),
                "grad_norm": _f(cb.get("grad_norm_epoch", cb.get("grad_norm"))),
            }
            record.update(self._collect_lr_values(trainer))

        if self.filepath is None:
            self.filepath = self._resolve_log_path(trainer)
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

        with self.filepath.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(record)

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _configure_lr_lookup(self, trainer: pl.Trainer) -> None:
        self._lr_lookup.clear()
        optimizers = getattr(trainer, "optimizers", None) or []
        if not optimizers:
            return

        single_optimizer = len(optimizers) == 1 and len(optimizers[0].param_groups) == 1
        for opt_idx, optimizer in enumerate(optimizers):
            for group_idx, _ in enumerate(optimizer.param_groups):
                if single_optimizer:
                    name = "lr"
                else:
                    name = f"lr_opt{opt_idx}_group{group_idx}"
                self._lr_lookup.append((name, opt_idx, group_idx))

    def _collect_lr_values(self, trainer: pl.Trainer) -> Dict[str, float | None]:
        values: Dict[str, float | None] = {
            name: None for name, _, _ in self._lr_lookup
        }
        optimizers = getattr(trainer, "optimizers", None) or []
        if not optimizers:
            return values

        for name, opt_idx, group_idx in self._lr_lookup:
            optimizer = optimizers[opt_idx]
            lr = optimizer.param_groups[group_idx].get("lr")
            values[name] = lr
        return values

    def _resolve_log_path(self, trainer: pl.Trainer) -> Path:
        logger = trainer.logger
        if logger is None:
            return Path(trainer.default_root_dir) / self.filename

        if hasattr(logger, "log_dir"):
            return Path(logger.log_dir) / self.filename

        if hasattr(logger, "save_dir"):
            save_dir = Path(logger.save_dir)
            name = getattr(logger, "name", "lightning_logs")
            version = getattr(logger, "version", None)
            if version is not None:
                return save_dir / name / f"version_{version}" / self.filename
            return save_dir / name / self.filename

        return Path(trainer.default_root_dir) / self.filename
