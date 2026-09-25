# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np

from emg2pose import utils
from emg2pose.data import WindowedEmgDataset
from emg2pose.metrics import get_default_metrics
from emg2pose.pose_modules import BasePoseModule
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


log = logging.getLogger(__name__)


class SessionShuffleSampler(DistributedSampler):
    """Shuffle fixed-window blocks while keeping each batch disk-local.

    Sessions are shuffled each epoch, concatenated in temporal order, and
    cut into batch-sized blocks. Blocks are shuffled globally. By default,
    windows remain temporal inside each block because sample order within
    a gradient batch has no statistical effect, while sequential 50%
    overlapping reads halve cold-page traffic.
    """

    def __init__(
        self,
        session_groups: Sequence[np.ndarray],
        *,
        batch_size: int,
        block_batches: int = 1,
        shuffle_within_block: bool = False,
        num_replicas: int | None = None,
        rank: int | None = None,
        seed: int = 0,
    ) -> None:
        if num_replicas is None:
            num_replicas = (
                torch.distributed.get_world_size()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 1
            )
        if rank is None:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
        self.session_groups = [
            np.asarray(group, dtype=np.int64) for group in session_groups
        ]
        self.batch_size = int(batch_size)
        self.block_batches = int(block_batches)
        self.shuffle_within_block = bool(shuffle_within_block)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        total = sum(len(group) for group in self.session_groups)
        block = max(1, self.batch_size * self.block_batches)
        per_rank = math.ceil(total / self.num_replicas) if total else 0
        self.num_samples = math.ceil(per_rank / block) * block
        self.total_size = self.num_samples * self.num_replicas

    def _rank_stream(self, epoch: int) -> np.ndarray:
        if not self.session_groups:
            return np.empty(0, dtype=np.int64)
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(len(self.session_groups))
        stream = np.concatenate([self.session_groups[i] for i in order])
        if len(stream) < self.total_size:
            repeats = math.ceil(self.total_size / len(stream))
            stream = np.tile(stream, repeats)[: self.total_size]
        lo = self.rank * self.num_samples
        return stream[lo : lo + self.num_samples]

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        stream = self._rank_stream(self.epoch)
        block_size = max(1, self.batch_size * self.block_batches)
        blocks = [
            stream[i : i + block_size]
            for i in range(0, len(stream), block_size)
        ]
        rng.shuffle(blocks)
        if self.shuffle_within_block:
            for block in blocks:
                rng.shuffle(block)
        return iter(int(i) for block in blocks for i in block)

    def __len__(self) -> int:
        return self.num_samples


class WindowedEmgDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path], # keeping signature but overriding inside
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        metadata_csv: str = None, # ADDED metadata path
        data_location: str = None, # ADDED root dir
        val_test_window_length: int | None = None,
        skip_ik_failures: bool = False,
        stride: int | None = None,  
        max_windows_per_session: int | None = None,  
        num_train_sessions: int | None = None,  
        random_sampled_windows: bool = False,  
        windows_per_session: int | None = None,
        val_include_test: bool = False,
        val_test_stride: int | None = None,
        memmap_dir: str | None = None,
        apply_per_dataset_emg_norm: bool = True,
        emg_norm_mean: float = 0.0,
        emg_norm_std: float = 18.49,
        emg_channel_indices: list[int] | None = None,
        train_target_length: int | None = None,
        prefetch_factor: int = 2,
        train_window_index_cache: str | None = None,
        fixed_window_seed: int = 42,
        session_shuffle: bool = False,
        session_block_batches: int = 1,
        session_shuffle_within_block: bool = False,
        train_target_cache_dir: str | None = None,
        val_num_workers: int | None = None,
        delete_target_cache_after_run: bool = True,
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.val_test_window_length = val_test_window_length or window_length
        self.padding = padding
        self.stride = stride  
        self.max_windows_per_session = max_windows_per_session  
        self.num_train_sessions = num_train_sessions  
        self.random_sampled_windows = random_sampled_windows
        self.windows_per_session = windows_per_session
        self.metadata_csv = metadata_csv
        self.data_location = data_location

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transforms = None
        self.val_transforms = None
        self.test_transforms = None

        self.skip_ik_failures = skip_ik_failures
        self.val_include_test = val_include_test
        self.val_test_stride = val_test_stride
        self.memmap_dir = memmap_dir
        self.apply_per_dataset_emg_norm = apply_per_dataset_emg_norm
        self.emg_norm_mean = emg_norm_mean
        self.emg_norm_std = emg_norm_std
        self.emg_channel_indices = (
            None
            if emg_channel_indices is None
            else [int(i) for i in emg_channel_indices]
        )
        self.train_target_length = train_target_length
        self.prefetch_factor = int(prefetch_factor)
        self.train_window_index_cache = train_window_index_cache
        self.fixed_window_seed = int(fixed_window_seed)
        self.session_shuffle = bool(session_shuffle)
        self.session_block_batches = int(session_block_batches)
        self.session_shuffle_within_block = bool(session_shuffle_within_block)
        self.train_target_cache_dir = train_target_cache_dir
        self.delete_target_cache_after_run = bool(delete_target_cache_after_run)
        if val_num_workers is None:
            self.val_num_workers = min(4, int(num_workers)) if num_workers else 0
        else:
            self.val_num_workers = int(val_num_workers)
        self.target_cache_run_spec: dict | None = None

    def setup(self, stage: str | None = None) -> None:
        import pandas as pd
        import os

        if self.memmap_dir:
            from emg2pose.memmap_dataset import MemmapWindowedEmgDataset

            eval_stride = (
                self.val_test_stride
                if self.val_test_stride is not None
                else self.stride
            )
            val_splits = ["val", "test"] if self.val_include_test else ["val"]
            log.info(f"Memmap EMG2Pose corpus: {self.memmap_dir}")
            log.info(
                "Per-dataset EMG norm: "
                f"enabled={self.apply_per_dataset_emg_norm} "
                f"mean={self.emg_norm_mean} std={self.emg_norm_std}"
            )
            train_target_cache_dir = None
            if (
                self.train_target_cache_dir
                and self.train_target_length
                and self.train_window_index_cache
                and stage in (None, "fit")
            ):
                from emg2pose.window_target_cache import (
                    ensure_window_target_cache,
                    write_target_cache_run_log,
                )

                self.target_cache_run_spec = ensure_window_target_cache(
                    memmap_dir=self.memmap_dir,
                    window_index_cache=self.train_window_index_cache,
                    output_dir=self.train_target_cache_dir,
                    window_length=self.window_length,
                    stride=int(self.stride or self.window_length),
                    target_length=int(self.train_target_length),
                    seed=self.fixed_window_seed,
                    skip_ik_failures=self.skip_ik_failures,
                    log_fn=log.info,
                )
                write_target_cache_run_log(self.target_cache_run_spec)
                log.info(
                    "Train target cache spec: %s",
                    {
                        k: v
                        for k, v in self.target_cache_run_spec.items()
                        if k != "manifest"
                    },
                )
                train_target_cache_dir = self.train_target_cache_dir
            self.train_dataset = MemmapWindowedEmgDataset(
                self.memmap_dir,
                window_length=self.window_length,
                stride=self.stride,
                padding=self.padding,
                jitter=self.train_window_index_cache is None,
                transform=self.train_transforms,
                skip_ik_failures=self.skip_ik_failures,
                allowed_splits=["train"],
                apply_per_dataset_emg_norm=self.apply_per_dataset_emg_norm,
                emg_norm_mean=self.emg_norm_mean,
                emg_norm_std=self.emg_norm_std,
                emg_channel_indices=self.emg_channel_indices,
                target_length=self.train_target_length,
                window_index_cache=self.train_window_index_cache,
                fixed_window_seed=self.fixed_window_seed,
                target_cache_dir=train_target_cache_dir,
            )
            self.val_dataset = MemmapWindowedEmgDataset(
                self.memmap_dir,
                window_length=self.val_test_window_length,
                stride=eval_stride,
                padding=self.padding,
                jitter=False,
                transform=self.val_transforms,
                skip_ik_failures=self.skip_ik_failures,
                allowed_splits=val_splits,
                apply_per_dataset_emg_norm=self.apply_per_dataset_emg_norm,
                emg_norm_mean=self.emg_norm_mean,
                emg_norm_std=self.emg_norm_std,
                emg_channel_indices=self.emg_channel_indices,
            )
            self.test_dataset = MemmapWindowedEmgDataset(
                self.memmap_dir,
                window_length=self.val_test_window_length,
                stride=eval_stride,
                padding=(0, 0),
                jitter=False,
                transform=self.test_transforms,
                skip_ik_failures=self.skip_ik_failures,
                allowed_splits=["test"],
                apply_per_dataset_emg_norm=self.apply_per_dataset_emg_norm,
                emg_norm_mean=self.emg_norm_mean,
                emg_norm_std=self.emg_norm_std,
                emg_channel_indices=self.emg_channel_indices,
            )
            log.info(
                f"Memmap windows: train={len(self.train_dataset)} "
                f"val={len(self.val_dataset)} test={len(self.test_dataset)}"
            )
            return
        
        # 1. Load the exact intact-row split CSV
        if self.metadata_csv and os.path.exists(self.metadata_csv):
            df = pd.read_csv(self.metadata_csv)
            # 2. Filter out validation/test to ensure pure training fraction
            train_df = df[df['split'] == 'train']
            val_df = df[df['split'] == 'val']
            test_df = df[df['split'] == 'test']
            if self.val_include_test:
                val_df = df[df['split'].isin(['val', 'test'])]
                log.info(
                    "val_include_test=True: val loader uses split in {val, test} "
                    f"({len(val_df)} sessions; test loader stays test-only, "
                    f"{len(test_df)} sessions)"
                )
            
            # 3. Your dataset MUST ONLY load the files listed in train_df['filename']
            # We assume filename columns don't have .hdf5 extension or we add it
            def get_paths(files):
                return [Path(self.data_location) / f"{f}.hdf5" if not str(f).endswith('.hdf5') else Path(self.data_location) / f for f in files]
            
            actual_train_sessions = get_paths(train_df['filename'].tolist())
            actual_val_sessions = get_paths(val_df['filename'].tolist())
            actual_test_sessions = get_paths(test_df['filename'].tolist())
            log.info(f"Loaded explicit datasets from {self.metadata_csv}: {len(actual_train_sessions)} train")
        else:
            # Fallback
            actual_train_sessions = self.train_sessions
            actual_val_sessions = self.val_sessions
            actual_test_sessions = self.test_sessions

        train_datasets = []
        for hdf5_path in actual_train_sessions:
            dataset = WindowedEmgDataset(
                hdf5_path,
                transform=self.train_transforms,
                window_length=self.window_length,
                stride=self.stride,
                padding=self.padding,
                jitter=True,
                skip_ik_failures=self.skip_ik_failures,
                random_sampled_windows=self.random_sampled_windows,
                windows_per_session=self.windows_per_session,
            )
            if len(dataset) == 0:
                continue
            train_datasets.append(dataset)
        self.train_dataset = ConcatDataset(train_datasets)
        
        eval_stride = (
            self.val_test_stride
            if self.val_test_stride is not None
            else self.stride
        )
        val_datasets = []
        for hdf5_path in actual_val_sessions:
            dataset = WindowedEmgDataset(
                hdf5_path,
                transform=self.val_transforms,
                window_length=self.val_test_window_length,
                stride=eval_stride,
                padding=self.padding,
                jitter=False,
                skip_ik_failures=self.skip_ik_failures,
            )
            if len(dataset) == 0:
                continue
            val_datasets.append(dataset)
        self.val_dataset = ConcatDataset(val_datasets)
        
        test_datasets = []
        for hdf5_path in actual_test_sessions:
            dataset = WindowedEmgDataset(
                hdf5_path,
                transform=self.test_transforms,
                window_length=self.val_test_window_length,
                stride=eval_stride,
                padding=(0, 0),
                jitter=False,
                skip_ik_failures=self.skip_ik_failures,
            )
            if len(dataset) == 0:
                continue
            test_datasets.append(dataset)
        self.test_dataset = ConcatDataset(test_datasets)

    def _loader_kwargs(
        self,
        shuffle: bool,
        persistent: bool,
        num_workers: int | None = None,
    ) -> dict:
        workers = self.num_workers if num_workers is None else int(num_workers)
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": workers,
            "pin_memory": True,
            "shuffle": shuffle,
        }
        if workers > 0:
            kwargs["persistent_workers"] = persistent
            kwargs["prefetch_factor"] = self.prefetch_factor
            kwargs["multiprocessing_context"] = "forkserver"
            kwargs["timeout"] = 300.0
            if self.memmap_dir:
                from emg2pose.memmap_dataset import memmap_worker_init_fn

                kwargs["worker_init_fn"] = memmap_worker_init_fn
        return kwargs

    def train_dataloader(self) -> DataLoader:
        kwargs = self._loader_kwargs(shuffle=True, persistent=True)
        if self.session_shuffle:
            sampler = SessionShuffleSampler(
                self.train_dataset.session_index_groups(),
                batch_size=self.batch_size,
                block_batches=self.session_block_batches,
                shuffle_within_block=self.session_shuffle_within_block,
                seed=int(os.environ.get("PL_GLOBAL_SEED", "0") or 0),
            )
            kwargs["sampler"] = sampler
            kwargs["shuffle"] = False
        return DataLoader(self.train_dataset, **kwargs)

    def val_dataloader(self) -> DataLoader:
        # Persistent + few workers: val used to fork 16 processes every epoch
        # and leave them alive into the next train epoch (~48 GB extra PSS).
        workers = self.val_num_workers
        return DataLoader(
            self.val_dataset,
            **self._loader_kwargs(
                shuffle=False,
                persistent=workers > 0,
                num_workers=workers,
            ),
        )

    def test_dataloader(self) -> DataLoader:
        workers = self.val_num_workers
        return DataLoader(
            self.test_dataset,
            **self._loader_kwargs(
                shuffle=False,
                persistent=False,
                num_workers=workers,
            ),
        )


class Emg2PoseModule(pl.LightningModule):
    def __init__(
        self,
        network_conf: DictConfig,
        optimizer_conf: DictConfig,
        lr_scheduler_conf: DictConfig,
        provide_initial_pos: bool = False,
        loss_weights: dict[str, float] | None = None,
        pose_mean_path: str = "pose_mean_train_split.npy",
        pose_std_path: str = "pose_std_train_split.npy",
        emg_dropout: float = 0.0,
        pre_tds_mask_prob: float = 0.0,
    ) -> None:

        super().__init__()
        self.save_hyperparameters()
        self.model: BasePoseModule = instantiate(network_conf, _convert_="all")
        self.provide_initial_pos = provide_initial_pos
        self.loss_weights = loss_weights or {"mae": 1}
        self.emg_dropout = emg_dropout
        self.pre_tds_mask_prob = pre_tds_mask_prob

        self.metrics_list = get_default_metrics()

        # Epoch-level metric accumulation for logging
        self._epoch_metric_storage: dict[str, dict[str, float]] = {}
        self._epoch_counts: dict[str, int] = {}
        self.train_epoch_summary: dict[str, float] | None = None
        self.val_epoch_summary: dict[str, float] | None = None

        try:
            self.pose_mean = torch.from_numpy(np.load(pose_mean_path)).float()
            self.pose_std = torch.from_numpy(np.load(pose_std_path)).float()
        except FileNotFoundError:
            log.warning(f"Normalization files not found. Using default values.")
            self.pose_mean = torch.zeros(20)
            self.pose_std = torch.ones(20)

    def forward(
        self, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        return self.model.forward(batch, self.provide_initial_pos)

    def _consume_network_aux_losses(self) -> dict[str, torch.Tensor]:
        network = getattr(self.model, "network", None)
        if network is None or not hasattr(network, "consume_aux_losses"):
            return {}
        aux_losses = network.consume_aux_losses()
        if not isinstance(aux_losses, dict):
            return {}
        return aux_losses

    def _step(
        self, batch: Mapping[str, torch.Tensor], stage: str = "train"
    ) -> torch.Tensor:
        batch["no_ik_failure"] = self.update_ik_failure_mask(batch["no_ik_failure"])
        
        # Check if we're using MAE
        from emg2pose.pose_modules import MAEPoseModule
        is_mae = isinstance(self.model, MAEPoseModule)
        
        if is_mae:
            # MAE training: self-supervised reconstruction
            preds, targets, no_ik_failure, mae_loss = self.forward(batch)
            
            # MAE loss is already computed in the network
            loss = mae_loss
            
            # Log MAE-specific metrics
            self.log(f"{stage}_mae_reconstruction_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
            
            # Accumulate for epoch summary
            if stage in {"train", "val"}:
                batch_size = batch["emg"].shape[0]
                metrics = {f"{stage}_mae_reconstruction_loss": loss.detach()}
                self._accumulate_epoch_metrics(stage, metrics, loss.detach(), batch_size)
            
            return loss
        
        # Standard pose prediction training
        # Apply dropout-like masking to EMG input during training for robustness
        if self.emg_dropout > 0 and stage == "train" and self.training:
            emg = batch["emg"]
            # Random binary mask: keep with probability (1 - emg_dropout)
            mask = torch.rand_like(emg) > self.emg_dropout
            batch["emg"] = emg * mask.float()
            
            # Log dropout statistics (only on first batch to avoid spam)
            if not hasattr(self, '_dropout_logged'):
                zeros_pct = (mask == 0).float().mean().item() * 100
                log.info(f"✓ EMG Dropout Active: {self.emg_dropout*100:.0f}% rate → {zeros_pct:.1f}% values zeroed (shape: {emg.shape})")
                self._dropout_logged = True

        # Optional raw-window masking before TDS stack (TDSTransformer only)
        if stage == "train" and self.training and self.pre_tds_mask_prob > 0:
            from emg2pose.networks import TDSTransformer
            network = getattr(self.model, "network", None)
            if isinstance(network, TDSTransformer):
                emg = batch["emg"]  # (B, C, T)
                B, C, T = emg.shape
                
                # Create mask for each batch sample
                mask = torch.ones_like(emg)  # Start with all ones (keep everything)
                
                for b in range(B):
                    # Determine how many chunks to drop based on probability
                    # Average chunk size is (50+250)/2 = 150
                    avg_chunk_size = 150
                    max_possible_chunks = T // avg_chunk_size
                    
                    for _ in range(max_possible_chunks):
                        if torch.rand(1).item() < self.pre_tds_mask_prob:
                            # Drop this chunk
                            chunk_size = torch.randint(50, 251, (1,)).item()
                            start_pos = torch.randint(0, max(1, T - chunk_size + 1), (1,)).item()
                            end_pos = min(start_pos + chunk_size, T)
                            
                            # Zero out the chunk for all channels
                            mask[b, :, start_pos:end_pos] = 0
                
                batch["emg"] = emg * mask

                if not hasattr(self, "_pre_tds_mask_logged"):
                    masked_pct = (mask == 0).float().mean().item() * 100
                    log.info(
                        f"✓ Pre-TDS Chunk Masking Active: {self.pre_tds_mask_prob*100:.0f}% chunk prob → {masked_pct:.1f}% raw samples zeroed (chunks: 50-250 samples)"
                    )
                    self._pre_tds_mask_logged = True
        
        preds, targets, no_ik_failure, temporal_mask = self.forward(batch)
        
        # Special handling for STFT/CST/ViT Transformers: align shapes
        # These models output (B, num_windows, joints)
        # But targets are (B, joints, time_samples)
        try:
            from emg2pose.stft_transformer_arch import STFTTransformer
            from emg2pose.circular_stft_transformer_arch import CyclicSpectralTransformer
            from emg2pose.stft_vit_arch import STFTViT
        except ImportError:
            STFTTransformer = CyclicSpectralTransformer = STFTViT = type(None)
        network = getattr(self.model, "network", None)
        if isinstance(network, (STFTTransformer, CyclicSpectralTransformer, STFTViT)):
            # Debug: Log shapes BEFORE alignment
            if not hasattr(self, '_stft_debug_logged'):
                log.info(f"DEBUG: Before alignment - preds: {preds.shape}, targets: {targets.shape}, no_ik_failure: {no_ik_failure.shape}")
                self._stft_debug_logged = True
            
            # Transpose predictions: (B, T_windows, J) -> (B, J, T_windows)
            preds = preds.transpose(1, 2)
            
            # Compute window centers for alignment
            if isinstance(network, STFTTransformer):
                hop = network.stft.config.stft_hop_length
                window = network.stft.config.stft_window_length
            elif isinstance(network, STFTViT):
                hop = network.stft.config.stft_hop_length
                window = network.stft.config.stft_window_length
            else:  # CyclicSpectralTransformer
                hop = network.config.filter_stride
                window = network.config.filter_length
            
            num_windows = preds.shape[2]  # T_windows dimension after transpose
            
            # Calculate centers, clipped to valid range
            centers = [min(i * hop + window // 2, targets.shape[2] - 1) for i in range(num_windows)]
            
            # Downsample targets to match STFT windows
            targets = targets[:, :, centers]  # (B, J, T_orig) -> (B, J, T_stft)
            
            # Downsample mask - it should be (B, T) format, not (B, J, T)
            # metrics.py will expand it from (B, T_stft) to (B, J, T_stft)
            if no_ik_failure.dim() == 3:
                # If somehow it's 3D (B, J, T), take mean across joints to get (B, T)
                # then downsample to (B, T_stft)
                no_ik_failure = no_ik_failure.all(dim=1)  # (B, J, T) -> (B, T)
                no_ik_failure = no_ik_failure[:, centers]  # (B, T_orig) -> (B, T_stft)
            elif no_ik_failure.dim() == 2:
                # 2D mask: (B, T) - downsample temporal dimension
                no_ik_failure = no_ik_failure[:, centers]  # (B, T_orig) -> (B, T_stft)
            elif no_ik_failure.dim() == 1:
                # 1D mask: (B,) - broadcast to (B, T_stft)
                no_ik_failure = no_ik_failure.unsqueeze(1).expand(-1, num_windows)  # (B,) -> (B, T_stft)
            else:
                raise ValueError(f"Unexpected no_ik_failure dimensions: {no_ik_failure.shape}")
            
            if not hasattr(self, '_stft_alignment_logged'):
                # Detect model type for logging
                if isinstance(network, STFTTransformer):
                    model_name = "STFT"
                elif isinstance(network, STFTViT):
                    model_name = "STFT-ViT"
                else:  # CyclicSpectralTransformer
                    model_name = "CST"
                
                log.info(
                    f"✓ {model_name} Alignment: predictions ({preds.shape}) aligned with targets ({targets.shape}) "
                    f"via {num_windows} window centers (hop={hop}, window={window})"
                )
                self._stft_alignment_logged = True
        
        # Combine IK failure mask with temporal dropout mask
        # Both masks should exclude positions from loss calculation
        combined_mask = no_ik_failure
        if temporal_mask is not None and stage == "train":
            # temporal_mask: 1=kept, 0=dropped
            # combined_mask: positions where BOTH are valid
            
            # Also align temporal_mask if using STFT/CST
            if isinstance(network, (STFTTransformer, CyclicSpectralTransformer)):
                if temporal_mask.dim() == 3:
                    # (B, J, T) -> take all across joints -> (B, T) -> downsample
                    temporal_mask = temporal_mask.all(dim=1)[:, centers]
                elif temporal_mask.dim() == 2:
                    # (B, T) -> downsample
                    temporal_mask = temporal_mask[:, centers]
                elif temporal_mask.dim() == 1:
                    # (B,) -> expand
                    temporal_mask = temporal_mask.unsqueeze(1).expand(-1, num_windows)
            
            combined_mask = no_ik_failure & temporal_mask
            
            if not hasattr(self, '_temporal_dropout_logged'):
                temporal_dropped_pct = (temporal_mask == 0).float().mean().item() * 100
                log.info(f"✓ Temporal Chunk Dropout Active: {temporal_dropped_pct:.1f}% frames excluded from loss")
                self._temporal_dropout_logged = True

        # skip_ik_failures=False can yield a window of all-zero IK labels.
        # Empty-mask L1 is NaN; one such val batch poisons epoch val_mae and
        # EarlyStopping kills the run. Skip the batch instead.
        if not combined_mask.any():
            loss = preds.reshape(-1)[0] * 0.0
            self.log(
                f"{stage}_empty_ik_batch",
                1.0,
                on_step=False,
                on_epoch=True,
                reduce_fx="sum",
                sync_dist=True,
            )
            return loss

        metrics = {}
        for metric in self._active_metrics(stage):
            metrics.update(metric(preds, targets, combined_mask, stage))

        aux_losses = self._consume_network_aux_losses()
        if aux_losses:
            aux_metrics = {f"{stage}_{name}": value for name, value in aux_losses.items()}
            metrics.update(aux_metrics)

        # Target-dependent aux losses (hook used by NeuroKine for diffusion).
        _net = getattr(self.model, "network", None)
        if _net is not None and hasattr(_net, "target_aux_losses"):
            _tgt_aux = _net.target_aux_losses(preds, targets, combined_mask, stage)
            if isinstance(_tgt_aux, dict):
                metrics.update(
                    {f"{stage}_{k}": v for k, v in _tgt_aux.items()}
                )

        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True)

        loss = preds.new_tensor(0.0)
        for loss_name, weight in self.loss_weights.items():
            metric_key = f"{stage}_{loss_name}"
            metric_value = metrics.get(metric_key)
            if metric_value is None:
                continue
            loss = loss + metric_value * weight
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        if stage in {"train", "val"}:
            batch_size = batch["emg"].shape[0]
            self._accumulate_epoch_metrics(stage, metrics, loss.detach(), batch_size)

        return loss

    def _active_metrics(self, stage: str):
        """Skip logging-only FK/smoothness metrics that are not in the loss.

        get_default_metrics() registers LandmarkDistances plus NeuroKine/
        KINE extras. Those still run every step even when loss_weights
        ignore them, and they dominate the autograd graph at 2 kHz.
        """
        if stage == "test":
            return self.metrics_list
        needed = {"mae"}
        for name, weight in self.loss_weights.items():
            if weight:
                needed.add(name)
        want_fk = bool(
            needed & {"fingertip_distance", "landmark_distance", "procrustes_landmark_distance"}
        )
        active = []
        for metric in self.metrics_list:
            name = type(metric).__name__
            if name == "AngleMAE":
                active.append(metric)
            elif name == "LandmarkDistances" and want_fk:
                active.append(metric)
            elif name == "ProcrustesLandmarkDistance" and "procrustes_landmark_distance" in needed:
                active.append(metric)
        if not active:
            return self.metrics_list
        return active

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step(batch, stage="val")

    def on_fit_start(self) -> None:
        if getattr(self, "compile_network", False) and not getattr(self, "_network_compiled", False):
            compile_threads = int(os.environ.get("TORCHINDUCTOR_COMPILE_THREADS", "8"))
            os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(compile_threads)
            try:
                import torch._inductor.config as inductor_config

                inductor_config.compile_threads = compile_threads
            except Exception:
                pass
            log.info(
                "torch.compile on training pose network (mode=%s, compile_threads=%s); eval remains eager",
                self.compile_network,
                compile_threads,
            )
            mode = self.compile_network if isinstance(self.compile_network, str) else "default"
            object.__setattr__(
                self.model,
                "_compiled_train_network",
                torch.compile(self.model.network, mode=mode, dynamic=False),
            )
            self._network_compiled = True

    def on_before_optimizer_step(self, _optimizer) -> None:
        """Log total gradient norm (L2) before every optimizer step."""
        from torch.nn.utils import clip_grad_norm_
        grad_norm = clip_grad_norm_(self.parameters(), max_norm=float("inf"))
        # sync_dist=False: avoids DDP all_reduce on a CPU tensor (crash).
        # Grad norm is a stability monitor — rank-0 value is sufficient.
        self.log(
            "grad_norm",
            float(grad_norm),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=False,
        )

    def on_train_epoch_start(self) -> None:
        self._reset_epoch_metrics("train")
        # Reset dropout logging flags for each epoch to show stats once per epoch
        if hasattr(self, '_dropout_logged'):
            delattr(self, '_dropout_logged')
        if hasattr(self, '_temporal_dropout_logged'):
            delattr(self, '_temporal_dropout_logged')
        if hasattr(self, '_pre_tds_mask_logged'):
            delattr(self, '_pre_tds_mask_logged')

    def on_validation_epoch_start(self) -> None:
        self._reset_epoch_metrics("val")

    def on_train_epoch_end(self) -> None:
        summary = self._finalize_epoch_metrics("train")
        self.train_epoch_summary = summary if summary else None

    def on_validation_epoch_end(self) -> None:
        summary = self._finalize_epoch_metrics("val")
        self.val_epoch_summary = summary if summary else None

    ### CHANGE: Added on_test_epoch_start to initialize accumulators.
    def on_test_start(self) -> None:
        """Initialize accumulators ONCE for the entire test session
        (across all dataloaders, not per-epoch — PL fires
        on_test_epoch_start/end per dataloader, which would clear
        between conditions)."""
        self.test_errors = []
        self.test_errors_idx = []

    def on_test_epoch_start(self) -> None:
        # Intentionally a no-op now; accumulators live for the full session.
        if not hasattr(self, "test_errors"):
            self.test_errors = []
            self.test_errors_idx = []

    ### CHANGE: test_step now calculates errors for the batch and stores them.
    def test_step(
        self, batch, batch_idx, dataloader_idx: int | None = None
    ) -> torch.Tensor:
        batch["no_ik_failure"] = self.update_ik_failure_mask(batch["no_ik_failure"])
        preds, targets, no_ik_failure, _ = self.forward(batch)  # Unpack 4 values (ignore temporal_mask/mae_loss)

        # Special handling for STFT/CST/ViT Transformer: align shapes (same as in _step)
        try:
            from emg2pose.stft_transformer_arch import STFTTransformer
            from emg2pose.circular_stft_transformer_arch import CyclicSpectralTransformer
            from emg2pose.stft_vit_arch import STFTViT
        except ImportError:
            STFTTransformer = CyclicSpectralTransformer = STFTViT = type(None)
        network = getattr(self.model, "network", None)
        if isinstance(network, (STFTTransformer, CyclicSpectralTransformer, STFTViT)):
            # Transpose predictions: (B, T_windows, J) -> (B, J, T_windows)
            preds = preds.transpose(1, 2)
            
            # Compute window centers for alignment
            if isinstance(network, STFTTransformer):
                hop = network.stft.config.stft_hop_length
                window = network.stft.config.stft_window_length
            elif isinstance(network, STFTViT):
                hop = network.stft.config.stft_hop_length
                window = network.stft.config.stft_window_length
            else:  # CyclicSpectralTransformer
                hop = network.config.filter_stride
                window = network.config.filter_length
            
            num_windows = preds.shape[2]  # T_windows dimension after transpose
            
            # Calculate centers, clipped to valid range
            centers = [min(i * hop + window // 2, targets.shape[2] - 1) for i in range(num_windows)]
            
            # Downsample targets to match STFT windows
            targets = targets[:, :, centers]  # (B, J, T_orig) -> (B, J, T_stft)
            
            # Downsample mask - keep 2D format
            if no_ik_failure.dim() == 3:
                no_ik_failure = no_ik_failure.all(dim=1)[:, centers]  # (B, J, T) -> (B, T_stft)
            elif no_ik_failure.dim() == 2:
                no_ik_failure = no_ik_failure[:, centers]  # (B, T) -> (B, T_stft)
            elif no_ik_failure.dim() == 1:
                no_ik_failure = no_ik_failure.unsqueeze(1).expand(-1, num_windows)  # (B,) -> (B, T_stft)

        # ---- Start of in-step processing ----
        valid_mask = no_ik_failure.any(dim=1)
        if valid_mask.sum() > 0:
            # Move stats to the correct device for this batch
            self.pose_mean = self.pose_mean.to(preds.device)
            self.pose_std = self.pose_std.to(preds.device)
            
            # De-normalize and convert to degrees
            # Shape after filtering: (num_valid, joints, time)
            # pose_std/mean: (joints,) -> need (1, joints, 1) for broadcasting
            preds_denorm = preds[valid_mask] * self.pose_std.view(1, -1, 1) + self.pose_mean.view(1, -1, 1)
            targets_denorm = targets[valid_mask] * self.pose_std.view(1, -1, 1) + self.pose_mean.view(1, -1, 1)
            preds_deg = torch.rad2deg(preds_denorm)
            targets_deg = torch.rad2deg(targets_denorm)
            
            # Calculate absolute error, average over time (dim=2), then over joints (dim=1), and move to CPU
            batch_errors = torch.abs(preds_deg - targets_deg).mean(dim=2).cpu()  # Average over time -> (num_valid, joints)
            self.test_errors.append(batch_errors)
            self.test_errors_idx.append(
                torch.full((batch_errors.shape[0],),
                           int(dataloader_idx) if dataloader_idx is not None else 0,
                           dtype=torch.long)
            )
        # ---- End of in-step processing ----

        # The rest of the test_step is for standard Lightning logging
        metrics = {}
        for metric in self.metrics_list:
            metrics.update(metric(preds, targets, no_ik_failure, "test"))
        self.log_dict(metrics, sync_dist=True)

        loss = preds.new_tensor(0.0)
        for loss_name, weight in self.loss_weights.items():
            metric_value = metrics.get(f"test_{loss_name}")
            if metric_value is None:
                continue
            loss = loss + metric_value * weight
        self.log("test_loss", loss, sync_dist=True)

        return loss

    def on_test_epoch_end(self) -> None:
        # No-op: aggregation moved to on_test_end so accumulators survive
        # across multiple dataloaders.
        return

    ### CHANGE: aggregate at on_test_end so per-dataloader epoch boundaries
    ### don't wipe the accumulators between conditions.
    def on_test_end(self) -> None:
        # Concatenate errors + dataloader idxs collected on this process
        local_errors = torch.cat(self.test_errors) if self.test_errors else None
        local_idxs = torch.cat(self.test_errors_idx) if self.test_errors_idx else None
        self.test_errors.clear()
        self.test_errors_idx.clear()

        # Gather from all DDP ranks so rank 0 sees the full dataset
        if self.trainer.world_size > 1:
            import torch.distributed as dist
            all_local = [None] * self.trainer.world_size
            all_local_idx = [None] * self.trainer.world_size
            dist.all_gather_object(all_local, local_errors)
            dist.all_gather_object(all_local_idx, local_idxs)
            if not self.trainer.is_global_zero:
                return
            valid = [(e, i) for e, i in zip(all_local, all_local_idx) if e is not None]
            if not valid:
                log.warning("No valid samples found to calculate per-joint MAE.")
                return
            all_errors = torch.cat([e for e, _ in valid])
            all_idxs = torch.cat([i for _, i in valid])
        else:
            if local_errors is None:
                log.warning("No valid samples found to calculate per-joint MAE.")
                return
            all_errors = local_errors
            all_idxs = local_idxs

        joint_names = [
            "wrist_flex", "wrist_dev", "pro_sup", "thumb_flex", "thumb_abd",
            "thumb_mcp_flex", "thumb_mcp_abd", "index_flex", "index_abd",
            "middle_flex", "middle_abd", "ring_flex", "ring_abd", "pinky_flex",
            "pinky_abd", "thumb_pip", "index_pip", "middle_pip", "ring_pip",
            "pinky_pip",
        ]

        # Optional labels per dataloader (set by the eval framework before trainer.test):
        #   self.dataloader_labels = [{"generalization": "user"}, {"generalization": "stage"}, ...]
        labels = getattr(self, "dataloader_labels", None)

        results_list = []
        unique_idxs = sorted(set(all_idxs.tolist()))
        for di in unique_idxs:
            mask = all_idxs == di
            errs = all_errors[mask]
            mean_per_joint = errs.mean(dim=0)
            std_per_joint = errs.std(dim=0)
            label = labels[di] if labels is not None and di < len(labels) else {}
            for j, name in enumerate(joint_names):
                row = {
                    "dataloader_idx": int(di),
                    **label,
                    "joint_name": name,
                    "mean_mae_deg": mean_per_joint[j].item(),
                    "std_mae_deg": std_per_joint[j].item(),
                    "n_samples": int(mask.sum().item()),
                }
                results_list.append(row)

        save_dir = Path(self.trainer.logger.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "per_joint_mae_analysis.csv"

        df = pd.DataFrame(results_list)
        df.to_csv(output_path, index=False)
        log.info(f"Saved detailed per-joint MAE analysis to {output_path}")

    def configure_optimizers(self):
        optimizer_conf = self.hparams.get("optimizer_conf")
        if optimizer_conf is None:
            raise ValueError("optimizer_conf is required")
        optimizer = instantiate(optimizer_conf, self.parameters())

        scheduler_conf = self.hparams.get("lr_scheduler_conf")
        if scheduler_conf is None:
            return optimizer

        scheduler_config = dict(scheduler_conf.scheduler)
        # ``verbose`` was removed from PyTorch scheduler constructors in 2.6.
        scheduler_config.pop("verbose", None)
        scheduler = instantiate(scheduler_config, optimizer)
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": scheduler_conf.get("interval", "epoch"),
            "frequency": scheduler_conf.get("frequency", 1),
        }
        monitor = scheduler_conf.get("monitor")
        if monitor is not None:
            scheduler_dict["monitor"] = monitor

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_dict,
        }

    def update_ik_failure_mask(self, no_ik_failure: torch.Tensor) -> torch.Tensor:
        mask = no_ik_failure.clone()

        if self.provide_initial_pos:
            mask[~mask[:, self.model.left_context]] = False

        if mask.sum() == 0:
            log.warning("All samples masked out due to missing initial state!")

        return mask

    def _reset_epoch_metrics(self, stage: str) -> None:
        self._epoch_metric_storage[stage] = {
            "mae_sum": 0.0,
            "vel_sum": 0.0,
            "landmark_distance_sum": 0.0,
            "loss_sum": 0.0,
        }
        self._epoch_counts[stage] = 0

    def _accumulate_epoch_metrics(
        self,
        stage: str,
        metrics: dict[str, torch.Tensor],
        loss: torch.Tensor,
        batch_size: int,
    ) -> None:
        storage = self._epoch_metric_storage.get(stage)
        if storage is None or batch_size == 0:
            return

        def _to_float(value: torch.Tensor | float) -> float:
            if isinstance(value, torch.Tensor):
                return value.detach().float().item()
            return float(value)

        mae_key = f"{stage}_mae"
        vel_key = f"{stage}_vel"
        landmark_key = f"{stage}_landmark_distance"

        if mae_key in metrics:
            storage["mae_sum"] += _to_float(metrics[mae_key]) * batch_size
        if vel_key in metrics:
            storage["vel_sum"] += _to_float(metrics[vel_key]) * batch_size
        if landmark_key in metrics:
            storage["landmark_distance_sum"] += _to_float(metrics[landmark_key]) * batch_size
        storage["loss_sum"] += _to_float(loss) * batch_size
        self._epoch_counts[stage] += batch_size

    def _finalize_epoch_metrics(self, stage: str) -> dict[str, float]:
        storage = self._epoch_metric_storage.get(stage)
        count = self._epoch_counts.get(stage, 0)
        if not storage or count == 0:
            return {}

        return {
            "mae": storage["mae_sum"] / count,
            "vel": storage["vel_sum"] / count,
            "landmark_distance": storage["landmark_distance_sum"] / count,
            "loss": storage["loss_sum"] / count,
        }