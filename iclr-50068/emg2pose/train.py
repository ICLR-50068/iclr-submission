# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import logging
import os
import pprint
import torch
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
from emg2pose import transforms

from emg2pose.lightning import Emg2PoseModule
from emg2pose.transforms import Transform
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf


log = logging.getLogger(__name__)


def run_preflight_architecture_check(module: Emg2PoseModule) -> bool:
    """
    Quick architecture validation with dummy data.
    Catches bugs before expensive data loading.
    
    Returns:
        True if check passes, False otherwise
    """
    log.info("\n" + "="*80)
    log.info("Running Pre-Flight Architecture Check")
    log.info("="*80)
    
    try:
        # Create dummy batch
        batch_size = 4
        network = getattr(getattr(module, "model", None), "network", None)
        n_channels = int(
            getattr(getattr(network, "config", None), "input_channels", 16)
        )
        sequence_length = 10000
        n_joints = 20
        
        log.info(f"Creating dummy batch: ({batch_size}, {n_channels}, {sequence_length})")
        
        emg_tensor = torch.randn(batch_size, n_channels, sequence_length)
        target = torch.randn(batch_size, n_joints, sequence_length)
        
        # Set model to training mode
        module.train()
        
        # Count parameters
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        log.info(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        log.info(f"Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        # Test 1: Forward pass
        log.info("\n[1/3] Testing forward pass...")
        
        # Create batch dictionary (like real dataloader)
        batch = {
            "emg": emg_tensor,
            "joint_angles": target,
            "no_ik_failure": torch.ones(batch_size, sequence_length, dtype=torch.bool)
        }
        
        # Get provide_initial_pos from module config
        provide_initial_pos = getattr(module, "provide_initial_pos", False)
        
        # Forward pass through module (handles both input and alignment)
        pred, joint_angles, no_ik_failure, temporal_mask = module.model(batch, provide_initial_pos)
        
        log.info(f"✅ Prediction shape: {pred.shape}")
        log.info(f"✅ Target shape: {joint_angles.shape}")
        
        if torch.isnan(pred).any():
            log.error("❌ ERROR: Predictions contain NaN!")
            return False
        
        if torch.isinf(pred).any():
            log.error("❌ ERROR: Predictions contain Inf!")
            return False
        
        # Test 2: Loss computation
        log.info("\n[2/3] Testing loss computation...")
        
        # Align shapes if needed (for STFT models where PoseModule skips alignment)
        try:
            from emg2pose.stft_transformer_arch import STFTTransformer
            from emg2pose.circular_stft_transformer_arch import CyclicSpectralTransformer
            from emg2pose.stft_vit_arch import STFTViT
        except ImportError:
            STFTTransformer = CyclicSpectralTransformer = STFTViT = type(None)

        network = getattr(module.model, "network", None)
        if isinstance(network, (STFTTransformer, CyclicSpectralTransformer, STFTViT)):
            # STFT alignment logic from LightningModule
            pred = pred.transpose(1, 2)  # (B, T, J) -> (B, J, T)
            
            if isinstance(network, (STFTTransformer, STFTViT)):
                hop = network.stft.config.stft_hop_length
                window = network.stft.config.stft_window_length
            else:
                hop = network.config.filter_stride
                window = network.config.filter_length
                
            num_windows = pred.shape[2]
            centers = [min(i * hop + window // 2, joint_angles.shape[2] - 1) for i in range(num_windows)]
            
            # Downsample target to match prediction
            joint_angles = joint_angles[:, :, centers]
            log.info(f"Aligned shapes - Pred: {pred.shape}, Target: {joint_angles.shape}")
        
        # Loss computation
        loss = torch.nn.functional.mse_loss(pred, joint_angles)
        log.info(f"✅ Loss computed: {loss.item():.4f}")

        log.info(f"✅ Loss computed: {loss.item():.4f}")
        
        # Test 3: Backward pass
        log.info("\n[3/3] Testing backward pass (gradients)...")
        loss.backward()
        
        # Check gradients
        has_grad = 0
        nan_grad = 0
        
        for name, param in module.named_parameters():
            if param.grad is not None:
                has_grad += 1
                if torch.isnan(param.grad).any():
                    log.error(f"❌ NaN gradient in: {name}")
                    nan_grad += 1
        
        if nan_grad > 0:
            log.error(f"❌ ERROR: {nan_grad} parameters have NaN gradients!")
            return False
        
        log.info(f"✅ Gradients computed: {has_grad} parameters")
        
        log.info("\n" + "="*80)
        log.info("✅ Pre-Flight Check PASSED - Architecture ready for training!")
        log.info("="*80 + "\n")
        
        return True
        
    except Exception as e:
        log.error(f"\n❌ Pre-Flight Check FAILED: {e}")
        import traceback
        traceback.print_exc()
        log.error("\n" + "="*80)
        log.error("Fix architecture errors before training!")
        log.error("="*80)
        return False


def _limit_inductor_compile_threads(n: int = 8) -> None:
    """Keep the inductor subprocess pool small; the default is min(32, ncpu)."""
    os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(n)
    try:
        import torch._inductor.config as inductor_config

        inductor_config.compile_threads = n
    except Exception:
        pass
    log.info("TORCHINDUCTOR_COMPILE_THREADS=%s", n)


def _cleanup_train_target_cache(datamodule) -> None:
    cache_dir = getattr(datamodule, "train_target_cache_dir", None)
    if not cache_dir or not getattr(datamodule, "delete_target_cache_after_run", True):
        return
    from emg2pose.memmap_dataset import reset_memmap_handles
    from emg2pose.window_target_cache import delete_window_target_cache

    for name in ("train_dataset", "val_dataset", "test_dataset"):
        dataset = getattr(datamodule, name, None)
        if dataset is not None:
            reset_memmap_handles(dataset)
    delete_window_target_cache(cache_dir)
    log.info(
        "Ephemeral train target cache removed; EMG memmap and window-index npz kept"
    )


def make_data_module(config: DictConfig):
    """Create datamodule from experiment config."""

    # Dataset session paths
    def _full_paths(root: str, dataset: ListConfig) -> list[Path]:
        # sessions = [session["session"] for session in dataset]
        sessions = dataset
        return [
            Path(root).expanduser().joinpath(f"{session}.hdf5") for session in sessions
        ]

    splits = instantiate(config.data_split)
    train_sessions = _full_paths(config.data_location, splits["train"])
    val_sessions = _full_paths(config.data_location, splits["val"])
    test_sessions = _full_paths(config.data_location, splits["test"])

    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=train_sessions,
        val_sessions=val_sessions,
        test_sessions=test_sessions,
        metadata_csv=config.datamodule.get("metadata_csv", None) or config.get("metadata_file", None),
        data_location=config.get("data_location", None),
        skip_ik_failures=config.datamodule.get("skip_ik_failures", False),
    )

    # Instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    datamodule.train_transforms = _build_transform(config.transforms.train)
    datamodule.val_transforms = _build_transform(config.transforms.val)
    datamodule.test_transforms = _build_transform(config.transforms.test)

    return datamodule


def make_lightning_module(config: DictConfig):
    """Create lightning module from experiment config."""
    return Emg2PoseModule(
        network_conf=config.pose_module,
        optimizer_conf=config.optimizer,
        lr_scheduler_conf=config.lr_scheduler,
        provide_initial_pos=config.provide_initial_pos,
        loss_weights=config.loss_weights,
        emg_dropout=config.emg_dropout,
        pre_tds_mask_prob=getattr(config, "pre_tds_mask_prob", 0.0),
    )


def train(
    config: DictConfig,
    extra_callbacks: Sequence[Callable] | None = None,
):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")
    _limit_inductor_compile_threads(
        int(config.get("inductor_compile_threads", 8))
    )

    # PyTorch 2.6 defaulted torch.load to weights_only=True, which refuses
    # omegaconf objects pickled inside our Lightning checkpoints (not just
    # DictConfig/ListConfig — also ContainerMetadata). Same trusted-ckpt
    # override as test_analysis.py.
    if not getattr(torch.load, "_emg2pose_trust_ckpt", False):
        _orig_torch_load = torch.load

        def _trusting_torch_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_torch_load(*args, **kwargs)

        _trusting_torch_load._emg2pose_trust_ckpt = True  # type: ignore[attr-defined]
        torch.load = _trusting_torch_load

    matmul_precision = config.get("matmul_precision", None)
    if matmul_precision:
        torch.set_float32_matmul_precision(matmul_precision)
        log.info(f"torch.set_float32_matmul_precision({matmul_precision!r})")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    if config.checkpoint is not None:
        log.info(f"Loading from checkpoint {config.checkpoint}")
        module = Emg2PoseModule.load_from_checkpoint(
            config.checkpoint,
            network=config.network,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
        )
    else:
        log.info(f"Instantiating LightningModule {Emg2PoseModule}")
        module = make_lightning_module(config)
    compile_flag = config.get("compile_network", False)
    if compile_flag:
        module.compile_network = compile_flag
        log.info(f"Will torch.compile pose network after CUDA move: {compile_flag}")

    # Pre-flight architecture check before loading data
    log.info("\n🚀 Running quick architecture validation before data loading...")
    if not run_preflight_architecture_check(module):
        raise RuntimeError(
            "Pre-flight architecture check failed! "
            "Fix architecture errors before training. "
            "See logs above for details."
        )
    # Clear gradients accumulated during the preflight backward pass so they
    # don't pollute the first real optimizer step.
    module.zero_grad()

    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = make_data_module(config)

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    if extra_callbacks is not None:
        callbacks.extend(extra_callbacks)

    # Detect resume checkpoint BEFORE building the Trainer so we can pin the
    # TensorBoard logger to version_0 — otherwise Lightning creates version_1.
    resume_ckpt = config.get("resume_checkpoint", None)
    if resume_ckpt is None:
        # Auto-detect last.ckpt in the current hydra output dir so that
        # re-running the exact same command resumes automatically.
        auto_last = Path("lightning_logs/version_0/checkpoints/last.ckpt")
        if auto_last.exists():
            resume_ckpt = str(auto_last)
            log.info(f"Auto-resuming from last checkpoint: {resume_ckpt}")

    trainer_kwargs = dict(config.trainer)
    if resume_ckpt is not None:
        # Pin logger to version_0 so TensorBoard events append to the same run
        # and EpochMetricsLogger writes to the same CSV (not a new version dir).
        from pytorch_lightning.loggers import TensorBoardLogger
        logger = TensorBoardLogger(
            save_dir=".",          # relative to hydra.run.dir (cwd)
            name="lightning_logs",
            version=0,             # pin to existing version_0
        )
        trainer_kwargs["logger"] = logger
        log.info("Resume mode: TensorBoard logger pinned to lightning_logs/version_0")

    accumulate_grad_batches = config.get("accumulate_grad_batches", None)
    if accumulate_grad_batches is not None:
        trainer_kwargs["accumulate_grad_batches"] = accumulate_grad_batches

    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=callbacks,
    )

    results = {}
    try:
        if config.train:

            if resume_ckpt is not None:
                log.info(f"Resuming training from checkpoint: {resume_ckpt}")
            trainer.fit(module, datamodule, ckpt_path=resume_ckpt)

            # Load the best checkpoint
            checkpoint_callback = trainer.checkpoint_callback
            if checkpoint_callback is None:
                raise RuntimeError("No checkpoint callback found in trainer")
            best_checkpoint_path = checkpoint_callback.best_model_path
            from omegaconf.base import ContainerMetadata
            if hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals([ContainerMetadata]):
                    module = module.__class__.load_from_checkpoint(best_checkpoint_path)
            else:
                # torch < 2.4 — weights_only defaults to False, no safe_globals needed
                module = module.__class__.load_from_checkpoint(best_checkpoint_path)

            results["best_checkpoint"] = best_checkpoint_path

        if config.eval:

            # Compute validation and test set metrics
            module.eval()
            val_metrics = trainer.validate(module, datamodule)
            test_metrics = trainer.test(module, datamodule)

            results["val_metrics"] = val_metrics
            results["test_metrics"] = test_metrics

        pprint.pprint(results, sort_dicts=False)
    finally:
        _cleanup_train_target_cache(datamodule)


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def cli(config: DictConfig):
    train(config)


if __name__ == "__main__":
    cli()
