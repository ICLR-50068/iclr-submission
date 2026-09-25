# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass

import hydra
import pandas as pd
import pytorch_lightning as pl
import torch
from emg2pose.data import WindowedEmgDataset

from emg2pose.train import make_lightning_module
from emg2pose.transforms import Compose
from hydra.utils import instantiate

from omegaconf import DictConfig, ListConfig
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

DEFAULT_DATA_DIR = "/emg2pose_data/"

log = logging.getLogger(__name__)

# PyTorch 2.6 flipped torch.load's default to weights_only=True. Lightning
# checkpoints saved by this codebase pickle omegaconf/defaultdict objects
# that the safelist unpickler refuses to populate (SETITEM on defaultdict
# is hardcoded out). Force weights_only=False at the torch.load layer so
# every downstream caller (PL's pl_load, etc.) gets the legacy behavior.
# We own the checkpoints, so the trust assumption holds.
_orig_torch_load = torch.load
def _trusting_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _trusting_torch_load


@dataclass
class EMG2PoseEvaluation:
    """
    Run offline evaluation for a trained emg2pose model.

    Metrics are computed for each combination of conditions, e.g. for each
    (generalization, user) or each (generalization, user, stage).
    """

    config: DictConfig
    checkpoint: str
    conditions: list[str]
    window_length: int = 10_000
    split: str = "test"
    skip_ik_failures: bool = True
    batch_size: int = 32

    def __post_init__(self):
        self.data_dir = self.config.data_location
        # Normalize `conditions` in case it's a Hydra/OmegaConf ListConfig or a string.
        try:
            from omegaconf import ListConfig
        except Exception:
            ListConfig = tuple()  # fallback harmless sentinel

        if isinstance(self.conditions, ListConfig):
            self.conditions = list(self.conditions)
        elif isinstance(self.conditions, str):
            self.conditions = [self.conditions]

        self.batch_size = int(self.config.get("batch_size", self.batch_size))
        self.num_workers = int(self.config.get("num_workers", 0))

        self.df = self.get_corpus_df()
        # groupby expects a list of column names (or a single column name)
        self.groupby = self.df.groupby(self.conditions)
        self.module = self.get_module()
        self.dataloaders = self.get_dataloaders()

    def get_corpus_df(self):
        metadata_file = self.config.get("metadata_file", None)
        if not metadata_file:
            metadata_file = os.path.join(self.data_dir, "metadata.csv")
        df = pd.read_csv(str(metadata_file)).query(f"split=='{self.split}'")

        gen_filter = self.config.get("generalization_filter", None)
        if gen_filter:
            n0 = len(df)
            df = df[df["generalization"] == gen_filter]
            print(f"Filtered to generalization=='{gen_filter}': {len(df)}/{n0} rows")

        # Per-user filter — for single-user iteration ("does this one
        # known-good user reproduce its expected number?"). Comma-separated
        # for picking a small set: +user_filter=29ddab35d7 or
        # +user_filter='29ddab35d7,7f2a...'
        user_filter = self.config.get("user_filter", None)
        if user_filter:
            users = {u.strip() for u in str(user_filter).split(",") if u.strip()}
            n0 = len(df)
            df = df[df["user"].isin(users)]
            print(f"Filtered to users={sorted(users)}: {len(df)}/{n0} rows "
                  f"({df['filename'].nunique()} sessions)")
            if df.empty:
                raise RuntimeError(
                    f"No rows match user_filter={user_filter!r}. "
                    f"Available users in split={self.split}: see metadata.csv"
                )

        # Optionally subsample corpus for testing purposes
        corpus_subsample = self.config.get("corpus_subsample", 1)
        if corpus_subsample < 1:
            print(f"Subsampling corpus by {corpus_subsample}")
            df = df.sample(frac=corpus_subsample, random_state=0)

        return df

    def get_module(self):
        # PyTorch 2.6 changed torch.load default to weights_only=True, which
        # blocks omegaconf types stored inside Lightning checkpoints.
        if hasattr(torch.serialization, "add_safe_globals"):
            safe = [DictConfig, ListConfig]
            try:
                from omegaconf.base import ContainerMetadata, Metadata
                safe += [ContainerMetadata, Metadata]
            except Exception:
                pass
            try:
                from omegaconf.nodes import (
                    AnyNode, ValueNode, IntegerNode, FloatNode,
                    BooleanNode, StringNode,
                )
                safe += [AnyNode, ValueNode, IntegerNode, FloatNode,
                         BooleanNode, StringNode]
            except Exception:
                pass
            try:
                import typing
                safe += [typing.Any]
            except Exception:
                pass
            from collections import defaultdict
            safe += [defaultdict, dict, list, tuple, set]
            torch.serialization.add_safe_globals(safe)

        from omegaconf import OmegaConf
        from emg2pose.lightning import Emg2PoseModule

        # ── Resolve rope_mode (which forward pass to use) ────────────────
        # Three sources of input, in priority order:
        #   1. Explicit +rope_mode=<...> on the CLI
        #   2. Legacy +legacy_checkpoint=true (kept for backwards compat) →
        #      maps to rope_mode="legacy_mha".
        #   3. Default "fixed" (current correct order: project then rotate).
        #
        # Valid rope_mode values:
        #   "fixed"                   — current correct order.
        #   "legacy_mha"              — old nn.MultiheadAttention with RoPE
        #                               applied to features before W_q/W_k.
        #   "legacy_v2_pre_transpose" — separate w_q/w_k/w_v projections, but
        #                               RoPE applied to projected Q,K BEFORE
        #                               the (1,2) transpose to head-major.
        #                               Rotation runs over n_heads axis.
        #   "legacy_v2_pre_proj"      — separate w_q/w_k/w_v projections, but
        #                               RoPE applied to LayerNorm(h) BEFORE
        #                               W_q/W_k. Rotation runs over n_heads.
        #
        # The two legacy_v2 variants are intended for checkpoints whose
        # state_dict already has separate w_q/w_k/w_v keys but were trained
        # with a pre-fix forward pass. Try one, then the other if metrics
        # still look wrong.
        valid_modes = (
            "fixed", "legacy_mha",
            "legacy_v2_pre_transpose", "legacy_v2_pre_proj",
        )
        rope_mode = str(self.config.get("rope_mode", "fixed"))
        if self.config.get("legacy_checkpoint", False) and rope_mode == "fixed":
            rope_mode = "legacy_mha"
        if rope_mode not in valid_modes:
            raise RuntimeError(
                f"Unknown rope_mode={rope_mode!r}. Valid options: {valid_modes}"
            )

        # ── Inspect checkpoint: validate against requested rope_mode ─────
        ckpt = torch.load(self.config.checkpoint,
                          map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        saved_hparams = ckpt.get("hyper_parameters", {})

        # Detect MHA-style block keys vs separate-projection block keys.
        # We match only block-prefixed keys to ignore channel_pooling.attn.
        has_block_mha = any(
            k.endswith(".attn.in_proj_weight") and ".blocks." in k for k in sd
        )
        has_block_split = any(
            k.endswith(".w_q.weight") and ".blocks." in k for k in sd
        )

        # ── Build network_conf to pass to load_from_checkpoint ───────────
        # Use the SAVED network_conf from the checkpoint (it has the right
        # dimensions: d_model, d_ff, etc.), not the experiment-config defaults.
        saved_network_conf = saved_hparams.get("network_conf", None)
        if saved_network_conf is None:
            raise RuntimeError(
                "Checkpoint does not contain saved 'network_conf' in "
                "hyper_parameters. Cannot infer architecture dims. "
                "Use train.py-style restore instead."
            )
        network_conf = OmegaConf.create(
            OmegaConf.to_container(saved_network_conf, resolve=True)
            if OmegaConf.is_config(saved_network_conf)
            else dict(saved_network_conf)
        )

        # Locate the actual network node (PoseModule wraps it under .network).
        target_node = (
            network_conf.network
            if "network" in network_conf and OmegaConf.is_config(network_conf.network)
            else network_conf
        )
        target = str(target_node.get("_target_", ""))
        is_cs_tds_ct = "cs_tds_ct" in target

        # rope_mode only applies to the cs_tds_ct architecture. For other
        # networks (TdsNetwork, etc.) we skip injection entirely so legacy
        # eval flows keep working. If the user explicitly requested a
        # non-default rope_mode on a non-cs_tds_ct checkpoint, raise — the
        # flag would be silently ignored otherwise.
        rope_mode_explicit = (
            self.config.get("rope_mode", None) is not None
            or self.config.get("legacy_checkpoint", False)
        )
        if rope_mode_explicit and not is_cs_tds_ct:
            raise RuntimeError(
                f"\n{'=' * 70}\n"
                f"+rope_mode=/+legacy_checkpoint flags only apply to the\n"
                f"cs_tds_ct architecture. This checkpoint targets:\n"
                f"  {target!r}\n"
                f"  Checkpoint: {self.config.checkpoint}\n\n"
                f"Drop the flag and re-run — the network has no RoPE bug to\n"
                f"reproduce.\n"
                f"{'=' * 70}"
            )

        # ── TdsNetwork / vemg2pose: detect legacy norm layout ────────────
        # Old checkpoints (pre-GroupNorm refactor) store keys like
        #   layers.0.norm.weight              (separate self.norm child)
        #   ...tds_conv_blocks.0.layer_norm.weight
        # Current code uses self.conv.1 (GroupNorm) and self.group_norm.
        # If we detect those legacy keys, flip every Conv1dBlock/TdsStage
        # in the saved network_conf to legacy_norm=True so the rebuilt
        # modules have the matching layout.
        is_tds_legacy_norm = any(
            (k.endswith(".norm.weight") and ".layers." in k and ".conv." not in k)
            or k.endswith(".layer_norm.weight")
            for k in sd
        )
        if is_tds_legacy_norm and not is_cs_tds_ct:
            log.info(
                "Detected legacy TdsNetwork norm layout in checkpoint. "
                "Injecting legacy_norm=True into Conv1dBlock/TdsStage entries."
            )
            OmegaConf.set_struct(network_conf, False)

            def _inject_legacy_norm(node):
                """Walk the network_conf tree and set legacy_norm=True on
                every Conv1dBlock/TdsStage we find."""
                if not OmegaConf.is_config(node):
                    return
                if hasattr(node, "_target_"):
                    tgt = str(node.get("_target_", ""))
                    if tgt.endswith(".Conv1dBlock") or tgt.endswith(".TdsStage"):
                        node.legacy_norm = True
                # Recurse into mapping values and list items
                if OmegaConf.is_dict(node):
                    for k in node.keys():
                        _inject_legacy_norm(node[k])
                elif OmegaConf.is_list(node):
                    for i in range(len(node)):
                        _inject_legacy_norm(node[i])

            _inject_legacy_norm(network_conf)
            OmegaConf.set_struct(network_conf, True)

        if is_cs_tds_ct:
            # Sanity-check: requested rope_mode must match the stored key layout.
            if rope_mode == "legacy_mha" and not has_block_mha:
                raise RuntimeError(
                    f"\n{'=' * 70}\n"
                    f"rope_mode=legacy_mha requires a checkpoint with "
                    f"nn.MultiheadAttention keys (blocks.*.attn.in_proj_weight),\n"
                    f"but this checkpoint has separate w_q/w_k/w_v projections.\n"
                    f"  Checkpoint: {self.config.checkpoint}\n\n"
                    f"Try one of: rope_mode=fixed, "
                    f"rope_mode=legacy_v2_pre_transpose, rope_mode=legacy_v2_pre_proj.\n"
                    f"{'=' * 70}"
                )
            if rope_mode in ("fixed", "legacy_v2_pre_transpose", "legacy_v2_pre_proj") \
                    and not has_block_split and has_block_mha:
                raise RuntimeError(
                    f"\n{'=' * 70}\n"
                    f"rope_mode={rope_mode!r} requires a checkpoint with separate\n"
                    f"w_q/w_k/w_v projection keys, but this checkpoint uses the\n"
                    f"old nn.MultiheadAttention layout (blocks.*.attn.in_proj_weight).\n"
                    f"  Checkpoint: {self.config.checkpoint}\n\n"
                    f"Use +rope_mode=legacy_mha (or +legacy_checkpoint=true) instead.\n"
                    f"{'=' * 70}"
                )

            log.info(
                "Loading cs_tds_ct checkpoint with rope_mode=%s "
                "(block keys: mha=%s, split=%s)",
                rope_mode, has_block_mha, has_block_split,
            )
            if rope_mode != "fixed":
                log.warning(
                    "rope_mode=%s reproduces a pre-fix forward pass. "
                    "If the trained metrics were collected under that forward "
                    "pass, this is what you want; otherwise switch to "
                    "rope_mode=fixed.", rope_mode,
                )

            OmegaConf.set_struct(network_conf, False)
            target_node.rope_mode = rope_mode
            target_node.use_legacy_attention = (rope_mode == "legacy_mha")
            OmegaConf.set_struct(network_conf, True)
        else:
            log.info(
                "Loading non-cs_tds_ct checkpoint (target=%s); "
                "rope_mode is not applicable.", target,
            )

        # Optimizer/scheduler: pass dummy ones so the constructor accepts them.
        # We're only loading weights for inference; optimiser is irrelevant.
        dummy_opt   = OmegaConf.create({"_target_": "torch.optim.Adam", "lr": 1e-3})
        dummy_sched = OmegaConf.create({
            "scheduler": {"_target_": "torch.optim.lr_scheduler.ConstantLR",
                          "factor": 1.0, "total_iters": 1},
            "interval": "epoch",
        })

        # ── Resolve pose-norm files BEFORE Lightning loads ───────────────
        # Emg2PoseModule.__init__ does `np.load("pose_mean_train_split.npy")`
        # via the relative path from its default arg. Hydra has already
        # chdir'd to hydra.run.dir (typically a fresh empty folder), so
        # the load fails and the lightning module silently falls back to
        # `pose_mean=zeros(20)`, `pose_std=ones(20)`. With raw-radians
        # preds and targets that no-ops the denormalize block in
        # test_step (so MAE comes out in correct degrees), but if a future
        # version starts predicting z-score it would silently break. Pass
        # absolute paths through `+pose_mean_path=` / `+pose_std_path=`
        # if you have them outside the run dir; otherwise we search a few
        # standard locations relative to the original CWD and the package.
        import os as _os
        from hydra.utils import get_original_cwd as _orig_cwd
        try:
            search_dirs = [_orig_cwd()]
        except Exception:
            search_dirs = [_os.getcwd()]
        # package root and one level above (where the user keeps the npy files)
        _pkg_root = _os.path.dirname(_os.path.dirname(__file__))
        search_dirs += [_pkg_root, _os.path.dirname(_pkg_root)]

        def _resolve(name, override_key):
            cli = self.config.get(override_key, None)
            if cli is not None and _os.path.isabs(str(cli)) and _os.path.exists(str(cli)):
                return str(cli)
            for d in search_dirs:
                p = _os.path.join(d, name)
                if _os.path.exists(p):
                    return _os.path.abspath(p)
            return name  # let Emg2PoseModule print its FileNotFound warning

        pose_mean_path = _resolve("pose_mean_train_split.npy", "pose_mean_path")
        pose_std_path  = _resolve("pose_std_train_split.npy",  "pose_std_path")
        log.info("Resolved pose_mean_path=%s", pose_mean_path)
        log.info("Resolved pose_std_path =%s", pose_std_path)

        module = Emg2PoseModule.load_from_checkpoint(
            self.config.checkpoint,
            strict=True,
            network_conf=network_conf,
            optimizer_conf=dummy_opt,
            lr_scheduler_conf=dummy_sched,
            pose_mean_path=pose_mean_path,
            pose_std_path=pose_std_path,
        )
        module.eval()
        return module

    def get_dataloaders(self) -> list[DataLoader]:
        """
        Get list of dataloaders, each corresponding to a single groupby condition
        (e.g., [user, stage]).
        """

        transforms = Compose(
            instantiate(self.config.transforms[self.split], _convert_="all")
        )
        
        # ── Test-time EMG perturbations (robustness sweeps) ──────────────
        # All three are appended after the base val/test transforms so they
        # operate on the (T, C) windowed tensor right before it goes into
        # the model. Each is opt-in via Hydra:
        #   +rotation=N             integer cyclic shift over channels
        #   +amplitude_jitter_std=σ per-channel log-normal gain (matches
        #                           cs_tds_ct training augmentation)
        #   +channel_dropout_p=p    bernoulli(p) zeroing of entire channels
        extras = []

        rotation = self.config.get("rotation", 0)
        if rotation != 0:
            from emg2pose.transforms import FixedChannelRotation
            print(f"Applying fixed channel rotation: {rotation} positions")
            extras.append(FixedChannelRotation(rotation=rotation))

        amp_jitter_std = float(self.config.get("amplitude_jitter_std", 0.0))
        if amp_jitter_std > 0:
            from emg2pose.transforms import AmplitudeJitter
            print(f"Applying amplitude jitter: log-normal σ={amp_jitter_std}")
            extras.append(AmplitudeJitter(std=amp_jitter_std))

        chan_drop_p = float(self.config.get("channel_dropout_p", 0.0))
        if chan_drop_p > 0:
            from emg2pose.transforms import ChannelDropout
            print(f"Applying channel dropout: p={chan_drop_p}")
            extras.append(ChannelDropout(p=chan_drop_p))

        if extras:
            transforms.transforms = list(transforms.transforms) + extras
        
        # Window-length convention (matches WindowedEmgDataModule used at
        # training time): `self.window_length` IS the full window pulled from
        # the HDF5, INCLUDING any left/right context the network needs. The
        # model then trims its own context internally and predicts
        # `window_length - left_context - right_context` frames per window.
        #
        # The earlier code did `effective = window_length + context`, which
        # double-counted context for any model with non-zero left_context
        # (e.g. vemg2pose's TdsNetwork has left_context=1790 → eval saw
        # 13580-sample windows when training used 11790). For cs_tds_ct
        # (left_context=0) the behaviour is unchanged.
        context_length = (
            self.module.model.left_context + self.module.model.right_context
        )
        effective_window_length = self.window_length
        prediction_length = self.window_length - context_length
        if prediction_length <= 0:
            raise RuntimeError(
                f"window_length={self.window_length} is shorter than the "
                f"model's required context ({context_length}). Pass a larger "
                f"+window_length= or fix datamodule.val_test_window_length."
            )
        stride = prediction_length

        print("Creating dataloaders for each condition.")
        memmap_dir = None
        if "datamodule" in self.config:
            memmap_dir = self.config.datamodule.get("memmap_dir", None)
        channel_idx = None
        if "datamodule" in self.config:
            channel_idx = self.config.datamodule.get("emg_channel_indices", None)
        if memmap_dir:
            print(f"Using memmap corpus: {memmap_dir} channels={channel_idx}")
        dataloaders = []
        for _, df_ in tqdm(self.groupby):
            if memmap_dir:
                from emg2pose.memmap_dataset import (
                    MemmapWindowedEmgDataset,
                    memmap_worker_init_fn,
                )

                padding = tuple(
                    int(x) for x in self.config.datamodule.get("padding", [0, 0])
                )
                filenames = [str(name) for name in df_["filename"].tolist()]
                dataset = MemmapWindowedEmgDataset(
                    memmap_dir=memmap_dir,
                    window_length=effective_window_length,
                    stride=stride,
                    padding=padding,
                    jitter=False,
                    transform=transforms,
                    skip_ik_failures=self.skip_ik_failures,
                    allowed_splits=[self.split],
                    allowed_filenames=filenames,
                    apply_per_dataset_emg_norm=bool(
                        self.config.datamodule.get("apply_per_dataset_emg_norm", False)
                    ),
                    emg_norm_mean=float(
                        self.config.datamodule.get("emg_norm_mean", 0.0)
                    ),
                    emg_norm_std=float(
                        self.config.datamodule.get("emg_norm_std", 18.49)
                    ),
                    emg_channel_indices=channel_idx,
                )
                loader_kwargs = {
                    "batch_size": self.batch_size,
                    "shuffle": False,
                    "num_workers": self.num_workers,
                    "pin_memory": True,
                    "persistent_workers": False,
                    "prefetch_factor": 2 if self.num_workers > 0 else None,
                }
                if self.num_workers > 0:
                    loader_kwargs["worker_init_fn"] = memmap_worker_init_fn
                dataloaders.append(DataLoader(dataset, **loader_kwargs))
                continue
            dataset: ConcatDataset = ConcatDataset(
                [
                    WindowedEmgDataset(
                        os.path.join(self.data_dir, hdf5_path),
                        transform=transforms,
                        window_length=effective_window_length,
                        stride=stride,
                        jitter=False,
                        skip_ik_failures=self.skip_ik_failures,
                    )
                    for hdf5_path in df_.filename + ".hdf5" if os.path.exists(os.path.join(self.data_dir, hdf5_path))
                ]
            )
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=2 if self.num_workers > 0 else None,
            )
            dataloaders.append(dataloader)

        return dataloaders

    def create_results_df(self, results: list[dict[str, float]]):
        """Convert results list to a dataframe with all condition information."""
        records = []
        for (vals, _), result in zip(self.groupby, results):
            record = dict(zip(self.conditions, vals))
            result = {k.split("/")[0]: v for k, v in result.items()}
            record.update(result)
            records.append(record)
        return pd.DataFrame(records)

    def evaluate(self) -> pd.DataFrame:
        """Run analysis for split.

        For per-session evaluation (one dataloader per file/session), Lightning
        Trainer.test() with N dataloaders pays ~2s per dataloader of overhead
        (DistributedSampler setup, DDP barrier, dataloader_idx switch, logger)
        even when each dataloader is a single 0.1s batch. With 6000+ sessions
        that overhead dominates: 4h of trainer plumbing for ~10 minutes of
        actual GPU work.

        Default to a fast single-process custom loop that bypasses the trainer
        and replicates the relevant slice of test_step + on_test_end inline.
        Set +fast_eval=false to fall back to the original trainer.test path.
        """
        # Tag each dataloader with its condition values so per-condition CSV
        # rows can be built (used by both the fast path and the lightning path).
        self.module.dataloader_labels = [
            dict(zip(self.conditions,
                     vals if isinstance(vals, tuple) else (vals,)))
            for vals, _ in self.groupby
        ]

        if bool(self.config.get("fast_eval", True)):
            return self._evaluate_fast()

        trainer = pl.Trainer(**self.config.trainer)
        results = trainer.test(self.module, dataloaders=self.dataloaders, verbose=True)
        return self.create_results_df(results)

    def _evaluate_fast(self) -> pd.DataFrame:
        """Single-process per-session inference without Lightning Trainer.

        Outputs match the trainer.test path:
          - returns a results_df with one row per condition (from create_results_df)
          - writes per_joint_mae_analysis.csv to the same lightning_logs dir
        """
        from collections import defaultdict

        # Guard against torchrun / Lightning DDP launcher: with WORLD_SIZE>1
        # every rank would redundantly run the loop and clobber the CSV. The
        # fast path is single-process by design — non-zero ranks bail early.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size > 1:
            if local_rank != 0:
                log.info(
                    f"fast_eval: skipping rank {local_rank}; only rank 0 runs. "
                    f"Launch with `python test_analysis.py ...` (single process, "
                    f"trainer.devices=1) to avoid spawning idle ranks."
                )
                return pd.DataFrame()
            log.warning(
                f"fast_eval: detected WORLD_SIZE={world_size} but the fast loop "
                f"is single-process; only rank 0 will do work. To use all GPUs, "
                f"shard sessions manually (e.g. user_filter chunks) and launch "
                f"one process per chunk."
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.module.to(device).eval()
        metrics_list = model.metrics_list
        pose_mean = model.pose_mean.to(device)
        pose_std = model.pose_std.to(device)

        # STFT-family alignment matches what test_step does for these nets.
        try:
            from emg2pose.stft_transformer_arch import STFTTransformer
            from emg2pose.circular_stft_transformer_arch import CyclicSpectralTransformer
            from emg2pose.stft_vit_arch import STFTViT
        except ImportError:
            STFTTransformer = CyclicSpectralTransformer = STFTViT = type(None)
        network = getattr(model.model, "network", None)
        is_stft = isinstance(network, (STFTTransformer, CyclicSpectralTransformer, STFTViT))

        joint_names = [
            "wrist_flex", "wrist_dev", "pro_sup", "thumb_flex", "thumb_abd",
            "thumb_mcp_flex", "thumb_mcp_abd", "index_flex", "index_abd",
            "middle_flex", "middle_abd", "ring_flex", "ring_abd", "pinky_flex",
            "pinky_abd", "thumb_pip", "index_pip", "middle_pip", "ring_pip",
            "pinky_pip",
        ]

        per_loader_metrics: list[dict[str, float]] = []
        per_joint_rows: list[dict] = []
        labels = getattr(model, "dataloader_labels", None)

        print(f"Fast per-session inference on {len(self.dataloaders)} sessions (device={device}).")

        with torch.inference_mode():
            for di, dl in enumerate(tqdm(self.dataloaders)):
                # Per-batch metric values; averaged across batches at end
                # (matches Lightning's default on_epoch=True reduce_fx="mean").
                metric_vals: dict[str, list[float]] = defaultdict(list)
                joint_errs_chunks: list[torch.Tensor] = []

                for batch in dl:
                    batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                             for k, v in batch.items()}
                    batch["no_ik_failure"] = model.update_ik_failure_mask(batch["no_ik_failure"])
                    preds, targets, no_ik_failure, _ = model.forward(batch)

                    if is_stft:
                        preds = preds.transpose(1, 2)
                        if isinstance(network, (STFTTransformer, STFTViT)):
                            hop = network.stft.config.stft_hop_length
                            window = network.stft.config.stft_window_length
                        else:
                            hop = network.config.filter_stride
                            window = network.config.filter_length
                        num_windows = preds.shape[2]
                        centers = [min(i * hop + window // 2, targets.shape[2] - 1)
                                   for i in range(num_windows)]
                        targets = targets[:, :, centers]
                        if no_ik_failure.dim() == 3:
                            no_ik_failure = no_ik_failure.all(dim=1)[:, centers]
                        elif no_ik_failure.dim() == 2:
                            no_ik_failure = no_ik_failure[:, centers]
                        elif no_ik_failure.dim() == 1:
                            no_ik_failure = no_ik_failure.unsqueeze(1).expand(-1, num_windows)

                    # Per-joint MAE (mirrors test_step)
                    valid_mask = no_ik_failure.any(dim=1)
                    if valid_mask.sum() > 0:
                        preds_dn = preds[valid_mask] * pose_std.view(1, -1, 1) + pose_mean.view(1, -1, 1)
                        targets_dn = targets[valid_mask] * pose_std.view(1, -1, 1) + pose_mean.view(1, -1, 1)
                        batch_errors = torch.abs(torch.rad2deg(preds_dn) - torch.rad2deg(targets_dn)).mean(dim=2).cpu()
                        joint_errs_chunks.append(batch_errors)

                    # General metrics (test_loss et al.)
                    for metric in metrics_list:
                        out = metric(preds, targets, no_ik_failure, "test")
                        for k, v in out.items():
                            if v is None:
                                continue
                            if torch.is_tensor(v) and (v.numel() == 0 or torch.isnan(v).any()):
                                continue
                            metric_vals[k].append(float(v))

                session_metrics = {k: sum(vs) / len(vs) for k, vs in metric_vals.items() if vs}
                # Provide test_loss for parity with the trainer path
                if "test_loss" not in session_metrics:
                    loss = 0.0
                    for loss_name, weight in (model.loss_weights or {"mae": 1}).items():
                        v = session_metrics.get(f"test_{loss_name}")
                        if v is not None:
                            loss += v * weight
                    session_metrics["test_loss"] = loss
                per_loader_metrics.append(session_metrics)

                if joint_errs_chunks:
                    errs = torch.cat(joint_errs_chunks)
                    mean_pj = errs.mean(dim=0)
                    std_pj = errs.std(dim=0) if errs.shape[0] > 1 else torch.zeros_like(mean_pj)
                    label = labels[di] if labels is not None and di < len(labels) else {}
                    for j, name in enumerate(joint_names):
                        per_joint_rows.append({
                            "dataloader_idx": int(di),
                            **label,
                            "joint_name": name,
                            "mean_mae_deg": mean_pj[j].item(),
                            "std_mae_deg": std_pj[j].item(),
                            "n_samples": int(errs.shape[0]),
                        })

        save_dir = os.path.join(os.getcwd(), "lightning_logs")
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, "per_joint_mae_analysis.csv")
        pd.DataFrame(per_joint_rows).to_csv(output_path, index=False)
        log.info(f"Saved detailed per-joint MAE analysis to {output_path}")

        return self.create_results_df(per_loader_metrics)


@hydra.main(config_path="../config", config_name="base", version_base="1.1")
def cli(config: DictConfig):
    # Determine window length from config. Precedence:
    #   1. Top-level +window_length=N on the CLI (shortcut for sweeps)
    #   2. config.datamodule.val_test_window_length
    #   3. config.datamodule.window_length
    #   4. 10_000 default
    window_length = 10_000
    if "datamodule" in config:
        if "val_test_window_length" in config.datamodule:
            window_length = config.datamodule.val_test_window_length
        elif "window_length" in config.datamodule:
            window_length = config.datamodule.window_length
    cli_window_length = config.get("window_length", None)
    if cli_window_length is not None:
        window_length = int(cli_window_length)

    print(f"Running evaluation with window_length={window_length}")

    # Optional reproducibility seed for stochastic test-time perturbations
    # (amplitude_jitter, channel_dropout). Without this, each run draws
    # fresh noise and metrics jitter run-to-run.
    seed = config.get("seed", None)
    if seed is not None:
        pl.seed_everything(int(seed), workers=True)
        print(f"Seed: {seed}")

    # Check for channel rotation parameter
    rotation = config.get("rotation", 0)
    if rotation != 0:
        print(f"⚠️  Channel rotation enabled: {rotation} positions")
        print(f"   This will circularly rotate EMG channels for testing sensitivity")
    else:
        print("Channel rotation: disabled (rotation=0)")

    # Surface other test-time perturbations
    amp_jitter_std = float(config.get("amplitude_jitter_std", 0.0))
    chan_drop_p = float(config.get("channel_dropout_p", 0.0))
    if amp_jitter_std > 0:
        print(f"⚠️  Amplitude jitter enabled: log-normal σ={amp_jitter_std}")
    if chan_drop_p > 0:
        print(f"⚠️  Channel dropout enabled: p={chan_drop_p}")

    # Surface RoPE mode for the user. Selects which forward-pass variant
    # the attention blocks use; affects which checkpoint forward pass is
    # reproduced. Defaults to "fixed" (current correct order). The legacy
    # +legacy_checkpoint=true flag is treated as rope_mode=legacy_mha.
    cli_rope_mode = config.get("rope_mode", None)
    legacy_flag = config.get("legacy_checkpoint", False)
    if cli_rope_mode is not None:
        effective_rope_mode = str(cli_rope_mode)
        print(f"RoPE mode: {effective_rope_mode} (cs_tds_ct only)")
    elif legacy_flag:
        effective_rope_mode = "legacy_mha"
        print(f"RoPE mode: {effective_rope_mode} (cs_tds_ct only)")
    if cli_rope_mode is not None and effective_rope_mode != "fixed":
        print(
            "   Reproduces a pre-fix forward pass for older checkpoints. "
            "Use +rope_mode=fixed to switch back."
        )

    # Allow overriding split and conditions from config/CLI
    split = config.get("split", "test")
    conditions = config.get("conditions", ["generalization", "user"])

    if isinstance(conditions, ListConfig):
        conditions = list(conditions)
    elif isinstance(conditions, str):
        conditions = [conditions]

    print(f"Evaluating on split='{split}' with conditions={conditions}")

    skip_ik_failures = True
    if "datamodule" in config and "skip_ik_failures" in config.datamodule:
        skip_ik_failures = bool(config.datamodule.skip_ik_failures)
    print(f"skip_ik_failures={skip_ik_failures}")

    evaluation = EMG2PoseEvaluation(
        config=config,
        checkpoint=config.checkpoint,
        conditions=conditions,
        window_length=window_length,
        split=split,
        skip_ik_failures=skip_ik_failures,
    )
    results_df = evaluation.evaluate()

    # Save results to a csv in the logs folder
    results_filename = os.path.join(os.getcwd(), "results.csv")
    print(f"Saving results to {results_filename}")
    results_df.to_csv(results_filename, index=False)


if __name__ == "__main__":
    cli()
