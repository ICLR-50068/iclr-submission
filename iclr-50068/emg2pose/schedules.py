"""Learning-rate warmup and input-augmentation ramping.

Both exist for the same reason: a large CS-TDS-CT (d_model 768) hit with the
full LR and full input corruption from step 0 collapses onto the mean pose —
predicting a constant hand is an easier immediate win than reading heavily
masked EMG, and it does not recover. Easing both in over the first epochs
keeps the model on the input.
"""

import logging
import math

import pytorch_lightning as pl
from torch.optim.lr_scheduler import LRScheduler

log = logging.getLogger(__name__)


class WarmupCosineLR(LRScheduler):
    """Linear warmup, then cosine decay to eta_min over the remaining span.

    Stepped once per epoch (``interval: epoch``), so all durations are epochs.
    The LR is a closed-form function of ``last_epoch`` rather than a running
    product, so resuming mid-run reproduces the correct value.
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        warmup_epochs: int = 5,
        warmup_start_factor: float = 0.01,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.warmup_start_factor = warmup_start_factor
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        e = self.last_epoch

        if self.warmup_epochs > 0 and e < self.warmup_epochs:
            span = 1.0 - self.warmup_start_factor
            factor = self.warmup_start_factor + span * (e / self.warmup_epochs)
            return [base * factor for base in self.base_lrs]

        decay_span = max(1, self.T_max - self.warmup_epochs)
        progress = min(1.0, (e - self.warmup_epochs) / decay_span)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [self.eta_min + (base - self.eta_min) * cosine for base in self.base_lrs]


class AugmentationRamp(pl.Callback):
    """Scale the network's input-augmentation strengths from ~0 up to their
    configured targets over the first ``ramp_epochs`` epochs.

    Targets are passed explicitly rather than read off the network at startup
    so that resuming from a checkpoint mid-ramp does not latch onto an
    already-scaled value.
    """

    def __init__(
        self,
        targets: dict[str, float],
        ramp_epochs: int = 15,
        start_factor: float = 0.0,
    ) -> None:
        super().__init__()
        self.targets = dict(targets)
        self.ramp_epochs = ramp_epochs
        self.start_factor = start_factor
        self._logged_done = False

    def _factor(self, epoch: int) -> float:
        if self.ramp_epochs <= 0 or epoch >= self.ramp_epochs:
            return 1.0
        span = 1.0 - self.start_factor
        return self.start_factor + span * (epoch / self.ramp_epochs)

    @staticmethod
    def _resolve_network(pl_module: pl.LightningModule):
        """Locate the CycloFormer module that owns the aug knobs.

        emg2pose Lightning stores it at ``model.network``. EgoEMG PoseModule
        stores the same object at ``model.featurizer``.
        """
        model = getattr(pl_module, "model", None)
        if model is None:
            return None
        for attr in ("network", "featurizer"):
            net = getattr(model, attr, None)
            if net is not None and hasattr(net, "config"):
                return net
        if hasattr(model, "config"):
            return model
        return None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        network = self._resolve_network(pl_module)
        config = getattr(network, "config", None)
        if config is None:
            if not self._logged_done:
                log.warning("AugmentationRamp: network has no .config; ramp disabled.")
                self._logged_done = True
            return

        factor = self._factor(trainer.current_epoch)
        applied = {}
        for name, target in self.targets.items():
            if not hasattr(config, name):
                log.warning("AugmentationRamp: %r not on network config, skipping.", name)
                continue
            value = target * factor
            setattr(config, name, value)
            applied[name] = value

        if factor < 1.0 or not self._logged_done:
            log.info(
                "AugmentationRamp epoch %d: factor %.2f -> %s",
                trainer.current_epoch,
                factor,
                {k: round(v, 4) for k, v in applied.items()},
            )
            self._logged_done = factor >= 1.0


class ForceConfigLR(pl.Callback):
    """Overwrite resumed warmup LR (3e-6) with cosine from ``lr``.

    Lightning restores the old WarmupCosineLR state from last.ckpt. This
    callback stomps param-group LR at each epoch start so a restart actually
    trains at the intended 3e-4 instead of sitting in the mean-pose basin.
    """

    def __init__(
        self,
        lr: float = 3.0e-4,
        eta_min: float = 5.0e-6,
        T_max: int = 250,
    ) -> None:
        super().__init__()
        self.lr = float(lr)
        self.eta_min = float(eta_min)
        self.T_max = int(T_max)

    def _apply(self, trainer: pl.Trainer, epoch: int) -> None:
        t = max(1, self.T_max)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(epoch, t) / t))
        lr = self.eta_min + (self.lr - self.eta_min) * cosine
        for opt in trainer.optimizers:
            for group in opt.param_groups:
                group["lr"] = lr
        log.info("ForceConfigLR epoch=%d lr=%.4e", epoch, lr)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._apply(trainer, int(trainer.current_epoch))

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._apply(trainer, int(trainer.current_epoch))
