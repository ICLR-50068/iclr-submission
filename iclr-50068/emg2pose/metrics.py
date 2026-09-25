# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F
from emg2pose.constants import (
    EMG_SAMPLE_RATE,
    FINGERS,
    JOINTS,
    LANDMARKS,
    NO_MOVEMENT_LANDMARKS,
    NUM_JOINTS,
    PD_GROUPS,
)

from emg2pose.kinematics import (
    forward_kinematics,
    load_default_hand_model,
    TorchHandModel,
)

PA_MPJPE_EVAL_IDXS = (5, 6, 7, 0, 9, 10, 1, 12, 13, 2, 15, 16, 3, 18, 19, 4)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
    """L1 on masked entries, or None if the mask is empty (no NaN).

    skip_ik_failures=False emits windows that can be all IK-failure zeros.
    L1Loss on an empty slice is NaN and Lightning then kills EarlyStopping.
    """
    if mask is None or not torch.as_tensor(mask).any():
        return None
    selected_pred = pred[mask]
    if selected_pred.numel() == 0:
        return None
    return F.l1_loss(selected_pred, target[mask])


def procrustes_align(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Batched similarity alignment for row-vector point clouds.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted point clouds of shape (N, J, 3).
    target : torch.Tensor
        Target point clouds of shape (N, J, 3).
    eps : float, optional
        Small constant to guard against degenerate point clouds.

    Returns
    -------
    torch.Tensor
        Aligned predicted point clouds with shape (N, J, 3).
    """
    orig_dtype = pred.dtype
    device_type = pred.device.type
    # CUDA gesvdj is not implemented for bfloat16/float16. AMP autocast
    # would recast the covariance matmul even after an explicit .float().
    with torch.autocast(device_type=device_type, enabled=False):
        pred = pred.float()
        target = target.float()
        pred_mean = pred.mean(dim=1, keepdim=True)
        target_mean = target.mean(dim=1, keepdim=True)
        pred_centered = pred - pred_mean
        target_centered = target - target_mean

        covariance = target_centered.transpose(1, 2) @ pred_centered
        u, _, vh = torch.linalg.svd(covariance)
        rotation = u @ vh

        reflected = torch.det(rotation) < 0
        if reflected.any():
            correction = torch.eye(3, device=pred.device, dtype=pred.dtype).repeat(
                pred.shape[0], 1, 1
            )
            correction[reflected, -1, -1] = -1
            rotation = u @ correction @ vh

        rotated_pred = pred_centered @ rotation.transpose(1, 2)
        denom = pred_centered.square().sum(dim=(1, 2), keepdim=True)
        numer = (target_centered * rotated_pred).sum(dim=(1, 2), keepdim=True)
        scale = numer / denom.clamp_min(eps)
        aligned = scale * rotated_pred + target_mean

        valid = denom > eps
        return torch.where(valid, aligned, target).to(orig_dtype)


def masked_pa_mpjpe(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    idxs: tuple[int, ...] = PA_MPJPE_EVAL_IDXS,
) -> torch.Tensor:
    """PA-MPJPE over a masked set of batched landmark trajectories.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted landmarks of shape (B, T, 21, 3).
    target : torch.Tensor
        Ground-truth landmarks of shape (B, T, 21, 3).
    mask : torch.Tensor
        Valid-timestep mask of shape (B, T).
    idxs : tuple[int, ...], optional
        Landmark indices used for evaluation.

    Returns
    -------
    torch.Tensor
        Scalar PA-MPJPE in millimeters.
    """
    pred_eval = pred[:, :, idxs].contiguous().reshape(-1, len(idxs), 3)
    target_eval = target[:, :, idxs].contiguous().reshape(-1, len(idxs), 3)
    valid = mask.reshape(-1)

    if not valid.any():
        return pred.new_tensor(float("nan"))

    pred_valid = pred_eval[valid]
    target_valid = target_eval[valid]
    aligned = procrustes_align(pred_valid, target_valid)
    return torch.linalg.norm(aligned - target_valid, dim=-1).mean()


class Metric:
    """Compute a dictionary of metrics from predicted and target joint angles."""

    weight: float = 0

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError


class AnglularDerivatives(Metric):
    """Mean absolute value of angular velocity, acceleration, and jerk."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        vel = torch.diff(pred, dim=-1)
        acc = torch.diff(vel, dim=-1)
        jerk = torch.diff(acc, dim=-1)

        mask = mask.unsqueeze(1).expand(-1, NUM_JOINTS, -1)  # BT -> BCT
        mask_vel = self.adjust_mask(mask)
        mask_acc = self.adjust_mask(mask_vel)
        mask_jerk = self.adjust_mask(mask_acc)

        # vel, acc, and jerk are in (radians / sample), so we multiply by the emg sample
        # rate to get (radians / second). Also take absolute value. Skip empty
        # masks (all-IK-failure windows) so the epoch mean stays finite.
        out: dict[str, torch.Tensor] = {}
        if mask_vel.any():
            out[f"{stage}_vel"] = vel[mask_vel].abs().mean() * EMG_SAMPLE_RATE
        if mask_acc.any():
            out[f"{stage}_acc"] = acc[mask_acc].abs().mean() * EMG_SAMPLE_RATE
        if mask_jerk.any():
            out[f"{stage}_jerk"] = jerk[mask_jerk].abs().mean() * EMG_SAMPLE_RATE
        return out

    def adjust_mask(self, mask: torch.tensor):
        # Adjust mask to eliminate boundaries between IK failures
        # Note that this operation reduces the length of time by 1
        return ~F.max_pool1d((~mask).float(), kernel_size=2, stride=1).to(bool)


class AngleMAE(Metric):
    """Angular mean absolute error."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        no_ik_failure: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:
        mask = no_ik_failure.unsqueeze(1).expand(-1, NUM_JOINTS, -1)
        mae = _masked_l1(pred, target, mask)
        return {f"{stage}_mae": mae} if mae is not None else {}


class PerFingerAngleMAE(Metric):
    """Angular mean absolute error for each finger."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        out: dict[str, torch.Tensor] = {}
        for finger in FINGERS:
            err = self.get_error_for_finger(pred, target, mask, finger)
            if err is not None:
                out[f"{stage}_mae_{finger}"] = err
        return out

    @staticmethod
    def get_error_for_finger(pred, target, mask, finger: str):
        idxs = [j.index for j in JOINTS if finger in j.groups]
        mask = mask.unsqueeze(1).expand(-1, len(idxs), -1)
        return _masked_l1(pred[:, idxs], target[:, idxs], mask)


class PDAngleMAE(Metric):
    """Angular mean absolute error grouped by proximal-distal joints."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        out: dict[str, torch.Tensor] = {}
        for group in PD_GROUPS:
            err = self.get_error_for_group(pred, target, mask, group)
            if err is not None:
                out[f"{stage}_mae_{group}"] = err
        return out

    @staticmethod
    def get_error_for_group(pred, target, mask, group: str):
        idxs = [j.index for j in JOINTS if group in j.groups]
        mask = mask.unsqueeze(1).expand(-1, len(idxs), -1)
        return _masked_l1(pred[:, idxs], target[:, idxs], mask)


class LandmarkDistances(Metric):
    """Mean Euclidian error for landmark positions."""

    def __init__(self, downsampling: int = 40):
        self.downsampling = downsampling
        self.hand_model = TorchHandModel(load_default_hand_model())

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: bool,
        stage: str,
    ) -> dict[str, torch.Tensor]:

        if self.hand_model.device != pred.device:
            self.hand_model.to(pred.device)

        # Convert angles to 3D positions
        # We downsample in time to avoid OOM in forward_kinematics
        sl = slice(None, None, self.downsampling)
        pred_pos = forward_kinematics(pred[:, :, sl], self.hand_model)
        target_pos = forward_kinematics(target[:, :, sl], self.hand_model)
        mask_sliced = mask[:, sl]

        # Landmark distances
        lm_idxs = [lm.index for lm in LANDMARKS if lm.name not in NO_MOVEMENT_LANDMARKS]
        landmark_distance = self.get_mean_distance(
            pred_pos, target_pos, mask_sliced, lm_idxs
        )

        # Fingertip distances
        joint_idxs = [lm.index for lm in LANDMARKS if "fingertip" in lm.groups]
        fingertip_distance = self.get_mean_distance(
            pred_pos, target_pos, mask_sliced, joint_idxs
        )

        metrics: dict[str, torch.Tensor] = {}
        if fingertip_distance is not None:
            metrics[f"{stage}_fingertip_distance"] = fingertip_distance
        if landmark_distance is not None:
            metrics[f"{stage}_landmark_distance"] = landmark_distance
        if stage == "test":
            metrics[f"{stage}_pa_mpjpe"] = masked_pa_mpjpe(
                pred_pos,
                target_pos,
                mask_sliced,
            )

        return metrics

    def get_mean_distance(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        idxs: list,
    ):
        """
        Mean distance for given landmark idxs. Inputs are (batch, time, landmark, xyz).
        """
        mask = mask[..., None].expand(-1, -1, len(idxs))
        dist = torch.linalg.norm(pred[:, :, idxs] - target[:, :, idxs], dim=-1)[mask]
        if dist.numel() == 0:
            return None
        return dist.mean()


def get_default_metrics() -> list[Metric]:
    metrics: list[Metric] = [
        AngleMAE(),
        AnglularDerivatives(),
        PerFingerAngleMAE(),
        PDAngleMAE(),
        LandmarkDistances(),
    ]
    # Extend with KINE-Pose metrics if available.  Kept as a soft import
    # so this file has no hard dependency on the KINE module.
    try:
        from emg2pose.kine_metrics import get_kine_metrics  # noqa: WPS433
        metrics.extend(get_kine_metrics())
    except Exception:  # pragma: no cover — metrics module optional.
        pass
    try:
        from emg2pose.neurokine_losses import get_neurokine_metrics  # noqa: WPS433
        metrics.extend(get_neurokine_metrics())
    except Exception:  # pragma: no cover — metrics module optional.
        pass
    return metrics
