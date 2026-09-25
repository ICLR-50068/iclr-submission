# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from emg2pose.constants import EMG_SAMPLE_RATE
from emg2pose.networks import SequentialLSTM

from torch import nn
from torch.nn.functional import interpolate


class BasePoseModule(nn.Module):
    """
    Pose module consisting of a network with a left and right context. Predictions span
    the inputs[left_context : -right_context], and are upsampled to match the sample
    rate of the inputs.
    """

    def __init__(
        self,
        network: nn.Module,
        out_channels: int = 20,
    ):
        super().__init__()
        self.network = network
        self.out_channels = out_channels

        self.left_context = network.left_context
        self.right_context = network.right_context

    def forward(
        self, batch: dict[str, torch.Tensor], provide_initial_pos: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:

        emg = batch["emg"]
        joint_angles = batch["joint_angles"]
        no_ik_failure = batch["no_ik_failure"]

        # Get initial position
        initial_pos = joint_angles[..., self.left_context]
        if not provide_initial_pos:
            initial_pos = torch.zeros_like(initial_pos)

        # Generate prediction (may return temporal mask)
        pred, temporal_mask = self._predict_pose(emg, initial_pos)

        # Slice joint angles to match the span of the predictions
        start = self.left_context
        stop = None if self.right_context == 0 else -self.right_context
        joint_angles = joint_angles[..., slice(start, stop)]
        no_ik_failure = no_ik_failure[..., slice(start, stop)]

        # Match the sample rate of the predictions to that of the joint angles
        # EXCEPT for STFT/CST/ViT Transformers which return dense predictions at window rate
        # (alignment will be done in lightning.py instead)
        try:
            from emg2pose.stft_transformer_arch import STFTTransformer
            from emg2pose.circular_stft_transformer_arch import CyclicSpectralTransformer
            from emg2pose.stft_vit_arch import STFTViT
        except ImportError:
            STFTTransformer = CyclicSpectralTransformer = STFTViT = type(None)
        if isinstance(self.network, (STFTTransformer, CyclicSpectralTransformer, STFTViT)):
            # STFT/CST Transformer: Keep predictions at window rate
            # Shape: (B, num_windows, J) - no interpolation needed
            # Note: no_ik_failure and temporal_mask will be aligned in lightning.py
            pass
        else:
            n_time = joint_angles.shape[-1]
            n_pred = pred.shape[-1]
            keep_rate = bool(getattr(self.network, "keep_encoder_rate", False))
            # Train: downsample labels to T' so backward stays at encoder
            # rate. Eval/test still upsample preds so test_analysis is 2 kHz.
            if keep_rate and self.training and n_pred != n_time:
                joint_angles = interpolate(joint_angles, size=n_pred, mode="linear")
                no_ik_failure = self.align_mask(no_ik_failure, n_pred)
                if temporal_mask is not None and temporal_mask.shape[-1] != n_pred:
                    temporal_mask = self.align_mask(temporal_mask, n_pred)
            else:
                pred = self.align_predictions(pred, n_time)
                no_ik_failure = self.align_mask(no_ik_failure, n_time)
                if temporal_mask is not None:
                    temporal_mask = self.align_mask(temporal_mask, n_time)

        return pred, joint_angles, no_ik_failure, temporal_mask

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        raise NotImplementedError

    def align_predictions(self, pred: torch.Tensor, n_time: int):
        """Temporally resamples predictions to match the length of targets."""
        return interpolate(pred, size=n_time, mode="linear")

    def align_mask(self, mask: torch.Tensor, n_time: int):
        """Temporally resample mask to match the length of targets."""
        # 2D Inputs don't work for interpolate(), so we add a dummy channel dimension
        mask = mask[:, None].to(torch.float32)
        aligned = interpolate(mask, size=n_time, mode="nearest")
        return aligned.squeeze(1).to(torch.bool)


class PoseModule(BasePoseModule):
    """
    Tracks pose by predicting posititions or velocities,
    optionally given the initial state.
    """

    def __init__(self, network: nn.Module, predict_vel: bool = False):
        super().__init__(network)
        self.predict_vel = predict_vel

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        # The compiled training graph is substantially faster, but compiled
        # eval hits a CUDA flash-attention launch limit at validation batch
        # 640. Keep validation on the original eager network.
        compiled = getattr(self, "_compiled_train_network", None)
        network = compiled if self.training and compiled is not None else self.network
        output = network(emg)  # BCT or (BCT, mask)
        
        # Check if network returned mask (for temporal dropout)
        temporal_mask = None
        if isinstance(output, tuple):
            pred, temporal_mask = output
        else:
            pred = output
        
        if self.predict_vel:
            pred = initial_pos[..., None] + torch.cumsum(pred, -1)
        
        return pred, temporal_mask


class StatePoseModule(BasePoseModule):
    """
    Tracks pose by predicting posititions or velocities, optionally given the initial
    state and conditioned on the previous state at each time point.
    """

    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        state_condition: bool = True,
        predict_vel: bool = False,
        rollout_freq: int = 50,
    ):
        super().__init__(network)
        self.decoder = decoder
        self.state_condition = state_condition
        self.predict_vel = predict_vel
        self.rollout_freq = rollout_freq

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):

        features = self.network(emg)  # BCT
        preds = [initial_pos]

        # Resample features to rollout frequency
        seconds = (
            emg.shape[-1] - self.left_context - self.right_context
        ) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)

        # Reset LSTM hidden state
        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()

        for t in range(features.shape[-1]):

            # Prepare decoder inputs
            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)

            # Predict pose
            pred = self.decoder(inputs)
            if self.predict_vel:
                pred = pred + preds[-1]
            preds.append(pred)

        # Remove first pred, because it is the initial_pos (not a network prediction)
        return torch.stack(preds[1:], dim=-1), None


class VEMG2PoseWithInitialState(BasePoseModule):
    """
    Predict pose for num_position_steps steps, then integrate the velocity thereafter.
    """

    def __init__(
        self,
        network: nn.Module,
        decoder: nn.Module,
        num_position_steps: int,
        state_condition: bool = True,
        rollout_freq: int = 50,
    ):
        super().__init__(network)
        self.decoder = decoder
        self.num_position_steps = num_position_steps
        self.state_condition = state_condition
        self.rollout_freq = rollout_freq

    def _predict_pose(self, emg: torch.Tensor, initial_pos: torch.Tensor):
        features = self.network(emg)  # BCT

        # Resample features to rollout frequency
        seconds = (
            emg.shape[-1] - self.left_context - self.right_context
        ) / EMG_SAMPLE_RATE
        n_time = round(seconds * self.rollout_freq)
        features = interpolate(features, n_time, mode="linear", align_corners=True)

        # Reset LSTM hidden state
        if isinstance(self.decoder, SequentialLSTM):
            self.decoder.reset_state()

        # Compute num_position_steps at the new sample rate
        num_position_steps = round(
            self.num_position_steps * (self.rollout_freq / EMG_SAMPLE_RATE)
        )
        preds = [initial_pos]

        for t in range(features.shape[-1]):

            # Prepare decoder inputs
            inputs = features[:, :, t]
            if self.state_condition:
                inputs = torch.concat([inputs, preds[-1]], dim=-1)

            # Predict pose and velocity
            output = self.decoder(inputs)  # BC
            pos, vel = torch.split(output, output.shape[1] // 2, dim=1)

            # Predict pose for the first num_position_steps
            # then integrate velocity thereafter
            pred = pos if t < num_position_steps else preds[-1] + vel
            preds.append(pred)

        # Remove first pred, because it is the initial_pos (not a network prediction)
        return torch.stack(preds[1:], dim=-1), None


class MAEPoseModule(nn.Module):
    """
    Pose module for Masked Autoencoder (MAE) pretraining.
    
    Unlike other pose modules, this one doesn't predict poses directly.
    Instead, it wraps an EMG_MAE network and handles the self-supervised
    reconstruction task.
    
    The MAE learns to reconstruct randomly masked EMG patches, creating
    a foundation model that can later be used for downstream pose prediction.
    """
    
    def __init__(
        self,
        network: nn.Module,  # Should be EMG_MAE instance
        out_channels: int = 20,  # For compatibility, not used in MAE
    ):
        super().__init__()
        self.network = network
        self.out_channels = out_channels
        
        # MAE has no context (processes full windows)
        self.left_context = getattr(network, 'left_context', 0)
        self.right_context = getattr(network, 'right_context', 0)
    
    def forward(
        self, 
        batch: dict[str, torch.Tensor], 
        provide_initial_pos: bool = False  # Ignored for MAE
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for MAE training.
        
        Args:
            batch: dict with 'emg', 'joint_angles', 'no_ik_failure'
            provide_initial_pos: ignored (MAE doesn't use pose conditioning)
        
        Returns:
            pred: reconstructed EMG (B, C, T)
            joint_angles: original joint angles (for compatibility)
            no_ik_failure: mask (for compatibility)
            mae_loss: the MAE reconstruction loss
        """
        emg = batch["emg"]  # (B, C, T)
        joint_angles = batch["joint_angles"]
        no_ik_failure = batch["no_ik_failure"]
        
        # Forward through MAE network
        mae_output = self.network(emg, return_loss=True)
        
        # Extract outputs
        pred_emg = mae_output['pred']  # (B, C, T)
        mae_loss = mae_output['loss']  # scalar
        
        # For compatibility with existing training loop:
        # - pred: We return the reconstructed EMG (not pose!)
        # - joint_angles: Return as-is (unused in MAE loss)
        # - no_ik_failure: Return as-is (unused in MAE loss)
        # - temporal_mask: We'll store the MAE loss here for now
        
        # Note: The actual loss will be computed externally
        # We need to modify the training loop to handle MAE differently
        
        return pred_emg, joint_angles, no_ik_failure, mae_loss
