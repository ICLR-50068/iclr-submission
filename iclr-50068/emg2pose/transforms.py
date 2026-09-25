# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]


@dataclass
class ExtractToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    field: str = "emg"

    def __call__(self, data: np.ndarray | torch.Tensor | dict) -> torch.Tensor:
        if torch.is_tensor(data):
            return data
        if isinstance(data, dict):
            return torch.as_tensor(data[self.field])
        return torch.as_tensor(data[self.field])


@dataclass
class RotationAugmentation:
    """Rotate EMG along the channel dimension by a random integer."""

    channel_dim: int = -1

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        rotation = np.random.choice([-1, 0, 1])
        return torch.roll(data, rotation, dims=self.channel_dim)


@dataclass
class FixedChannelRotation:
    """Rotate EMG along the channel dimension by a fixed amount.
    
    Args:
        rotation (int): Number of positions to rotate channels. 
                       0 means no rotation (identity transform).
    """
    
    rotation: int = 0
    
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.rotation == 0:
            return data
        return torch.roll(data, self.rotation, dims=-1)


@dataclass
class AmplitudeJitter:
    """Per-channel log-normal amplitude jitter.

    gain = exp(N(0, std)) — strictly positive, median 1.0. The same gain is
    applied to every time step of a given (window, channel), matching the
    cs_tds_ct training-time augmentation. Set std<=0 for identity.

    Operates on a single window of shape (T, C) (transform-time convention).
    """

    std: float = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return data
        C = data.shape[-1]
        log_gain = torch.randn(C, dtype=data.dtype, device=data.device) * self.std
        gain = torch.exp(log_gain)
        return data * gain


@dataclass
class ChannelDropout:
    """Independently zero entire channels with probability p, broadcast over T.

    Each channel is kept with probability (1 - p); the mask is sampled once
    per window and applied to every time step of that channel. Mirrors the
    cs_tds_ct training-time channel dropout.
    """

    p: float = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.p <= 0:
            return data
        C = data.shape[-1]
        keep = (torch.rand(C, device=data.device) >= self.p).to(data.dtype)
        return data * keep


@dataclass
class RandomChannelMask:
    """Zero entire channels with probability mask_prob (EMGFormer aug_best)."""

    mask_prob: float = 0.0
    min_masked: int = 0
    max_masked: int | None = None
    channel_dim: int = -1
    mask_value: float = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.mask_prob <= 0.0 and self.min_masked == 0:
            return data
        n_channels = data.shape[self.channel_dim]
        mask = np.random.rand(n_channels) < self.mask_prob
        mask_indices = np.flatnonzero(mask)
        if self.max_masked is not None and mask_indices.size > self.max_masked:
            mask_indices = np.random.choice(
                mask_indices, size=self.max_masked, replace=False
            )
        if mask_indices.size < self.min_masked:
            available = np.setdiff1d(np.arange(n_channels), mask_indices)
            if available.size > 0:
                need = min(self.min_masked - mask_indices.size, available.size)
                extra = np.random.choice(available, size=need, replace=False)
                mask_indices = np.concatenate([mask_indices, extra], axis=0)
        if mask_indices.size == 0:
            return data
        masked = data.clone()
        index = torch.as_tensor(mask_indices, dtype=torch.long, device=masked.device)
        return masked.index_fill(self.channel_dim, index, self.mask_value)


@dataclass
class RandomTimeMask:
    """Zero contiguous time spans (EMGFormer aug_best; num_masks=0 is a no-op)."""

    max_mask_size: int
    num_masks: int = 1
    min_mask_size: int = 0
    time_dim: int = 0
    mask_value: float = 0.0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.max_mask_size == 0 or self.num_masks == 0:
            return data
        n_time = data.shape[self.time_dim]
        masked = data.clone()
        for _ in range(self.num_masks):
            if self.max_mask_size == self.min_mask_size:
                mask_size = self.max_mask_size
            else:
                mask_size = np.random.randint(
                    self.min_mask_size, self.max_mask_size + 1
                )
            mask_size = min(mask_size, n_time)
            if mask_size == 0:
                continue
            start = np.random.randint(0, n_time - mask_size + 1)
            index = torch.arange(
                start, start + mask_size, device=masked.device, dtype=torch.long
            )
            masked.index_fill_(self.time_dim, index, self.mask_value)
        return masked


@dataclass
class RandomFrequencyMask:
    """Zero contiguous rFFT bands along time (EMGFormer aug_best)."""

    max_mask_size: int
    num_masks: int = 1
    min_mask_size: int = 0
    time_dim: int = 0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.max_mask_size == 0 or self.num_masks == 0:
            return data
        n_time = data.shape[self.time_dim]
        spec = torch.fft.rfft(data, dim=self.time_dim)
        n_freq = spec.shape[self.time_dim]
        for _ in range(self.num_masks):
            if self.max_mask_size == self.min_mask_size:
                mask_size = self.max_mask_size
            else:
                mask_size = np.random.randint(
                    self.min_mask_size, self.max_mask_size + 1
                )
            mask_size = min(mask_size, n_freq)
            if mask_size == 0:
                continue
            start = np.random.randint(0, n_freq - mask_size + 1)
            index = torch.arange(
                start, start + mask_size, device=spec.device, dtype=torch.long
            )
            spec.index_fill_(self.time_dim, index, 0)
        return torch.fft.irfft(spec, n=n_time, dim=self.time_dim)


@dataclass
class RandomGaussianNoise:
    """SNR-bounded Gaussian noise (EMGFormer aug_best)."""

    min_snr_db: float = 25.0
    max_snr_db: float = 35.0
    apply_prob: float = 0.5
    time_dim: int = 0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if self.apply_prob == 0.0:
            return data
        if torch.rand((), device=data.device).item() > self.apply_prob:
            return data
        signal_power = data.pow(2).mean()
        if signal_power.item() == 0.0:
            return data
        snr = torch.empty((), device=data.device).uniform_(
            self.min_snr_db, self.max_snr_db
        )
        noise_std = (signal_power / (10 ** (snr / 10))).sqrt()
        return data + torch.randn_like(data) * noise_std


@dataclass
class ChannelDownsampling:
    """Downsample number of emg channels."""

    downsampling: int = 2

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        return data[:, :: self.downsampling]


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data
