# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

from collections.abc import KeysView

from dataclasses import dataclass, field, InitVar
from pathlib import Path
from typing import Any, ClassVar

import h5py
import numpy as np
import torch

from emg2pose import transforms
from emg2pose.transforms import Transform
from emg2pose.utils import get_contiguous_ones, get_ik_failures_mask


@dataclass
class Emg2PoseSessionData:
    """A read-only interface to a single emg2pose session file stored in
    HDF5 format.

    ``self.timeseries`` is a `h5py.Dataset` instance with a compound data type
    as in a numpy structured array containing three fields - EMG data from the
    left and right wrists, and their corresponding timestamps.
    The sampling rate of EMG is 2kHz, each EMG device has 16 electrode
    channels, and the signal has been high-pass filtered. Therefore, the fields
    corresponding to left and right EMG are 2D arrays of shape ``(T, 16)`` each
    and ``timestamps`` is a 1D array of length ``T``.

    NOTE: Only the metadata and ground-truth are loaded into memory while the
    EMG data is accesssed directly from disk. When wrapping this interface
    within a PyTorch Dataset, use multiple dataloading workers to mask the
    disk seek and read latencies."""

    HDF5_GROUP: ClassVar[str] = "emg2pose"
    # timeseries keys
    TIMESERIES: ClassVar[str] = "timeseries"
    EMG: ClassVar[str] = "emg"
    JOINT_ANGLES: ClassVar[str] = "joint_angles"
    TIMESTAMPS: ClassVar[str] = "time"
    # metadata keys
    SESSION_NAME: ClassVar[str] = "session"
    SIDE: ClassVar[str] = "side"
    STAGE: ClassVar[str] = "stage"
    START_TIME: ClassVar[str] = "start"
    END_TIME: ClassVar[str] = "end"
    NUM_CHANNELS: ClassVar[str] = "num_channels"
    DATASET_NAME: ClassVar[str] = "dataset"
    USER: ClassVar[str] = "user"
    SAMPLE_RATE: ClassVar[str] = "sample_rate"

    hdf5_path: Path

    def __post_init__(self) -> None:
        self._file = h5py.File(self.hdf5_path, "r")
        emg2pose_group: h5py.Group = self._file[self.HDF5_GROUP]

        # ``timeseries`` is a HDF5 compound Dataset
        self.timeseries: h5py.Dataset = emg2pose_group[self.TIMESERIES]
        assert self.timeseries.dtype.fields is not None
        assert self.EMG in self.timeseries.dtype.fields
        assert self.JOINT_ANGLES in self.timeseries.dtype.fields
        assert self.TIMESTAMPS in self.timeseries.dtype.fields

        # Load the metadata entirely into memory as it's rather small
        self.metadata: dict[str, Any] = {}
        for key, val in emg2pose_group.attrs.items():
            self.metadata[key] = val

    def __enter__(self) -> Emg2PoseSessionData:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._file.close()

    def __len__(self) -> int:
        return len(self.timeseries)

    def __getitem__(self, key: slice) -> np.ndarray:
        return self.timeseries[key]

    def slice(self, start_t: float = -np.inf, end_t: float = np.inf) -> np.ndarray:
        """Load and return a contiguous slice of the timeseries windowed
        by the provided start and end timestamps.

        Args:
            start_t (float): The start time of the window to grab
                (in absolute unix time). Defaults to selecting from the
                beginning of the session. (default: ``-np.inf``).
            end_t (float): The end time of the window to grab
                (in absolute unix time). Defaults to selecting until the
                end of the session. (default: ``np.inf``)
        """
        start_idx, end_idx = self.timestamps.searchsorted([start_t, end_t])
        return self[start_idx:end_idx]

    @property
    def fields(self) -> KeysView[str]:
        """The names of the fields in ``timeseries``."""
        fields: KeysView[str] = self.timeseries.dtype.fields.keys()
        return fields

    @property
    def timestamps(self) -> np.ndarray:
        """EMG timestamps.

        NOTE: This reads the entire sequence of timesetamps from the underlying
        HDF5 file and therefore incurs disk latency. Avoid this in the critical
        path."""
        emg_timestamps = self.timeseries[self.TIMESTAMPS]
        assert (np.diff(emg_timestamps) >= 0).all(), "Not monotonic"
        return emg_timestamps

    @property
    def session_name(self) -> str:
        """Unique name of the session."""
        return self.metadata[self.SESSION_NAME]

    @property
    def user(self) -> str:
        """Unique ID of the user this session corresponds to."""
        return self.metadata[self.USER]

    @property
    def no_ik_failure(self):
        if not hasattr(self, "_no_ik_failure"):
            joint_angles = self.timeseries[self.JOINT_ANGLES]
            self._no_ik_failure = get_ik_failures_mask(joint_angles)
        return self._no_ik_failure

    def __str__(self) -> str:
        return f"{self.__class__.__name__} {self.session_name} ({len(self)} samples)"


@dataclass
class WindowedEmgDataset(torch.utils.data.Dataset):
    """A `torch.utils.data.Dataset` corresponding to an instance of
    `Emg2PoseSessionData` that iterates over EMG windows of configurable
    length and stride.

    Args:
        hdf5_path (str): Path to the session file in hdf5 format.
        window_length (int): Size of each window. Specify None for no windowing
            in which case this will be a dataset of length 1 containing the
            entire session. (default: ``None``)
        stride (int): Stride between consecutive windows. Specify None to set
            this to window_length, in which case there will be no overlap
            between consecutive windows. (default: ``window_length``)
        padding (tuple[int, int]): Left and right contextual padding for
            windows in terms of number of raw EMG samples.
        jitter (bool): If True, randomly jitter the offset of each window.
            Use this for training time variability. (default: ``False``)
        transform (Callable): A composed sequence of transforms that takes
            a window/slice of `Emg2PoseSessionData` in the form of a numpy
            structured array and returns a `torch.Tensor` instance.
            (default: ``emg2pose.transforms.ExtractToTensor()``)
        skip_ik_failures (bool): If True, produces only windows with no IK failures.
        random_sampled_windows (bool): If True, randomly sample windows from the session
            each epoch instead of using fixed stride-based windows. (default: ``False``)
        windows_per_session (int): Number of random windows to sample per session.
            Only used when random_sampled_windows=True. (default: ``None``)
    """

    hdf5_path: Path
    window_length: InitVar[int | None] = 10_000
    stride: InitVar[int | None] = None
    padding: InitVar[tuple[int, int]] = (0, 0)
    jitter: bool = False
    transform: Transform[np.ndarray, torch.Tensor] = field(
        default_factory=transforms.ExtractToTensor
    )
    skip_ik_failures: bool = False
    random_sampled_windows: bool = False
    windows_per_session: int | None = None

    def __post_init__(
        self,
        window_length: int | None,
        stride: int | None,
        padding: tuple[int, int],
    ) -> None:
        # Get session length and close file immediately to prevent handle leak
        self.session_length = len(self.session)
        
        self.window_length = (
            window_length if window_length is not None else self.session_length
        )
        self.stride = stride if stride is not None else self.window_length
        assert self.window_length > 0 and self.stride > 0

        (self.left_padding, self.right_padding) = padding
        assert self.left_padding >= 0 and self.right_padding >= 0

        if window_length is None and self.skip_ik_failures:
            raise ValueError(
                "skip_ik_failures=True requires window_length to be specified."
            )
        
        if self.random_sampled_windows and self.windows_per_session is None:
            raise ValueError(
                "random_sampled_windows=True requires windows_per_session to be specified."
            )

        # Precompute (start, end) windows based on skip_ik_failures setting
        # For random sampling, we don't precompute - we sample on-the-fly
        if not self.random_sampled_windows:
            self.windows: list[tuple[int, int]] = self.precompute_windows()
        else:
            # For random sampling, we still need to cache the blocks
            # Access blocks to trigger caching, then close the session
            _ = self.blocks
            self.windows = None
            self.close_session()

    def __len__(self) -> int:
        if self.random_sampled_windows:
            return self.windows_per_session
        return sum(self._get_block_len(b) for b in self.blocks)

    def _get_block_len(self, block: tuple[float, float]) -> int:
        """Get number of windows in a (start, end) time block."""
        n = (block[1] - block[0] - self.window_length) // self.stride + 1
        return max(0, int(n))

    @property
    def session(self):
        if not hasattr(self, "_session") or self._session._file.id.valid == 0:
            self._session = Emg2PoseSessionData(self.hdf5_path)
        return self._session
    
    def close_session(self):
        """Close the HDF5 file handle if it's open."""
        if hasattr(self, "_session"):
            self._session._file.close()
            del self._session

    @property
    def blocks(self) -> list[tuple[int, int]]:
        """List of (start, end) times to be included in the dataset."""

        if not hasattr(self, "_blocks"):

            # Include all time
            if not self.skip_ik_failures:
                self._blocks = [(0, self.session_length)]

            # Include only time without IK failures
            else:
                blocks = get_contiguous_ones(self.session.no_ik_failure)
                blocks = [
                    (t0, t1 - 1)
                    for (t0, t1) in blocks
                    if (t1 - t0) >= self.window_length
                ]
                self._blocks = blocks
            
            # Close the session file to prevent file handle leak
            self.close_session()

        return self._blocks

    def precompute_windows(self) -> list[tuple[int, int]]:
        """For each dataset idx, precompute the EMG start and times."""
        windows = []
        cumsum = np.cumsum([0] + [self._get_block_len(b) for b in self.blocks])

        for idx in range(len(self)):
            block_idx = np.searchsorted(cumsum, idx, "right") - 1
            start_idx, end_idx = self.blocks[block_idx]
            relative_idx = idx - cumsum[block_idx]
            windows.append((start_idx + relative_idx * self.stride, end_idx))

        # Close the session file after precomputing windows to free file handles
        self.close_session()
        
        return windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:

        # Random sampling mode: sample a random valid window position
        if self.random_sampled_windows:
            # Get valid blocks (regions without IK failures if skip_ik_failures=True)
            valid_blocks = self.blocks
            
            # Filter blocks to only those large enough for a complete window
            valid_blocks = [(s, e) for s, e in valid_blocks if (e - s) >= self.window_length]
            
            # Handle edge case: no valid blocks in this session
            if len(valid_blocks) == 0:
                # Return a dummy sample with zeros - will be filtered by no_ik_failure mask
                dummy_emg = torch.zeros(16, self.window_length)
                dummy_angles = torch.zeros(20, self.window_length)
                dummy_mask = torch.zeros(self.window_length, dtype=torch.bool)
                return {
                    "emg": dummy_emg,
                    "joint_angles": dummy_angles,
                    "no_ik_failure": dummy_mask,
                    "window_start_idx": 0,
                    "window_end_idx": self.window_length,
                }
            
            # Randomly select a block
            block_idx = np.random.randint(0, len(valid_blocks))
            start_idx, end_idx = valid_blocks[block_idx]
            
            # Randomly sample window start position within this block
            max_offset = end_idx - self.window_length
            offset = np.random.randint(start_idx, max_offset + 1)
        else:
            # Fixed stride mode: use precomputed windows
            offset, end_idx = self.windows[idx]
            leftover = end_idx - (offset + self.window_length)
            if leftover < 0:
                raise IndexError(f"Index {idx} out of bounds, leftover {leftover}")
            if leftover > 0 and self.jitter:
                offset += np.random.randint(0, min(self.stride, leftover))

        # Expand window to include contextual padding and fetch
        window_start = max(offset - self.left_padding, 0)
        window_end = offset + self.window_length + self.right_padding
        
        # Open session file, read data, and immediately close it
        with Emg2PoseSessionData(self.hdf5_path) as session:
            window = session[window_start:window_end]
            no_ik_failure_slice = session.no_ik_failure[window_start:window_end]

        # Extract EMG tensor corresponding to the window
        emg = self.transform(window)
        assert torch.is_tensor(emg)

        # Extract joint angle labels
        joint_angles = window[Emg2PoseSessionData.JOINT_ANGLES]
        joint_angles = torch.as_tensor(joint_angles)

        # Mask of samples without IK failures
        no_ik_failure = torch.as_tensor(no_ik_failure_slice)
        
        return {
            "emg": emg.T,  # CT
            "joint_angles": joint_angles.T,  # CT
            "no_ik_failure": no_ik_failure,  # T
            "window_start_idx": window_start,
            "window_end_idx": window_end,
        }
    