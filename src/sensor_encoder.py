"""
sensor_encoder.py
-----------------
1D-CNN encoder that maps a raw inertial signal window into a fixed-size
embedding vector understood by the projector.

Input  : (B, 128, 9)   -- B windows, 128 timesteps, 9 sensor channels
Output : (B, encoder_dim)  -- default encoder_dim = 128
"""

import torch
import torch.nn as nn


class SensorEncoder(nn.Module):
    """
    Lightweight 1-D convolutional encoder for multi-channel inertial signals.

    Architecture
    ------------
    Three stacked Conv1d blocks, each followed by BatchNorm + GELU + MaxPool,
    that progressively downsample the time axis while expanding the channel
    depth. A global-average-pool at the end collapses the time dimension so the
    output is a single fixed-length vector per window.

    Shapes (default encoder_dim=128)
    ---------------------------------
        Input  : (B, 128, 9)    -- (batch, time, channels)
        After transpose to Conv1d convention: (B, 9, 128)
        Block 1: (B, 64,  64)   -- kernel=7, pool=2
        Block 2: (B, 128, 32)   -- kernel=5, pool=2
        Block 3: (B, 128, 16)   -- kernel=3, pool=2
        GlobalAvgPool: (B, 128) == (B, encoder_dim)
    """

    def __init__(self, in_channels: int = 9, encoder_dim: int = 128):
        super().__init__()
        self.encoder_dim = encoder_dim

        self.blocks = nn.Sequential(
            # Block 1 ---------------------------------------------------------
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),          # 128 -> 64

            # Block 2 ---------------------------------------------------------
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),          # 64  -> 32

            # Block 3 ---------------------------------------------------------
            nn.Conv1d(128, encoder_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(encoder_dim),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),          # 32  -> 16
        )

        # Collapse the remaining time axis into a single vector
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C)  -- e.g. (B, 128, 9)

        Returns
        -------
        (B, encoder_dim)
        """
        # Conv1d expects (B, C, T)
        x = x.permute(0, 2, 1)          # (B, 9, 128)
        x = self.blocks(x)               # (B, encoder_dim, T')
        x = self.global_avg_pool(x)      # (B, encoder_dim, 1)
        x = x.squeeze(-1)               # (B, encoder_dim)
        return x


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    enc = SensorEncoder(in_channels=9, encoder_dim=128)
    dummy = torch.randn(4, 128, 9)          # batch of 4 windows
    out = enc(dummy)
    print(f"SensorEncoder  input : {tuple(dummy.shape)}")
    print(f"SensorEncoder  output: {tuple(out.shape)}")   # (4, 128)
    assert out.shape == (4, 128), f"Unexpected shape: {out.shape}"
    print("SensorEncoder self-test passed.")
