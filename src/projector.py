"""
projector.py
------------
MLP projector that maps the SensorEncoder's output into the frozen LLM's
token embedding space.

Data flow (called explicitly in testllm.py):

  raw window  (B, 128, 9)
      |  SensorEncoder          [sensor_encoder.py]
      v
  enc_out     (B, encoder_dim=128)
      |  SensorProjector        [this file]
      v
  soft token  (B, llm_hidden_size=960)
      |
  Frozen SmolLM2-360M-Instruct

The projector takes the encoder's OUTPUT as input, not the raw window.
The encoder is instantiated and called separately (see testllm.py).
"""

import torch
import torch.nn as nn


class SensorProjector(nn.Module):
    """
    Two-layer MLP projector:
        encoder_dim  -->  (encoder_dim * 2)  -->  llm_hidden_size

    Takes the OUTPUT of SensorEncoder as input (B, encoder_dim).
    The encoder is NOT owned here — it lives in sensor_encoder.py and
    is called explicitly before this module.

    Parameters
    ----------
    encoder_dim     : output dim of SensorEncoder   (default 128)
    llm_hidden_size : hidden size of the target LLM (default 960)
    """

    def __init__(self, encoder_dim: int = 128, llm_hidden_size: int = 960):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.llm_hidden_size = llm_hidden_size

        # MLP: encoder_dim -> encoder_dim*2 -> llm_hidden_size
        mid = encoder_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, llm_hidden_size),
        )

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        enc_out : (B, encoder_dim)  -- output of SensorEncoder

        Returns
        -------
        (B, llm_hidden_size)  -- soft token embedding ready to splice
                                 into the LLM input sequence
        """
        return self.proj(enc_out)    # (B, llm_hidden_size)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sensor_encoder import SensorEncoder

    ENCODER_DIM = 128
    LLM_HIDDEN  = 960

    encoder = SensorEncoder(in_channels=9, encoder_dim=ENCODER_DIM)
    proj    = SensorProjector(encoder_dim=ENCODER_DIM, llm_hidden_size=LLM_HIDDEN)

    raw_windows = torch.randn(4, 128, 9)        # (B, T, C)
    enc_out     = encoder(raw_windows)          # (B, 128)
    soft_token  = proj(enc_out)                 # (B, 960)

    print(f"raw_windows : {tuple(raw_windows.shape)}")
    print(f"enc_out     : {tuple(enc_out.shape)}")
    print(f"soft_token  : {tuple(soft_token.shape)}")
    assert soft_token.shape == (4, LLM_HIDDEN)

    # Gradient check
    soft_token.sum().backward()
    for name, p in proj.named_parameters():
        assert p.grad is not None, f"No grad for proj.{name}"
    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"No grad for encoder.{name}"
    print("Gradient check passed.")
    print("SensorProjector self-test passed.")
