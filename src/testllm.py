"""
testllm.py
----------
End-to-end pipeline test:

  raw sensor window (B, 128, 9)
      |
  SensorEncoder  [imported directly]     (B, 128)
      |
  SensorProjector  [via proejctor.py]   (B, 960)
      |  (one soft token spliced into prompt)
  Frozen SmolLM2-360M-Instruct          (B, 960)
      |
  final hidden state at "Activity:" position

Only the SensorEncoder + SensorProjector weights are trainable.
Gradients flow *through* the frozen LLM back to those modules.
"""

import sys
import os

# Allow `from sensor_encoder import ...` and `from projector import ...`
# when running this file directly from the project root or from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sensor_encoder import SensorEncoder
from projector import SensorProjector

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

SENSOR_PLACEHOLDER = "<SENSOR>"

PROMPT_TEXT = (
    "Classify the activity as walking, walking upstairs, walking downstairs, "
    "sitting, standing, or laying.\n\n"
    f"Sensor context: {SENSOR_PLACEHOLDER}\n\n"
    "Activity:"
)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Load in float16 to save memory (especially critical for 2GB GPUs like MX550)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to(device)

# Freeze
for p in model.parameters():
    p.requires_grad = False
model.eval()
# NOTE: gradients must still flow *through* the frozen model back to the
# projector, so we never wrap forward() in torch.no_grad().

# Pull hidden_size from the loaded model config so projector stays in sync
llm_hidden_size = model.config.hidden_size   # 960 for SmolLM2-360M-Instruct

# Trainable modules — instantiated separately, called in sequence
encoder = SensorEncoder(in_channels=9, encoder_dim=128).to(device)
projector = SensorProjector(
    encoder_dim=128,
    llm_hidden_size=llm_hidden_size,   # read from model.config, not hard-coded
).to(device)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def build_inputs_embeds(sensor_window: torch.Tensor):
    """
    Run the full encoder -> projector -> embed-splice pipeline for one batch.

    Parameters
    ----------
    sensor_window : (B, 128, 9)  -- raw inertial signal windows

    Returns
    -------
    inputs_embeds  : (B, seq_len, llm_hidden_size)
    attention_mask : (B, seq_len)
    sensor_pos     : int  -- token index where the sensor soft-token sits
    """
    B = sensor_window.shape[0]
    model_dtype = next(model.parameters()).dtype

    # 1. Encode: raw window -> encoder embedding
    enc_out = encoder(sensor_window.to(device))           # (B, encoder_dim=128)

    # 2. Project: encoder embedding -> LLM token space
    sensor_embed = projector(enc_out)                     # (B, llm_hidden_size)
    sensor_embed = sensor_embed.to(dtype=model_dtype)     # match LLM dtype
    sensor_embed = sensor_embed.unsqueeze(1)              # (B, 1, llm_hidden_size)

    # 3. Build the text halves of the prompt
    messages = [{"role": "user", "content": PROMPT_TEXT}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    before_text, after_text = input_text.split(SENSOR_PLACEHOLDER)

    before_ids = tokenizer(before_text, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)
    after_ids  = tokenizer(after_text,  return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)

    embed_layer = model.get_input_embeddings()           # frozen nn.Embedding
    before_embeds = embed_layer(before_ids)              # (1, n_before, 960)
    after_embeds  = embed_layer(after_ids)               # (1, n_after,  960)

    # Expand text embeddings to match batch size B
    before_embeds = before_embeds.expand(B, -1, -1)     # (B, n_before, 960)
    after_embeds  = after_embeds.expand(B, -1, -1)      # (B, n_after,  960)

    # 4. Splice: [before | sensor_soft_token | after] 
    inputs_embeds = torch.cat([before_embeds, sensor_embed, after_embeds], dim=1)
    seq_len = inputs_embeds.shape[1]
    attention_mask = torch.ones((B, seq_len), dtype=torch.long, device=device)

    sensor_pos = before_embeds.shape[1]
    return inputs_embeds, attention_mask, sensor_pos


def forward(sensor_window: torch.Tensor):
    """
    Full forward pass: sensor window -> LLM final hidden state.

    Returns the hidden state at the LAST token position (after "Activity:"),
    which is what the classification head consumes.

    Parameters
    ----------
    sensor_window : (B, 128, 9)

    Returns
    -------
    final_hidden : (B, llm_hidden_size)
    sensor_pos   : int
    """
    # sensor window goes into building the embeddings
    inputs_embeds, attention_mask, sensor_pos = build_inputs_embeds(sensor_window)

    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    last_hidden_state = outputs.hidden_states[-1]        # (B, seq_len, 960)
    final_token_hidden = last_hidden_state[:, -1, :]     # (B, 960)
    return final_token_hidden, sensor_pos


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"LLM  : {checkpoint}  on {device}")
    print(f"LLM hidden_size  : {llm_hidden_size}")
    print(f"Encoder out_dim  : {encoder.encoder_dim}")
    print(f"Projector in_dim : {projector.encoder_dim}  out_dim: {projector.llm_hidden_size}")
    print()

    # Simulate a batch of 2 raw sensor windows - not zeros so random 
    dummy_windows = torch.randn(2, 128, 9, device=device) 

    final_hidden, sensor_pos = forward(dummy_windows)
    print(f"Sensor token position : {sensor_pos}")
    print(f"Final hidden shape    : {tuple(final_hidden.shape)}")   # (2, 960)

    # Gradient check: grads must reach both the projector AND the encoder
    loss = final_hidden.sum()
    loss.backward()

    for name, p in projector.named_parameters():
        assert p.grad is not None, f"No gradient for projector.{name}"
    for name, p in encoder.named_parameters():
        assert p.grad is not None, f"No gradient for encoder.{name}"
    print("Gradient check passed — grads reach encoder + projector weights.")