"""
Loads the frozen SmolLM2-360M-Instruct model and provides helpers to inject
a continuous "sensor" embedding directly into the input embedding sequence,
replacing the <SENSOR> placeholder.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
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
# Force float32: BFloat16 is the model default but causes dtype mismatches
# on CPU when mixed with float32 sensor embeddings from the encoder/projector.
model = AutoModelForCausalLM.from_pretrained(
    checkpoint,
    trust_remote_code=True,
    torch_dtype=torch.float32,
).to(device)
 
# Freeze
for p in model.parameters():
    p.requires_grad = False
model.eval()
# NOTE: we still need gradients to FLOW THROUGH the frozen model into the
# projector, so we never wrap the forward pass itself in torch.no_grad().
 
hidden_size = model.config.hidden_size
 
 
def build_inputs_embeds(sensor_embed: torch.Tensor):
    """
    Build the full inputs_embeds sequence for one example by:
      1. Formatting the prompt with the chat template (as a user message).
      2. Splitting the formatted text at the <SENSOR> placeholder.
      3. Tokenizing both chunks and looking up their frozen embeddings.
      4. Splicing the projected sensor_embed in as a single soft token
         between the two chunks.
 
    sensor_embed: tensor of shape (hidden_size,) or (1, hidden_size),
                  with grad_fn tracing back to the sensor encoder/projector.
 
    Returns:
        inputs_embeds:  (1, seq_len, hidden_size)
        attention_mask: (1, seq_len)
        sensor_pos:     index of the sensor token (for debugging)
    """
    if sensor_embed.dim() == 1:
        sensor_embed = sensor_embed.unsqueeze(0)  # (1, hidden)
    # Coerce to the model's dtype (float32) so linear layers don't mismatch
    model_dtype = next(model.parameters()).dtype
    sensor_embed = sensor_embed.to(device=device, dtype=model_dtype)
 
    messages = [{"role": "user", "content": PROMPT_TEXT}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
 
    before_text, after_text = input_text.split(SENSOR_PLACEHOLDER)
 
    before_ids = tokenizer(before_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    after_ids = tokenizer(after_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
 
    embed_layer = model.get_input_embeddings()  # frozen nn.Embedding
    before_embeds = embed_layer(before_ids)  # (1, n_before, hidden)
    after_embeds = embed_layer(after_ids)    # (1, n_after, hidden)
 
    sensor_embed = sensor_embed.unsqueeze(1)  # (1, 1, hidden) -> one soft token
 
    inputs_embeds = torch.cat([before_embeds, sensor_embed, after_embeds], dim=1)
    seq_len = inputs_embeds.shape[1]
    attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
 
    sensor_pos = before_embeds.shape[1]
    return inputs_embeds, attention_mask, sensor_pos
 
 
def forward(sensor_embed: torch.Tensor):
    """
    Simple forward pass: build the spliced input sequence, run it through
    the frozen LLM, and return the hidden state at the LAST position
    (i.e. right after "Activity:"). This is what the trainable
    classification head consumes -- no generation involved.
    """
    inputs_embeds, attention_mask, sensor_pos = build_inputs_embeds(sensor_embed)
 
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    last_hidden_state = outputs.hidden_states[-1]      # (1, seq_len, hidden)
    final_token_hidden = last_hidden_state[:, -1, :]    # (1, hidden)
    return final_token_hidden, sensor_pos
 
 
if __name__ == "__main__":
    print(f"Loaded {checkpoint} on {device}. hidden_size={hidden_size}")
 
    # Dummy stand-in for a real sensor_encoder -> projector output.
    model_dtype = next(model.parameters()).dtype
    dummy_sensor_embed = torch.randn(hidden_size, device=device, dtype=model_dtype, requires_grad=True)
 
    final_hidden, sensor_pos = forward(dummy_sensor_embed)
    print(f"Sensor token position: {sensor_pos}")
    print(f"Final hidden state shape: {tuple(final_hidden.shape)}")
 
    # Gradient check: confirm the frozen backbone is differentiable
    # end-to-end so gradients reach the (eventual) projector output.
    loss = final_hidden.sum()
    loss.backward()
    assert dummy_sensor_embed.grad is not None, "Gradient did not reach sensor embedding!"
    print("Gradient check passed:", dummy_sensor_embed.grad.shape)