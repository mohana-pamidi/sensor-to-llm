"""
Loads the frozen SmolLM2-360M-Instruct model and provides helpers to inject
a continuous "sensor" embedding directly into the input embedding sequence,
replacing the <SENSOR> placeholder.
"""

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
SENSOR_PLACEHOLDER = "<SENSOR>"


PROMPT_TEMPLATE = (
    "Classify the activity as walking, walking upstairs, walking downstairs, "
    "sitting, standing, or laying.\n\n"
    f"Sensor context: {SENSOR_PLACEHOLDER}\n\n"
    "Activity:"
)

