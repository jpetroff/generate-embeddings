from llama_index.core import Settings
from typing import Any

def update_llama_settings(config: Any):
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.chunk_overlap = config.CHUNK_OVERLAP
    Settings.num_output = config.NUM_OUTPUT
    Settings.context_window = config.CONTEXT_WINDOW