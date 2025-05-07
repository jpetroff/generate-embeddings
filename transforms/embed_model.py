from llama_index.core import Settings
from typing import Any

from llama_index.embeddings.ollama import (
    OllamaEmbedding
)

def update_embed_model(config: Any):
    Settings.embed_model = OllamaEmbedding(
        model_name=config.EMBED_MODEL,
        base_url=config.OLLAMA_URL,
        callback_manager=Settings.callback_manager
    )