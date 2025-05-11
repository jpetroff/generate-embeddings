#!/usr/bin/env python3

# import to reload definition of tqdm
import math
import modules.extend_tqdm


from tqdm.auto import tqdm
import logging
from typing import List
from dataclasses import dataclass, field

from pathlib import Path

from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import Settings

from time import time

from rich.logging import RichHandler

from modules.ingest import IngestPipelineComponent
from modules.arguments import args, set_arg_config
from modules.connectors import defaults_from_config, init_storage_context
from modules.interactive import ask_before_connects, ask_or_get_files, ask_to_confirm
from modules.env import set_env_config

from transforms.llama_settings import update_llama_settings
from transforms.node_parser import node_parser
from transforms.embed_model import update_embed_model

from modules.progress import global_console

@dataclass
class user_config:
    env_file: Path = Path('.default.env')
    ingest_directory: Path = Path('./data')
    ingest_files: List[Path] = field(default_factory=list)
    ignored_files: List[Path] = field(default_factory=list)
    non_interactive_mode: bool = False
    # embed model defaults
    EMBED_MODEL="mxbai-embed-large"
    OLLAMA_URL="http://localhost:11434"
    CHUNK_SIZE=1024
    CHUNK_OVERLAP=20
    NUM_OUTPUT=512
    CONTEXT_WINDOW=8096
    BUFFER_SIZE=96

if __name__ == "__main__":

    print(tqdm) # type: ignore


    global_console.clear(home=True)  #clear screen

    # Read args as first priority
    set_arg_config(config=user_config)

    # ↓
    # Clarify setup details
    ask_before_connects(args, user_config)

    # ↓
    # Read .env config
    env = set_env_config(config=user_config)

    # ↓ 
    # Connect to remote storage
    storage_context = init_storage_context(
        **defaults_from_config(user_config)
    )
    assert storage_context is not None, exit(1)

    # ↓
    # Clarify scope of processing - which files and where to ingest
    ask_or_get_files(args, user_config)
    

    global_console.print(
        f"Using embedding model [yellow]{user_config.EMBED_MODEL}[/]\n"
        f"Collection name [yellow]{user_config.COLLECTION_NAME}[/]\n" # type: ignore
        f"Vector storage [purple]{user_config.QDRANT_URL}[/]\n" # type: ignore
        f"Document storage [purple]{user_config.MONGO_URI}[/]" # type: ignore
    )
    request_confirmation = ask_to_confirm(args=args, config=user_config)
    global_console.print("\n")

    # Set to 'DEBUG' to have extensive logging turned on, even for libraries
    LOG_LEVEL: str = user_config.LOG_LEVEL or "CRITICAL" # type: ignore

    PRETTY_LOG_FORMAT = (
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(name)+25s - %(message)s"
    )
    logging.basicConfig(
        format=PRETTY_LOG_FORMAT,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(rich_tracebacks=True)],
        level=LOG_LEVEL
    )
    logging.captureWarnings(False)

    logger = logging.getLogger('generate-app')

    """
    Main flow    
    -------------------------------------------------------
    """

    update_llama_settings(user_config)
    update_embed_model(user_config)

    ingest_service = IngestPipelineComponent(
        storage_context=storage_context, 
        embed_model=Settings.embed_model,
        transformations=[node_parser, Settings.embed_model]
    )

    try:
        files = [(str(p.name), p) for p in user_config.ingest_files]
        global_console.print(f"[dim]- Starting ingestion for {len(files)} files[/]")
        ingest_service.bulk_ingest(
            files=files,
            buffer_transforms=4
        )
        
    finally:
        del ingest_service

    global_console.print("[dim]- Done[/]")