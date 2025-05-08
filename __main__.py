#!/usr/bin/env python3

import extend_tqdm
from tqdm.auto import tqdm
import logging
from typing import List
from dataclasses import dataclass, field

from pathlib import Path

from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import Settings

import inquirer
from rich import print
from rich.console import Console
from rich.logging import RichHandler

from ingest import IngestPipelineComponent
from arguments import args, set_arg_config
from connectors import defaults_from_config, init_storage_context
from interactive import ask_before_connects, ask_or_get_files
from env import set_env_config

from transforms.llama_settings import update_llama_settings
from transforms.node_parser import node_parser
from transforms.embed_model import update_embed_model

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

    console = Console(log_time=False)
    console.clear(home=True)  #clear screen

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
    

    # print(f"Using embedding model [purple]{env['EMBED_MODEL']}[/] at [yellow]{env['OLLAMA_URL']}[/]")
    # print(f"Writing to vector collection [purple]{env['QDRANT_COLLECTION']}[/] at [yellow]{env['QDRANT_URL']}[/]")
    if not user_config.non_interactive_mode:
        confirm_collection = inquirer.confirm(
            message="Confirm?",
            default=True
        )
        confirm_collection or exit(0)  # type: ignore

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
        ingest_service.bulk_ingest(
            files=[(str(p.name), p) for p in user_config.ingest_files]
        )
        
    finally:
        del ingest_service

    print("Done")