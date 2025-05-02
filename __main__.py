#!/usr/bin/env python3

import argparse
from email.policy import default
import json
import logging
from random import choice
from sys import stdout
from typing import Any
from dotenv import dotenv_values

from pathlib import Path

from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceWindowNodeParser, SemanticSplitterNodeParser, TokenTextSplitter
from llama_index.embeddings.ollama import (
    OllamaEmbedding
)
from llama_index.core import Settings
from llama_index.core.storage.index_store.simple_index_store import SimpleIndexStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.indices.vector_store import VectorStoreIndex

from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client

import inquirer
from rich import print
from rich.pretty import pprint
from rich.console import Console
from rich.live import Live
from rich.status import Status
from rich.logging import RichHandler

from ingest import SimpleIngestComponent, ParallelizedIngestComponent
from callback import event_callback

class user_config:
    env_file = '.default.env'
    ingest_directory = Path('./data')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="__main__.py")
    parser.add_argument(
        "--source", '-s',
        nargs="*",
        help="Folder to ingest",
        default=None
    )
    parser.add_argument(
        "--ignored", "-i",
        nargs="*",
        help="List of files/directories to ignore",
        default=[],
    )
    parser.add_argument(
        "--config", '-c',
        nargs="*",
        help="Specific embedding configuration to run",
        default=None,
    )


    args = parser.parse_args()
    interactive_mode: bool = (args.source is None) and (args.config is None)

    console = Console(log_time=False)
    console.clear(home=True)  #clear screen

    """
    Interactive mode is skipped if enough args are supplied
    -------------------------------------------------------
    """
    if args.config is not None:
        user_config.env_file = args.config

    else:
        user_config.env_file = inquirer.list_input(
            message='Choose configuration', 
            choices=[
                ('default', '.default.env'),
                ('custom (modify .env)', '.env')
            ],
            default=[ user_config.env_file ]
        )
    
    if args.source is not None:
        user_config.ingest_directory = Path(args.source).resolve()
    
    else:
        directory_input: str = inquirer.list_input(
            message='Directory to ingest',
            choices=[
                (f"./{user_config.ingest_directory} — should contain all documents", str(user_config.ingest_directory) ),
                ('./append — only new items to be added to vector storage', './append'),
                ('specify your own path to ingest directory', None)
                ],
            default=[ str(user_config.ingest_directory) ],
        )

        if not directory_input:
            directory_input = inquirer.path(
                message="Directory path",
                exists=True, 
                path_type=inquirer.Path.DIRECTORY
            )

        user_config.ingest_directory = Path(directory_input).resolve()

    env = dotenv_values( user_config.env_file )

    print(f"Using embedding model [purple]{env['EMBED_MODEL']}[/] at [yellow]{env['OLLAMA_URL']}[/]")
    print(f"Writing to vector collection [purple]{env['QDRANT_COLLECTION']}[/] at [yellow]{env['QDRANT_URL']}[/]")
    if interactive_mode:
        confirm_collection = inquirer.confirm(
            message="Confirm?",
            default=True
        )
        confirm_collection or exit(0)

    """
    Main flow    
    -------------------------------------------------------
    """

    # Set to 'DEBUG' to have extensive logging turned on, even for libraries
    LOG_LEVEL: str = env['LOG_LEVEL'] or "CRITICAL"

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
    Connect to storage    
    -------------------------------------------------------
    """
    try:
        print("\n")
        _connect_qdrant_status = Status(
            status=f"Connecting to Qdrant [bold]{env['QDRANT_URL']}[/]",
            console=console
        )
        _connect_qdrant_status.start()
        client = qdrant_client.QdrantClient(
            # you can use :memory: mode for fast and light-weight experiments,
            # it does not require to have Qdrant deployed anywhere
            # but requires qdrant-client >= 1.1.1
            # location=":memory:"
            # otherwise set Qdrant instance address with:
            url=env['QDRANT_URL'] or ''
            # otherwise set Qdrant instance with host and port:
            # host="localhost",
            # port=6333
            # set API KEY for Qdrant Cloud
            # api_key="<qdrant-api-key>",
        )
        _connect_qdrant_status.stop()
        print(f"[[green]✓[/]] Connected to qdrant\n")
    except:
        exit(1)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=env['QDRANT_COLLECTION'] or '',
        index_doc_id=True
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=env['EMBED_MODEL'] or '', 
        base_url=env['OLLAMA_URL'] or ''
    )
    Settings.chunk_size = int(env['CHUNK_SIZE'] or 1024)
    Settings.chunk_overlap = int(env['CHUNK_OVERLAP'] or 20)
    Settings.num_output = int(env['NUM_OUTPUT'] or 512)
    Settings.context_window = int(env['CONTEXT_WINDOW'] or 8096)

    # # node_parser = SentenceWindowNodeParser.from_defaults()
    # # node_parser = SemanticSplitterNodeParser.from_defaults(
    # #     embed_model=Settings.embed_model,
    # #     buffer_size=96,
    # #     include_metadata=True
    # #     )

    node_parser = TokenTextSplitter.from_defaults(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap
        )
    
    vector_store_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=Settings.embed_model,
        store_nodes_override=True
        )
    vector_store_index.refresh()
    print(vector_store_index.ref_doc_info)
    
    try:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir='./store'
        )
    except:
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )


    # file_index_source = open('./store/file_index.json', 'r+')
    # file_index: Any = {}
    # file_index = json.loads(file_index_source.read())
    # file_index_source.close()
    # # try:
    # #     file_index = json.loads(file_index_source.read())
    # # except Exception:
    # #     print(str(Exception))
    # #     print("Initialized new index...")



    event_callback.__kwdefaults__ = {
        'qdrant_client': client,
        'qdrant_collection': env['QDRANT_COLLECTION']
    }

    assert storage_context is not None, print("[[red]×[/]] Failed to initialise storage context")
    ingest_service = ParallelizedIngestComponent(
        storage_context=storage_context, 
        embed_model=Settings.embed_model,
        transformations=[node_parser, Settings.embed_model],
        count_workers=8,
        callback=event_callback
        )
    try:
        directory_reader = SimpleDirectoryReader(input_dir=user_config.ingest_directory)
        files_to_ingest = directory_reader.input_files
        file_list = [f.name for f in files_to_ingest]
        inquirer.checkbox(
            message="Choose file to include or exclude",
            choices=file_list,
            default=file_list
            )

        documents = ingest_service.bulk_ingest(
            files=[(str(p.name), Path(p)) for p in files_to_ingest],
            callback=event_callback
            )
        
    finally:
        pass
        # del ingest_service
    
    #     file_index_source = open('./store/file_index.json', 'w+')
    #     file_index_source.write(json.dumps(file_index, indent=4))
    #     file_index_source.close()
    