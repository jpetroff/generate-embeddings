from typing import (
    Optional, Dict, Any
)
from llama_index.core import Settings
from llama_index.core.storage.storage_context import StorageContext

from pathlib import Path
import qdrant_client

from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore

def defaults_from_config(config: Any) -> Dict[str, Any]:

    vector_db_client_defaults = {
        'url': config.QDRANT_URL
    }

    vector_storage_defaults = {
        'collection_name': config.COLLECTION_NAME,
        'index_doc_id': True
    }

    document_storage_defaults = {
        'uri': config.MONGO_URI,
        'db_name': config.MONGO_DB,
        'namespace': config.COLLECTION_NAME
    }

    storage_context_defaults = {
    }

    return {
        'vector_storage_defaults': vector_storage_defaults,
        'document_storage_defaults': document_storage_defaults,
        'storage_context_defaults': storage_context_defaults,
        'vector_db_client_defaults': vector_db_client_defaults
    }


def init_storage_context(
    vector_storage_defaults: Dict[str, Any] = {},
    document_storage_defaults: Dict[str, Any] = {},
    storage_context_defaults: Dict[str, Any] = {},
    vector_db_client_defaults: Dict[str, Any] = {}
) -> Optional[StorageContext]:
    try:
        client = qdrant_client.QdrantClient(**vector_db_client_defaults)

        vector_store = QdrantVectorStore(
            client=client,
            **vector_storage_defaults
        )

        is_uri = 'uri' in document_storage_defaults.keys()
        if is_uri:
            doc_store = MongoDocumentStore.from_uri(**document_storage_defaults)
        else: 
            doc_store = SimpleDocumentStore(**document_storage_defaults)

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=doc_store,
            **storage_context_defaults
        )
        return storage_context
    except Exception as e:
        print(f"Error loading storage context")
        print(e)
        return None