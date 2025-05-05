import abc
import itertools
import logging
import multiprocessing
import multiprocessing.pool
import os
import threading
from pathlib import Path
from queue import Queue
from typing import Any
from hashlib import sha256

from llama_index.core.data_structs import IndexDict
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import BaseNode, Document, TransformComponent, MetadataMode
from llama_index.core.storage import StorageContext
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from pipeline import ExtIngestionPipeline, IngestionStep, IngestionStrategy
from typing import List

from rich import print

from reader import Reader

logger = logging.getLogger('generate-app')

class BaseIngestComponent(abc.ABC):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Initializing base ingest component type=%s", type(self).__name__)
        self.storage_context = storage_context
        self.embed_model = embed_model
        self.transformations = transformations

    @abc.abstractmethod
    def ingest(self, file_name: str, file_path: Path) -> list[Document]:
        pass

    @abc.abstractmethod
    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        pass

    @abc.abstractmethod
    def delete(self, doc_id: str) -> None:
        pass
    
class IngestPipelineComponent(BaseIngestComponent):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: list[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)

        self._docstore = storage_context.docstore
        self._vector_store = storage_context.vector_store

    def ingest(self, file_name: str, file_path: Path) -> list[Document]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = Reader.transform_file_into_documents(file_path=file_path)
        for document in documents:
            document.metadata['file_name'] = file_name
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )

        return documents

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> list[Document]:
        saved_documents: list[Document] = []
        for file_name, file_path in files:
            documents = self.ingest(
                file_name=file_name,
                file_path=file_path
            )
            saved_documents.extend(self._save_docs(documents))
        return saved_documents

    def _save_docs(self, documents: list[Document]) -> list[Document]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        
        # pipeline = IngestionPipeline(
        #     transformations=self.transformations or [],
        #     vector_store=self.storage_context.vector_store,
        #     docstore=self.storage_context.docstore,
        #     docstore_strategy=DocstoreStrategy.UPSERTS
        # )
        
        steps: List[IngestionStep] = []
        for transformation in self.transformations:
            steps.append(
                IngestionStep(
                    transform=transformation,
                    name=transformation.__class__.__name__,
                    threads=2 if transformation.__class__.__name__ == "OllamaEmbedding" else 8
                )
            )

        pipeline = ExtIngestionPipeline(
            steps=steps,
            vector_store=self.storage_context.vector_store,
            docstore=self.storage_context.docstore,
            ingestion_strategy=IngestionStrategy.UPSERTS
        )

        nodes = pipeline.run(
            documents=documents,
            num_workers=4,
            show_progress=True
        )

        # logger.info("Inserting count=%s nodes into docstore", len(nodes))
        # for document in documents:
        #     self._docstore.set_document_hash(
        #         document.get_doc_id(), document.hash
        #     )
        
        return documents
    
    def delete(self, doc_id: str) -> None:
        # Delete the document from the index
        self.storage_context.docstore.delete_ref_doc(doc_id)