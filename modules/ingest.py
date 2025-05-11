import abc
import logging
from pathlib import Path
from typing import Any, Generator, List, Sequence, Optional, Dict
from time import time
import multiprocessing
from itertools import chain

from llama_index.core.data_structs import IndexDict
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.indices import load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import BaseNode, Document, TransformComponent, MetadataMode
from llama_index.core.storage import StorageContext
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from modules.pipeline import ExtIngestionPipeline, IngestionStep, IngestionStrategy
from typing import List

from rich import print
from rich.status import Status

from modules.reader import Reader

from modules.progress import progress_relay, global_console

logger = logging.getLogger('generate-app')

class BaseIngestComponent(abc.ABC):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: List[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        logger.debug("Initializing base ingest component type=%s", type(self).__name__)
        self.storage_context = storage_context
        self.embed_model = embed_model
        self.transformations = transformations

    @abc.abstractmethod
    def ingest(self, file_name: str, file_path: Path) -> tuple[Sequence[Document], Sequence[BaseNode], Sequence[BaseNode]]:
        pass

    @abc.abstractmethod
    def bulk_ingest(self, files: List[tuple[str, Path]]) -> None:
        pass

    @abc.abstractmethod
    def delete(self, doc_id: str) -> None:
        pass
    
class IngestPipelineComponent(BaseIngestComponent):
    def __init__(
        self,
        storage_context: StorageContext,
        embed_model: EmbedType,
        transformations: List[TransformComponent],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(storage_context, embed_model, transformations, *args, **kwargs)

        self._docstore = storage_context.docstore
        self._vector_store = storage_context.vector_store
        self._document_cache: List[List[Document]] = list()

    def ingest_cached(self, documents: List[Document], file_path: Path) -> tuple[Sequence[Document], Sequence[BaseNode], Sequence[BaseNode]]:
        file_name = file_path.name
        logger.info("Ingesting file_name=%s", file_name)        
        logger.info("Transformed file=%s into count=%s documents", file_name, len(documents))
        
        (
            returned_documents, 
            nodes, 
            nodes_with_embeddings 
        ) = self._save_docs(documents)

        return returned_documents, nodes, nodes_with_embeddings

    def ingest(self, file_name: str, file_path: Path) -> tuple[Sequence[Document], Sequence[BaseNode], Sequence[BaseNode]]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = Reader.transform_file_into_documents(file_path=file_path)
        for document in documents:
            document.metadata['file_name'] = file_name
        logger.info("Transformed file=%s into count=%s documents", file_name, len(documents))
        (
            documents, 
            nodes, 
            nodes_with_embeddings 
        ) = self._save_docs(documents)

        return documents, nodes, nodes_with_embeddings

    def bulk_ingest(self, files: List[tuple[str, Path]], buffer_transforms: int = 1):
        all_file_paths = [ f for _, f in files ]
        reader = Reader(all_file_paths)
        for file_path, documents in reader:
            progress_relay.init_step_context(
                console=global_console,
                status=f"Reading",
                append=f"[bold blue]{file_path.name}[/]"
            )
            file_name = file_path.name
            raised_exception: Optional[Exception] = None
            ( returned_documents, generated_nodes, generated_nodes_with_embeddings ) = ([], [], [])
            try:

                # ↓ Actual processing here 
                (
                    returned_documents, 
                    generated_nodes, 
                    generated_nodes_with_embeddings
                ) = self.ingest_cached(
                    documents=documents,
                    file_path=file_path
                )
                
            except Exception as e:
                raised_exception = e

            finally:
                if raised_exception is not None:
                    end_message = (
                        f"[red]![/] Failed [red]{file_name}[/]\n"
                        f"\u2514\u2500 [dim][red]{str(raised_exception)}[/]"
                    )
                elif generated_nodes and len(generated_nodes) > 0:
                    end_message = (
                        f"[green]✓[/] Completed [green]{file_name}[/]\n"
                        f"\u2514\u2500 [dim]New documents: {len(returned_documents)} → Nodes: {len(generated_nodes)} → With embeddings: {len(generated_nodes_with_embeddings)}[/]"
                    )
                else:
                    end_message = (
                        f"[blue]•[/] Skipped [blue]{file_name}[/]"
                    )
                progress_relay.end_step_context(message=end_message)

    def _save_docs(self, documents: List[Document]) -> tuple[Sequence[Document], Sequence[BaseNode], Sequence[BaseNode]]:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        
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

        generated_nodes, generated_nodes_with_embeddings = pipeline.run(
            documents=documents,
            num_workers=4,
            show_progress=True
        )
        
        return documents, generated_nodes, generated_nodes_with_embeddings
    
    def delete(self, doc_id: str) -> None:
        # Delete the document from the index
        self.storage_context.docstore.delete_ref_doc(doc_id)