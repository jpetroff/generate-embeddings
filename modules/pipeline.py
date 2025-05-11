from math import inf
import multiprocessing.pool
import itertools
from enum import Enum
from functools import partial, reduce
from operator import mod
from pathlib import Path
from time import sleep
from typing import Any, Callable, Generator, List, Optional, Sequence, Union
import queue
import threading

from fsspec import AbstractFileSystem

from llama_index.core.constants import (
    DEFAULT_PIPELINE_NAME,
    DEFAULT_PROJECT_NAME,
)
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache
from llama_index.core.readers.base import ReaderConfig
from llama_index.core.schema import (
    BaseNode,
    Document,
    TransformComponent,
)
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore import (
    BaseDocumentStore
)
from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME
from llama_index.core.utils import concat_dirs
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from modules.extend_tqdm import ExtTqdm
from modules.node_tracking_sequence import NodeTrackingSequence
from modules.utils import EOQUEUE_SYMBOL, get_transformation_hash
from modules.progress import progress_relay, global_console


class IngestionStep(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(
        default='Generic Transformation Step', description="Name of the step to report"
    )
    transform: TransformComponent = Field(
        description="Transformation step", 
    )
    threads: Optional[int] = Field(
        default=None, description="Run in parallel threads (if > 1), None means global num_workers value is applied"
    )

    def __init__(
        self,
        transform: TransformComponent,
        name: str = 'Generic Transformation Step',
        threads: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            transform=transform,
            name=name,
            threads=threads,
            **kwargs
        )

    def __call__(
        self,
        nodes,
        **kwargs
    ) -> Sequence[BaseNode]:
        return self.transform(nodes, **kwargs)

class IngestionStrategy(str, Enum):
    """
    Document de-duplication de-deduplication strategies work by comparing the hashes or ids stored in the document store.
       They require a document store to be set which must be persisted across pipeline runs.

    Attributes:
        UPSERTS:
            ('upserts') Use upserts to handle duplicates. Checks if the a document is already in the doc store based on its id. If it is not, or if the hash of the document is updated, it will update the document in the doc store and run the transformations.
        DUPLICATES_ONLY:
            ('duplicates_only') Only handle duplicates. Checks if the hash of a document is already in the doc store. Only then it will add the document to the doc store and run the transformations
        UPSERTS_AND_DELETE:
            ('upserts_and_delete') Use upserts and delete to handle duplicates. Like the upsert strategy but it will also delete non-existing documents from the doc store
    """

    UPSERTS = "upserts"
    DUPLICATES_ONLY = "duplicates_only"
    UPSERTS_AND_DELETE = "upserts_and_delete"

def _process_worker(step, nodes, output_queue: queue.Queue, **kwargs):

    from modules.extend_tqdm import ExtTqdm

    ExtTqdm._output_queue = output_queue
    result = step(nodes, **kwargs)

    if isinstance(result, NodeTrackingSequence):
        return result._sequence

    return result

def run_step(
    nodes: Sequence[BaseNode],
    step: IngestionStep,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    num_workers: int = 1,
    _parent_name: str = DEFAULT_PIPELINE_NAME,
    **kwargs: Any,
) -> Sequence[BaseNode]:

    num_nodes = len(nodes)
    num_workers = min(int(step.threads or inf), num_workers, num_nodes)
    is_multiprocessing = num_workers > 1

    transform = step.transform

    if progress_relay.is_active_task:
        progress_relay.start_task(
            description=f"{step.name}[dim] x{num_workers}[/]",
            total=num_nodes
        )

    hash: str = ''
    if cache is not None:
        hash = get_transformation_hash(nodes, transform)
        cached_nodes = cache.get(hash, collection=cache_collection)
        if cached_nodes is not None:
            nodes = cached_nodes
            return nodes
    
    if is_multiprocessing:
        # Create a manager for sharing the queue
        manager = multiprocessing.Manager()
        output_queue = manager.Queue()

        # Start a thread to process the output queue
        def process_output():
            while True:
                try:
                    output = output_queue.get_nowait()

                    if output == EOQUEUE_SYMBOL:
                        return 0
                    else:
                        progress_relay.advance_progress(result=output)

                except queue.Empty:
                    continue
        
        output_thread = threading.Thread(target=process_output)
        output_thread.start()
        
        step_work_pool = multiprocessing.Pool(num_workers)

        node_batches = ExtIngestionPipeline._node_batcher(
            num_batches=num_workers, nodes=nodes
        )
        
        # Create partial function with the output queue
        worker_func = partial(_process_worker, 
            step, # nodes, ← expecting positional argument
            output_queue=output_queue, # named kwargs
            **kwargs # kwargs passed to TransformComponent
        )
        
        nodes = list(
            itertools.chain.from_iterable(
                step_work_pool.starmap(
                    worker_func,
                    zip(node_batches)
                )
            )
        )
        
        output_queue.put(EOQUEUE_SYMBOL)
        # Wait for output processing to complete
        output_thread.join()
        
    else:
        """
        Single-process
        """
        ExtTqdm._output_fn = lambda _,obj: progress_relay.advance_progress()
        nodes = transform(nodes, **kwargs)
        ExtTqdm._output_fn = None

    if cache is not None:
        cache.put(hash, nodes, collection=cache_collection)

    if progress_relay.is_active_task:
        progress_relay.end_task()
    return nodes

class ExtIngestionPipeline(BaseModel):
    """
    An ingestion pipeline that can be applied to data.

    Args:
        name (str, optional):
            Unique name of the ingestion pipeline. Defaults to DEFAULT_PIPELINE_NAME.
        project_name (str, optional):
            Unique name of the project. Defaults to DEFAULT_PROJECT_NAME.
        transformations (List[TransformComponent], optional):
            Transformations to apply to the data. Defaults to None.
        documents (Optional[Sequence[Document]], optional):
            Documents to ingest. Defaults to None.
        readers (Optional[List[ReaderConfig]], optional):
            Reader to use to read the data. Defaults to None.
        vector_store (Optional[BasePydanticVectorStore], optional):
            Vector store to use to store the data. Defaults to None.
        cache (Optional[IngestionCache], optional):
            Cache to use to store the data. Defaults to None.
        docstore (Optional[BaseDocumentStore], optional):
            Document store to use for de-duping with a vector store. Defaults to None.
        ingestion_strategy (IngestionStrategy, optional):
            Document de-dup strategy. Defaults to IngestionStrategy.UPSERTS.
        disable_cache (bool, optional):
            Disable the cache. Defaults to False.
        base_url (str, optional):
            Base URL for the LlamaCloud API. Defaults to DEFAULT_BASE_URL.
        app_url (str, optional):
            Base URL for the LlamaCloud app. Defaults to DEFAULT_APP_URL.
        api_key (Optional[str], optional):
            LlamaCloud API key. Defaults to None.

    Examples:
        ```python
        from llama_index.core.ingestion import IngestionPipeline
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.embeddings.openai import OpenAIEmbedding

        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(chunk_size=512, chunk_overlap=20),
                OpenAIEmbedding(),
            ],
        )

        nodes = pipeline.run(documents=documents)
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str = Field(
        default=DEFAULT_PIPELINE_NAME,
        description="Unique name of the ingestion pipeline",
    )
    project_name: str = Field(
        default=DEFAULT_PROJECT_NAME, description="Unique name of the project"
    )

    steps: List[IngestionStep] = Field(
        description="Transformations to apply to the data"
    )

    documents: Optional[Sequence[Document]] = Field(description="Documents to ingest")
    readers: Optional[List[ReaderConfig]] = Field(
        description="Reader to use to read the data"
    )
    vector_store: Optional[BasePydanticVectorStore] = Field(
        description="Vector store to use to store the data"
    )
    cache: IngestionCache = Field(
        default_factory=IngestionCache,
        description="Cache to use to store the data",
    )
    docstore: Optional[BaseDocumentStore] = Field(
        default=None,
        description="Document store to use for de-duping with a vector store.",
    )
    ingestion_strategy: IngestionStrategy = Field(
        default=IngestionStrategy.UPSERTS, description="Document de-dup strategy."
    )
    disable_cache: bool = Field(
        default=True, description="Disable the cache"
    )
    progressFn: Optional[Callable] = Field(
        description="Callback function to report progress"
    )


    def __init__(
        self,
        name: str = DEFAULT_PIPELINE_NAME,
        project_name: str = DEFAULT_PROJECT_NAME,
        steps: List[IngestionStep] = [],
        readers: Optional[List[ReaderConfig]] = None,
        documents: Optional[Sequence[Document]] = None,
        vector_store: Optional[BasePydanticVectorStore] = None,
        cache: Optional[IngestionCache] = None,
        docstore: Optional[BaseDocumentStore] = None,
        ingestion_strategy: IngestionStrategy = IngestionStrategy.UPSERTS,
        disable_cache: bool = True,
        progressFn: Optional[Callable] = None
    ) -> None:

        super().__init__(
            name=name,
            project_name=project_name,
            steps=steps,
            readers=readers,
            documents=documents,
            vector_store=vector_store,
            cache=cache or IngestionCache(),
            docstore=docstore,
            ingestion_strategy=ingestion_strategy,
            disable_cache=disable_cache,
            progressFn=progressFn
        )

    def persist(
        self,
        persist_dir: str = "./pipeline_storage",
        fs: Optional[AbstractFileSystem] = None,
        cache_name: str = DEFAULT_CACHE_NAME,
        docstore_name: str = DEFAULT_PERSIST_FNAME,
    ) -> None:
        """Persist the pipeline to disk."""
        if fs is not None:
            persist_dir = str(persist_dir)  # NOTE: doesn't support Windows here
            docstore_path = concat_dirs(persist_dir, docstore_name)
            cache_path = concat_dirs(persist_dir, cache_name)

        else:
            persist_path = Path(persist_dir)
            docstore_path = str(persist_path / docstore_name)
            cache_path = str(persist_path / cache_name)

        self.cache.persist(cache_path, fs=fs)
        if self.docstore is not None:
            self.docstore.persist(docstore_path, fs=fs)

    def _prepare_inputs(
        self,
        documents: Optional[Sequence[Document]],
        nodes: Optional[Sequence[BaseNode]],
    ) -> Sequence[BaseNode]:
        input_nodes: Sequence[BaseNode] = []

        if documents is not None:
            input_nodes += documents  # type: ignore

        if nodes is not None:
            input_nodes += nodes  # type: ignore

        if self.documents is not None:
            input_nodes += self.documents  # type: ignore

        if self.readers is not None:
            for reader in self.readers:
                input_nodes += reader.read()  # type: ignore

        return input_nodes
    
    def _handle_duplicates(
        self,
        nodes: Sequence[BaseNode],
        store_doc_text: bool = True,
    ) -> Sequence[BaseNode]:
        """Handle docstore duplicates by checking all hashes."""
        assert self.docstore is not None

        existing_hashes = self.docstore.get_all_document_hashes()
        current_hashes = []
        nodes_to_run = []
        for node in nodes:
            if node.hash not in existing_hashes and node.hash not in current_hashes:
                # self.docstore.set_document_hash(node.id_, node.hash)
                nodes_to_run.append(node)
                current_hashes.append(node.hash)

        # self.docstore.add_documents(nodes_to_run, store_text=store_doc_text)

        return nodes_to_run
    
    def _handle_upserts(
        self,
        nodes: Sequence[BaseNode],
        store_doc_text: bool = True,
    ) -> Sequence[BaseNode]:
        """Handle docstore upserts by checking hashes and ids."""
        assert self.docstore is not None

        doc_ids_from_nodes = set()
        deduped_nodes_to_run = {}
        for node in nodes:
            ref_doc_id = node.ref_doc_id if node.ref_doc_id else node.id_
            doc_ids_from_nodes.add(ref_doc_id)
            existing_hash = self.docstore.get_document_hash(ref_doc_id)
            if not existing_hash:
                # document doesn't exist, so add it
                deduped_nodes_to_run[ref_doc_id] = node
            elif existing_hash and existing_hash != node.hash:
                self.docstore.delete_ref_doc(ref_doc_id, raise_error=False)

                if self.vector_store is not None:
                    self.vector_store.delete(ref_doc_id)

                deduped_nodes_to_run[ref_doc_id] = node
            else:
                continue  # document exists and is unchanged, so skip it

        if self.ingestion_strategy == IngestionStrategy.UPSERTS_AND_DELETE:
            # Identify missing docs and delete them from docstore and vector store
            existing_doc_ids_before = set(
                self.docstore.get_all_document_hashes().values()
            )
            doc_ids_to_delete = existing_doc_ids_before - doc_ids_from_nodes
            for ref_doc_id in doc_ids_to_delete:
                self.docstore.delete_document(ref_doc_id)

                if self.vector_store is not None:
                    self.vector_store.delete(ref_doc_id)

        nodes_to_run = list(deduped_nodes_to_run.values())
        # self.docstore.set_document_hashes({n.id_: n.hash for n in nodes_to_run})
        # self.docstore.add_documents(nodes_to_run, store_text=store_doc_text)

        return nodes_to_run
    
    @staticmethod
    def _node_batcher(
        num_batches: int, nodes: Union[Sequence[BaseNode], List[Document]]
    ) -> Generator[Union[Sequence[BaseNode], List[Document]], Any, Any]:
        if not nodes:
            return
            
        # Calculate base batch size and remainder
        total_nodes = len(nodes)
        base_batch_size = total_nodes // num_batches
        remainder = total_nodes % num_batches
        
        # Calculate start indices for each batch
        start_indices = [0]
        current_index = 0
        
        # Distribute remainder nodes across batches
        for i in range(num_batches):
            batch_size = base_batch_size + (1 if i < remainder else 0)
            current_index += batch_size
            start_indices.append(current_index)
            
        # Yield batches
        for i in range(num_batches):
            yield nodes[start_indices[i]:start_indices[i + 1]]

    def run(
        self,
        documents: Optional[List[Document]] = None,
        nodes: Optional[Sequence[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        store_doc_text: bool = True,
        num_workers: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> tuple[Sequence[BaseNode], Sequence[BaseNode]]:
        """
        Run a series of transformations on a set of nodes.

        If a vector store is provided, nodes with embeddings will be added to the vector store.

        If a vector store + docstore are provided, the docstore will be used to de-duplicate documents.

        Args:
            show_progress (bool, optional): Shows execution progress bar(s). Defaults to False.
            documents (Optional[List[Document]], optional): Set of documents to be transformed. Defaults to None.
            nodes (Optional[Sequence[BaseNode]], optional): Set of nodes to be transformed. Defaults to None.
            cache_collection (Optional[str], optional): Cache for transformations. Defaults to None.
            in_place (bool, optional): Whether transformations creates a new list for transformed nodes or modifies the
                array passed to `run_transformations`. Defaults to True.
            num_workers (Optional[int], optional): The number of parallel processes to use.
                If set to None, then sequential compute is used. Defaults to None.

        Returns:
            Sequence[BaseNode]: The set of transformed Nodes/Documents
        """

        input_nodes = self._prepare_inputs(documents, nodes)

        if progress_relay.is_active_task:
            progress_relay.update_status(f"Checking {len(input_nodes)} docs for duplicated")

        # check if we need to dedup
        if self.docstore is not None and self.vector_store is not None:
            if self.ingestion_strategy in (
                IngestionStrategy.UPSERTS,
                IngestionStrategy.UPSERTS_AND_DELETE,
            ):
                nodes_to_run = self._handle_upserts(
                    input_nodes, store_doc_text=store_doc_text
                )
            elif self.ingestion_strategy == IngestionStrategy.DUPLICATES_ONLY:
                nodes_to_run = self._handle_duplicates(
                    input_nodes, store_doc_text=store_doc_text
                )
            else:
                raise ValueError(f"Invalid docstore strategy: {self.ingestion_strategy}")
        elif self.docstore is not None and self.vector_store is None:
            if self.ingestion_strategy == IngestionStrategy.UPSERTS:
                print(
                    "Docstore strategy set to upserts, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.ingestion_strategy = IngestionStrategy.DUPLICATES_ONLY
            elif self.ingestion_strategy == IngestionStrategy.UPSERTS_AND_DELETE:
                print(
                    "Docstore strategy set to upserts and delete, but no vector store. "
                    "Switching to duplicates_only strategy."
                )
                self.ingestion_strategy = IngestionStrategy.DUPLICATES_ONLY
            nodes_to_run = self._handle_duplicates(
                input_nodes, store_doc_text=store_doc_text
            )

        else:
            nodes_to_run = input_nodes

        if len(nodes_to_run) == 0:
            return [], []
        
        docs_to_save = nodes_to_run
        
        if progress_relay.is_active_task:
            progress_relay.update_status(f"Generating embeddings for {len(input_nodes)} documents")

        for step in self.steps:
            nodes_to_run = run_step(
                nodes=nodes_to_run,
                step=step,
                progressFn=self.progressFn,
                _parent_name=self.name,
                cache=self.cache,
                cache_collection=cache_collection,
                num_workers=num_workers,
                show_progress=show_progress
            )

        """
        Add transformation to indices
        """
        nodes_with_embeddings = [n for n in nodes_to_run if n.embedding is not None]
        if (self.vector_store is not None) and nodes_with_embeddings:
            self.vector_store.add(nodes_with_embeddings)

        if self.docstore is not None:
            self.docstore.set_document_hashes({n.id_: n.hash for n in docs_to_save})
            self.docstore.add_documents(docs_to_save, store_text=store_doc_text)

        return nodes_to_run, nodes_with_embeddings