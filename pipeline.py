from ast import Call
import asyncio
from math import inf
import multiprocessing
from multiprocessing import shared_memory, Array
import multiprocessing.pool
import itertools
import os
from platform import node
import re
import warnings
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from functools import partial, reduce
from hashlib import sha256
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Sequence, Union, Dict, TypeVar, Iterator
import sys
import queue
import threading
from collections.abc import Sequence as ABCSequence

from fsspec import AbstractFileSystem

from progress import SHR_NODE_ID_NAME

from llama_index.core.constants import (
    DEFAULT_PIPELINE_NAME,
    DEFAULT_PROJECT_NAME,
)
# from llama_index.core.bridge.pydantic import (
#     AnyUrl,
#     BaseModel,
#     BaseComponent,
#     ConfigDict,
#     Field
# )
from llama_index.core.bridge.pydantic import BaseModel, Field, ConfigDict
from llama_index.core.ingestion.cache import DEFAULT_CACHE_NAME, IngestionCache
from llama_index.core.instrumentation import get_dispatcher
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import ReaderConfig
from llama_index.core.schema import (
    BaseNode,
    BaseComponent,
    Document,
    MetadataMode,
    TransformComponent,
)
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore import (
    BaseDocumentStore,
    SimpleDocumentStore,
)
from llama_index.core.storage.docstore.types import DEFAULT_PERSIST_FNAME
from llama_index.core.utils import concat_dirs
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from sqlalchemy import func

import rich

T = TypeVar('T', bound=BaseNode)

class NodeTrackingSequence(ABCSequence[T]):
    """A sequence wrapper that tracks node IDs during iteration."""
    
    def __init__(self, sequence: Sequence[T], callback: Optional[Callable[[BaseNode], None]]):
        self._callback = callback
        self._sequence = sequence
        
    def __iter__(self) -> Iterator[T]:
        """Iterate over the sequence while tracking node IDs."""
        for node in self._sequence:
            if callable(self._callback):
                self._callback(node)
            yield node
            
    def __len__(self) -> int:
        return len(self._sequence)
        
    def __getitem__(self, idx: Union[int, slice]) -> Union[T, 'NodeTrackingSequence[T]']:  # type: ignore
        if isinstance(idx, slice):
            return NodeTrackingSequence(self._sequence[idx], self._callback)
        return self._sequence[idx]
        
    def __contains__(self, item: object) -> bool:
        return item in self._sequence
        
    def __reversed__(self) -> Iterator[T]:
        for node in reversed(self._sequence):
            if callable(self._callback):
                self._callback(node)
            yield node
            
    def index(self, value: T, start: int = 0, stop: Optional[int] = None) -> int:
        return self._sequence.index(value, start, stop or len(self._sequence))
        
    def count(self, value: T) -> int:
        return self._sequence.count(value)

def remove_unstable_values(s: str) -> str:
    """
    Remove unstable key/value pairs.

    Examples include:
    - <__main__.Test object at 0x7fb9f3793f50>
    - <function test_fn at 0x7fb9f37a8900>
    """
    pattern = r"<[\w\s_\. ]+ at 0x[a-z0-9]+>"
    return re.sub(pattern, "", s)

def get_transformation_hash(
    nodes: Sequence[BaseNode], transformation: TransformComponent
) -> str:
    """Get the hash of a transformation."""
    nodes_str = "".join(
        [str(node.get_content(metadata_mode=MetadataMode.ALL)) for node in nodes]
    )

    transformation_dict = transformation.to_dict()
    transform_string = remove_unstable_values(str(transformation_dict))

    return sha256((nodes_str + transform_string).encode("utf-8")).hexdigest()

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
        *args,
        **kwargs
    ) -> Sequence[BaseNode]:
        nodes: Sequence[BaseNode] = args[0]
        _kwargs: Dict[str, Any] = args[1] if len(args) > 1 else {}
        return self.transform(nodes, **_kwargs, **kwargs)

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

def iterable_wrapper(nodes: Sequence[BaseNode]):
    for node in nodes:
        yield node

def _process_worker(step, nodes, kwargs, output_queue: queue.Queue):
    """Worker function that captures stdout and sends it to the queue."""
    # Redirect stdout to a StringIO
    # from io import StringIO
    # old_stdout = sys.stdout
    # sys.stdout = mystdout = StringIO()

    def _callback(node: BaseNode):
        output_queue.put(node.id_)

    # try:
        # Run the step
    trackable_nodes = NodeTrackingSequence(nodes, _callback)
    result = step(trackable_nodes, **kwargs)
        
        # Get the captured output
        # output = mystdout.getvalue()
        # if output:
        #     output_queue.put(output)
    
    output_queue.put("\0")
    if hasattr(result, '_sequence'):
        return result._sequence

    return result 
    # finally:
    #     del trackable_nodes
        # Restore stdout
        # sys.stdout = old_stdout

def run_step(
    nodes: Sequence[BaseNode],
    step: IngestionStep,
    progressFn: Optional[Callable],
    _parent_name: str = DEFAULT_PIPELINE_NAME,
    cache: Optional[IngestionCache] = None,
    cache_collection: Optional[str] = None,
    num_workers: int = 1,
    **kwargs: Any,
) -> Sequence[BaseNode]:

    num_nodes = len(nodes)
    num_workers = min(int(step.threads or inf), num_workers)
    is_multiprocessing = num_nodes > 1 and num_workers > 1

    transform = step.transform

    print(f"Step {step.name} | {num_workers} threads ({step.threads} step threads) | {num_nodes} nodes")

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
                    output = output_queue.get()
                    if output == "\0":
                        break
                    
                    rich.print(output, end='')
                except queue.Empty:
                    continue
        
        output_thread = threading.Thread(target=process_output)
        output_thread.start()
        
        step_work_pool = multiprocessing.Pool(num_workers)

        node_batches = ExtIngestionPipeline._node_batcher(
            num_batches=num_workers, nodes=nodes
        )
        
        # Create partial function with the output queue
        worker_func = partial(_process_worker, step, kwargs=kwargs, output_queue=output_queue)
        
        nodes = list(
            itertools.chain.from_iterable(
                step_work_pool.starmap(
                    worker_func,
                    zip(node_batches)
                )
            )
        )
        
        # Wait for output processing to complete
        
        
    else: 
        nodes = transform(nodes, **kwargs)

    if cache is not None:
        cache.put(hash, nodes, collection=cache_collection)

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
                self.docstore.set_document_hash(node.id_, node.hash)
                nodes_to_run.append(node)
                current_hashes.append(node.hash)

        #self.docstore.add_documents(nodes_to_run, store_text=store_doc_text)

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
        self.docstore.set_document_hashes({n.id_: n.hash for n in nodes_to_run})
        self.docstore.add_documents(nodes_to_run, store_text=store_doc_text)

        return nodes_to_run
    
    @staticmethod
    def _node_batcher(
        num_batches: int, nodes: Union[Sequence[BaseNode], List[Document]]
    ) -> Generator[Union[Sequence[BaseNode], List[Document]], Any, Any]:
        """Yield successive n-sized chunks from lst."""
        batch_size = max(1, int(len(nodes) / num_batches))
        for i in range(0, len(nodes), batch_size):
            yield nodes[i : i + batch_size]

    def run(
        self,
        documents: Optional[List[Document]] = None,
        nodes: Optional[Sequence[BaseNode]] = None,
        cache_collection: Optional[str] = None,
        store_doc_text: bool = True,
        num_workers: int = 1,
        **kwargs: Any,
    ) -> Sequence[BaseNode]:
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

        # @TODO: remove, used for debugging
        nodes_to_run = input_nodes

        for step in self.steps:
            nodes_to_run = run_step(
                nodes=nodes_to_run,
                step=step,
                progressFn=self.progressFn,
                _parent_name=self.name,
                cache=self.cache,
                cache_collection=cache_collection,
                num_workers=num_workers,
                show_progress=True
            )

        if self.vector_store is not None:
            nodes_with_embeddings = [n for n in nodes_to_run if n.embedding is not None]
            if nodes_with_embeddings:
                self.vector_store.add(nodes_with_embeddings)

        return nodes_to_run