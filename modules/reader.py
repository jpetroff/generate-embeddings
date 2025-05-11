import logging
from pathlib import Path
from datetime import datetime, timezone
import fsspec
from fsspec.implementations.local import LocalFileSystem
import mimetypes
import os
from typing import List, Iterator, Optional, Tuple, Dict
from threading import Thread, Lock
import queue
import time

from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SimpleDirectoryReader

from modules.progress import progress_relay, global_console

logger = logging.getLogger('generate-app')

# Inspired by the `llama_index.core.readers.file.base` module
def _try_loading_included_file_formats() -> dict[str, BaseReader]:
    try:
        from llama_index.readers.file.docs import (  # type: ignore
            DocxReader,
            HWPReader,
            PDFReader,
        )
        from llama_index.readers.file.epub import EpubReader  # type: ignore
        from llama_index.readers.file.image import ImageReader  # type: ignore
        from llama_index.readers.file.ipynb import IPYNBReader  # type: ignore
        from llama_index.readers.file.markdown import MarkdownReader  # type: ignore
        from llama_index.readers.file.mbox import MboxReader  # type: ignore
        from llama_index.readers.file.slides import PptxReader  # type: ignore
        from llama_index.readers.file.tabular import PandasCSVReader  # type: ignore
        from llama_index.readers.file.video_audio import (  # type: ignore
            VideoAudioReader,
        )
    except ImportError as e:
        raise ImportError("`llama-index-readers-file` package not found") from e

    # uncomment to use custom reader
    default_file_reader_cls: dict[str, BaseReader] = {
        # ".hwp": HWPReader(),
        # ".pdf": PDFReader(),
        # ".docx": DocxReader(),
        # ".pptx": PptxReader(),
        # ".ppt": PptxReader(),
        # ".pptm": PptxReader(),
        # ".jpg": ImageReader(),
        # ".png": ImageReader(),
        # ".jpeg": ImageReader(),
        # ".mp3": VideoAudioReader(),
        # ".mp4": VideoAudioReader(),
        # ".csv": PandasCSVReader(),
        # ".epub": EpubReader(),
        # ".md": MarkdownReader(),
        # ".mbox": MboxReader(),
        # ".ipynb": IPYNBReader(),
    }
    return default_file_reader_cls


# Patching the default file reader to support other file types
FILE_READER_CLS: dict[str, BaseReader] = _try_loading_included_file_formats()
FILE_READER_CLS.update(
    {
        ".json": JSONReader(),
    }
)

def _format_file_timestamp(
    timestamp: float | None, include_time: bool = False
) -> str | None:
    """
    Format file timestamp to a string.
    The format will be %Y-%m-%d if include_time is False or missing,
    %Y-%m-%dT%H:%M:%SZ if include_time is True.

    Args:
        timestamp (float): timestamp in float
        include_time (bool): whether to include time in the formatted string

    Returns:
        str: formatted timestamp
        None: if the timestamp passed was None
    """
    if timestamp is None:
        return None

    # Convert timestamp to UTC
    # Check if timestamp is already a datetime object
    if isinstance(timestamp, datetime):
        timestamp_dt = timestamp.astimezone(timezone.utc)
    else:
        timestamp_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)

    if include_time:
        return timestamp_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return timestamp_dt.strftime("%Y-%m-%d")

def get_default_fs() -> fsspec.AbstractFileSystem:
    return LocalFileSystem()


class Reader:
    """Helper class to transform files into documents with background processing.

    This class processes multiple files in parallel (up to 3 at a time) and maintains
    a buffer of processed documents in memory. It provides an iterator interface to
    access the documents efficiently.
    """

    def __init__(self, file_paths: List[Path], max_parallel_files: int = 3):
        self._file_paths = file_paths
        self._current_file_index = 0
        self._max_parallel_files = max_parallel_files
        self._processing_threads: List[Thread] = []
        self._thread_file_map: Dict[int, Path] = {}  # Map thread IDs to file paths
        self._result_dict: Dict[str, List[Document]] = {}
        self._result_lock = Lock()
        self._error_queue: queue.Queue = queue.Queue()
        self._processed_files: List[Tuple[Path, List[Document]]] = []
        self._is_processing = False

    def _load_documents_in_background(self, file_path: Path, **kwargs):
        """Load documents in a background thread and store results in shared dict."""
        try:
            documents = self.transform_file_into_documents(file_path, **kwargs)
            with self._result_lock:
                self._result_dict[str(file_path)] = documents
        except Exception as e:
            logger.error(f"Error loading documents from {file_path}: {e}")
            self._error_queue.put((str(file_path), str(e)))

    def _start_processing_files(self, **kwargs) -> None:
        """Start processing up to max_parallel_files files in the background."""
        while (len(self._processing_threads) < self._max_parallel_files and 
               self._current_file_index < len(self._file_paths)):
            
            file_path = self._file_paths[self._current_file_index]
            thread = Thread(
                target=self._load_documents_in_background,
                args=(file_path,),
                kwargs=kwargs
            )
            
            self._processing_threads.append(thread)
            self._thread_file_map[id(thread)] = file_path
            thread.start()
            self._current_file_index += 1

    def _check_processed_files(self) -> None:
        """Check for completed file processing and move results to buffer."""
        completed_indices = []
        for i, thread in enumerate(self._processing_threads):
            if not thread.is_alive():
                thread.join()
                file_path = self._thread_file_map[id(thread)]
                with self._result_lock:
                    if str(file_path) in self._result_dict:
                        documents = self._result_dict[str(file_path)]
                        self._processed_files.append((file_path, documents))
                        del self._result_dict[str(file_path)]
                del self._thread_file_map[id(thread)]
                completed_indices.append(i)
        
        # Remove completed threads in reverse order to avoid index issues
        for i in sorted(completed_indices, reverse=True):
            self._processing_threads.pop(i)

    def _wait_for_document(self, timeout: float = 0.1) -> Optional[Tuple[Path, List[Document]]]:
        """Wait for documents from the queue, handling process completion."""
        if not self._is_processing and self._current_file_index == 0:
            self._start_processing_files()
            self._is_processing = True

        progress_relay.init_step_context(
            console=global_console, 
            status=f'[dim]Preparing files (remaining files: {len(self._file_paths) - self._current_file_index}, available in buffer: {len(self._processed_files)}, active threads: {len(self._processing_threads)})[/]'
        )

        while True:
            # Check for any completed files
            self._check_processed_files()

            # If we still have files to process, start them
            if self._current_file_index < len(self._file_paths):
                self._start_processing_files()

            # If we have processed files in buffer, return the next one
            if self._processed_files:
                progress_relay.end_step_context()
                return self._processed_files.pop(0)

            # If we have no more files and all threads are done
            if not self._processing_threads and self._current_file_index >= len(self._file_paths):
                progress_relay.end_step_context(message="[dim]- Done[/]")
                return None

            # Wait a bit and try again
            time.sleep(timeout)

    def get_next_document(self, timeout: float = 0.1) -> Optional[Tuple[Path, List[Document]]]:
        """Get all documents from the next file in the queue, waiting if necessary."""
        return self._wait_for_document(timeout)

    def __iter__(self) -> Iterator[Tuple[Path, List[Document]]]:
        """Make the Reader class iterable."""
        return self

    def __next__(self) -> Tuple[Path, List[Document]]:
        """Get all documents from the next file in the iteration."""
        batch = self.get_next_document()
        if batch is None:
            raise StopIteration
        return batch

    def __del__(self):
        """Clean up threads when the reader is destroyed."""
        for thread in self._processing_threads:
            if thread.is_alive():
                thread.join()

    @staticmethod
    def metadata(
        file_path: str, fs: fsspec.AbstractFileSystem | None = None
    ) -> dict:
        """
        Get some handy metadata from filesystem.

        Args:
            file_path: str: file path in str
        """
        fs = fs or get_default_fs()
        stat_result = fs.stat(file_path)

        try:
            file_name = os.path.basename(str(stat_result["name"]))
        except Exception as e:
            file_name = os.path.basename(file_path)

        creation_date = _format_file_timestamp(stat_result.get("created"))
        last_modified_date = _format_file_timestamp(stat_result.get("mtime"))
        last_accessed_date = _format_file_timestamp(stat_result.get("atime"))
        default_meta = {
            "file_path": file_path,
            "file_name": file_name,
            "file_type": mimetypes.guess_type(file_path)[0],
            "file_size": stat_result.get("size"),
            "creation_date": creation_date,
            "last_modified_date": last_modified_date,
            "last_accessed_date": last_accessed_date,
        }

        # Return not null value
        return {
            meta_key: meta_value
            for meta_key, meta_value in default_meta.items()
            if meta_value is not None
        }
    
    @staticmethod
    def _exclude_metadata(documents: list[Document]) -> list[Document]:
        """
        Exclude metadata from documents.

        Args:
            documents (List[Document]): List of documents.
        """
        for doc in documents:
            # Keep only metadata['file_name'] in both embedding and llm content
            # str, which contain extreme important context that about the chunks.
            # Dates is provided for convenience of postprocessor such as
            # TimeWeightedPostprocessor, but excluded for embedding and LLMprompts
            doc.excluded_embed_metadata_keys.extend(
                [
                    "file_path",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                    "doc_id"
                ]
            )
            doc.excluded_llm_metadata_keys.extend(
                [
                    "file_path",
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                    "doc_id",
                    "page_label"
                ]
            )

        return documents


    @staticmethod
    def transform_file_into_documents(
        file_path: Path,
        **kwargs
    ) -> List[Document]:
        documents = SimpleDirectoryReader.load_file(
            input_file=file_path,
            file_metadata=Reader.metadata,
            file_extractor=FILE_READER_CLS,
            filename_as_id=False,
            **kwargs
        )
        return Reader.prepare_documents(documents, file_path)
    
    @staticmethod
    def prepare_documents(documents: List[Document], file_path: Path) -> List[Document]:
        if len(documents) == 1:
            documents[0].id_ = f"{file_path.name}"
        else:
            for i, document in enumerate(documents):
                document.id_ = f"{file_path.name}#part{i}"
        documents = Reader._exclude_metadata(documents)
        return documents