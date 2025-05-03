import logging
from pathlib import Path
from datetime import datetime, timezone
import fsspec
from fsspec.implementations.local import LocalFileSystem
import mimetypes
import os

from llama_index.core.readers import StringIterableReader
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.json import JSONReader
from llama_index.core.schema import Document
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.file.base import default_file_metadata_func

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
    """Helper class to transform a file into a list of documents.

    This class should be used to transform a file into a list of documents.
    These methods are thread-safe (and multiprocessing-safe).
    """

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
    ) -> list[Document]:
        # documents = Reader._load_file_to_documents(file_name, file_data)
        # for document in documents:
        #     document.metadata["file_name"] = file_name
        # Reader._exclude_metadata(documents)
        documents = SimpleDirectoryReader.load_file(
            input_file=file_path,
            file_metadata=Reader.metadata,
            file_extractor=FILE_READER_CLS,
            filename_as_id=True,
            **kwargs
        )
        Reader._exclude_metadata(documents)
        return documents

    # @staticmethod
    # def _load_file_to_documents(file_name: str, file_data: Path) -> list[Document]:
    #     logger.debug("Transforming file_name=%s into documents", file_name)
    #     extension = Path(file_name).suffix
    #     reader_cls = FILE_READER_CLS.get(extension)
    #     if reader_cls is None:
    #         logger.debug(
    #             "No reader found for extension=%s, using default string reader",
    #             extension,
    #         )
    #         # Read as a plain text
    #         string_reader = StringIterableReader()
    #         return string_reader.load_data([file_data.read_text()])

    #     logger.debug("Specific reader found for extension=%s", extension)
    #     return reader_cls.load_data(file_data)