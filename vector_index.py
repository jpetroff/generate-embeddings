from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import MetadataFilters, MetadataFilter, FilterCondition
import qdrant_client
from qdrant_client import models
from qdrant_client.conversions import common_types as types
from rich import print
from rich.pretty import pprint
from rich.console import Console
from dotenv import dotenv_values
from typing import Dict, List, Set

env = dotenv_values('.default.env')

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

vector_store = QdrantVectorStore(
    client=client,
    collection_name=env['QDRANT_COLLECTION'] or '',
    index_doc_id=True
)

directory_reader = SimpleDirectoryReader(input_dir='./data')
files_to_ingest = directory_reader.input_files
file_list = [f.name for f in files_to_ingest]

pprint(file_list)

result = client.scroll(
    limit=1024*100,
    collection_name=env['QDRANT_COLLECTION'] or '',
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(
                key="file_name",
                match=models.MatchAny(any=file_list),
            )
        ]
    )
)
points = result[0]
refs: Dict[str, Set[str]] = {}

for point in points:
    point_file_name = point.payload['file_name'] if point.payload else None
    point_doc_ref_id = point.payload['doc_id'] if point.payload else None
    if point_file_name is None or point_doc_ref_id is None:
        continue

    if point_file_name not in refs.keys():
        refs[point_file_name] = set()

    refs[point_file_name].add(point_doc_ref_id)

print(refs)