from typing import Any
from hashlib import sha256
import qdrant_client
from qdrant_client import models
from typing import Dict, List, Set
from rich import print


def get_file_hash(**kwargs: Any) -> str:
        file_identity = str(kwargs)
        id = str(sha256(file_identity.encode("utf-8", "surrogatepass")).hexdigest())
        return id

def event_callback(**kwargs: Any) -> bool:
    return True

    if 'message' in kwargs:
        print(f"{kwargs['message']}")

    if ('type' in kwargs) and (kwargs['type'] == 'save'):
        return True

    if not(kwargs['file'] or kwargs['qdrant_client'] or kwargs['qdrant_collection']):
        return True

    client: qdrant_client.QdrantClient = kwargs['qdrant_client']
    collection: str = kwargs['qdrant_collection']
    result = client.scroll(
        limit=1024*100,
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="file_name",
                    match=models.MatchAny(any=[ kwargs['file'] ]),
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
    
    
    try: 
        if len(refs[kwargs['file']]) == int(kwargs['count']):
            # print(f"[✓] Skipped {kwargs['file']}\n")
            return False
        else: 
            client.delete(
                collection_name=collection,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="file_name",
                            match=models.MatchAny(any=[ kwargs['file'] ]),
                        )
                    ]
                )
            )
            return True
    except:
        return True

    return True