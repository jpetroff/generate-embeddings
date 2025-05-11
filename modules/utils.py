import re
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    TransformComponent,
)
from hashlib import sha256
from typing import Sequence

EOQUEUE_SYMBOL = "\0"

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