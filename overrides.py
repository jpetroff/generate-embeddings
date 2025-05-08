import llama_index.core.node_parser as parent_parser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import (
    BaseNode
)
from typing import (
    Sequence,
    List,
    Any
)
from llama_index.core.utils import get_tqdm_iterable

def _parse_nodes(
    self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
):
    all_nodes: List[BaseNode] = []
    nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
    for node in nodes_with_progress:
        splits = self.split_text(node.get_content())

        all_nodes.extend(
            build_nodes_from_splits(splits, node, id_func=self.id_func)
        )

    return all_nodes

parent_parser.NodeParser._parse_nodes = _parse_nodes