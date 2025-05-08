from llama_index.core.schema import (
    BaseNode
)

from typing import (
    Any, Callable, Generator, List, Optional, Sequence, Union, Dict,
    TypeVar, Iterator
)
    

from collections.abc import Sequence as ABCSequence

T = TypeVar('T', bound=BaseNode)

class NodeTrackingSequence(ABCSequence[T]):
    """A sequence wrapper that tracks node IDs during iteration."""
    
    def __init__(self, sequence: Sequence[T], callback: Optional[Callable[[BaseNode], None]]):
        self._callback = callback
        self._sequence = sequence
        
    def __iter__(self) -> Iterator[T]:
        """Iterate over the sequence while tracking node IDs."""
        for node in self._sequence:
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
            yield node
            
    def index(self, value: T, start: int = 0, stop: Optional[int] = None) -> int:
        return self._sequence.index(value, start, stop or len(self._sequence))
        
    def count(self, value: T) -> int:
        return self._sequence.count(value)