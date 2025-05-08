from importlib import reload
from typing import Callable, Optional

from pkg_resources import UnknownExtra
from tqdm.asyncio import tqdm_asyncio
# from tqdm.rich import tqdm_rich
import tqdm
import tqdm.asyncio
import tqdm.auto
from node_tracking_sequence import NodeTrackingSequence
import queue

class ExtTqdm(tqdm_asyncio):

    _output_queue: Optional[queue.Queue] = None

    def __init__(self, iterable=None, *args, **kwargs):
        super().__init__(iterable, disable=True, *args, **kwargs)
        # print('Updated with custom TQDM')
        self._is_modified = True

    def __iter__(self):
        """Backward-compatibility to use: for x in tqdm(iterable)"""

        # Inlining instance variables as locals (speed optimisation)
        iterable = self.iterable

        for obj in iterable:
            if hasattr(self, '_output_queue'):
                self._output_queue.put(obj)  # type: ignore
            yield obj



tqdm.asyncio.tqdm_asyncio = ExtTqdm
tqdm.asyncio.tqdm = tqdm.asyncio.tqdm_asyncio
reload(tqdm)
reload(tqdm.auto)