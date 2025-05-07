from typing import Any, Sequence
from dataclasses import dataclass
from typing import Dict, List, Set, Optional

from importlib import reload

# from rich import print
# from rich.live import Live
# from rich.console import Console
from llama_index.core import Settings
from datetime import time
import uuid

from tqdm.asyncio import tqdm_asyncio
from tqdm.rich import tqdm_rich
import tqdm
import tqdm.asyncio
import tqdm.auto

class ExtTqdm(tqdm_asyncio):
    def __init__(self, iterable=None, *args, **kwargs):
        super().__init__(iterable, *args, **kwargs)
        print('Updated with custom TQDM')
        self._is_modified = True

tqdm.asyncio.tqdm_asyncio = ExtTqdm
tqdm.asyncio.tqdm = tqdm.asyncio.tqdm_asyncio
reload(tqdm)
reload(tqdm.auto)