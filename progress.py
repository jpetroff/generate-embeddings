from typing import (
    Optional
)

from rich.console import Console, Group
from rich.status import Status
from rich.progress import Progress, TaskID
from rich.live import Live

class ProgressRelay:

    console: Console
    status: Status
    progress: Progress
    live: Live
    default_task: TaskID

    def __init__(self,
        **kwargs
    ):
        pass

    def init_step_context(self,
        console: Console,
        has_progress: bool = False,
        total: Optional[int] = None, 
        status: str = '',
        **kwargs
    ):
        self.console = console
        self.status = Status(status, **kwargs)
        self.progress = Progress(**kwargs)
        status_group= Group(
            self.status,
            self.progress
        )
        self.live = Live(
            status_group,
            console=self.console,
            transient=True
        )
        self.live.start()

    def end_step_context(self):
        self.total = None
        self.live.stop()

    def update_status(self, status: str, **kwargs):
        self.status.update(status=status, **kwargs)

    def start_task(self, 
        description: str = '', 
        total: Optional[int] = None,
        **kwargs
    ):
        self.default_task = self.progress.add_task(
            description=description,
            total=total
        )

    def advance_progress(self):
        self.progress.advance(self.default_task)

    def end_task(self):
        self.progress.remove_task(self.default_task)

global_console = Console()
progress_relay = ProgressRelay()