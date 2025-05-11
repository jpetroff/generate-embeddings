from enum import Enum
from typing import (
    Optional,
    Any
)

from rich.console import Console, Group
from rich.status import Status
from rich.progress import Progress, TaskID, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.live import Live

from time import time

class OP_CODE(str, Enum):
    SUCCESS = 'ended:success'
    SKIPPED = 'ended:skipped'
    PARTIAL = 'ended:partial'
    ERROR = 'ended:error'
    PROGRESS = 'progress'


class ProgressRelay:

    console: Console
    status: Status
    progress: Progress
    live: Live
    default_task: TaskID
    append: str = ''
    is_active_task: bool = False
    step_time: float = 0

    def __init__(self,
        **kwargs
    ):
        pass

    def init_step_context(self,
        console: Console,
        status: str = '',
        append: str = '',
        **kwargs
    ):
        self.console = console
        self.append = append
        self.is_active_task = True
        self.step_time = time()

        self.status = Status(f"{status} {self.append}", **kwargs)
        self.progress = Progress(
            TextColumn("  {task.description}"),
            BarColumn(complete_style="green", finished_style="bold green"),
            MofNCompleteColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
            **kwargs
        )
        status_group= Group(
            self.status,
            self.progress
        )
        self.live = Live(
            status_group,
            console=self.console,
            transient=True,
            refresh_per_second=10
        )
        self.live.start()

    def end_step_context(self, message: str = '', **kwargs):
        self.total = None
        self.is_active_task = False
        self.append = ''
        self.live.stop()
        _elapsed_time = round(time() - self.step_time, 2)
        if message != '':
            self.console.print(f"{message} [dim]{_elapsed_time}s[/]")
        return _elapsed_time

    def update_status(self, status: str, **kwargs):
        self.status.update(status=f"{status} {self.append}", **kwargs)

    def start_task(self, 
        description: str = '', 
        total: Optional[int] = None,
        **kwargs
    ):
        self.default_task = self.progress.add_task(
            description=description,
            total=total,
            **kwargs
        )

    def advance_progress(self, result: Any = None, **kwargs):
        task_id = kwargs['task_id'] if hasattr(kwargs, 'task_id') else self.default_task
        self.progress.advance(task_id, **kwargs)

    def end_task(self, message: str = '', **kwargs):
        task_id = kwargs['task_id'] if hasattr(kwargs, 'task_id') else self.default_task
        if message != '':
            self.progress.print(f"{message} [dim]{round(time() - self.step_time, 2)}s[/]")
        self.progress.remove_task(task_id)


"""
Initialize global console and progress objects
"""
global_console = Console(highlight=False)
progress_relay = ProgressRelay()


"""
Custom Inquirer theme
"""
from inquirer.themes import (
    Default as DefaultTheme,
    term,
)

class CustomTheme(DefaultTheme):
    def __init__(self):
        super().__init__()
        self.Question.brackets_color = term.lightsteelblue3 # type: ignore
        self.Question.mark_color = term.lightsteelblue3 # type: ignore
        # self.Checkbox.selection_color = term.bold_white_on_blue # type: ignore
        self.Checkbox.selection_color = term.bright_white_on_blue # type: ignore
        self.Checkbox.selection_icon = " " # type: ignore
        self.Checkbox.selected_icon = "[X]" # type: ignore
        self.Checkbox.selected_color = term.blue # type: ignore
        self.Checkbox.unselected_icon = "[ ]" # type: ignore
        # self.List.selection_color = term.bold_white_on_bright_blue # type: ignore
        self.List.selection_color = term.bright_white_on_blue # type: ignore
        self.List.selection_cursor = " " # type: ignore

from inquirer.render import ConsoleRender

inquirer_render = ConsoleRender(theme=CustomTheme()) # type: ignore