import inquirer
from typing import Any, Dict
from pathlib import Path
from llama_index.core import SimpleDirectoryReader
from modules.progress import inquirer_render

__kwargs: Dict[str, Any] = {
    "render": inquirer_render
}

def ask_before_connects(args: Any, config: Any):

    # run in silent mode without any interactive questions
    if config.non_interactive_mode == True: 
        return

    if args.config is None:
        env_file = inquirer.list_input(
            message='Choose configuration', 
            choices=[
                ('default', '.default.env'),
                ('custom (modify .env)', '.env')
            ],
            default=[ config.env_file ],
            **__kwargs
        )
        setattr(config, 'env_file', env_file)

def ask_or_get_files(args: Any, config: Any):

    if (
        args.folder is None and
        config.non_interactive_mode == False
    ):
        directory_input: str = inquirer.list_input(
            message='Directory to ingest',
            choices=[
                (f"\u2514\u2500 {config.ingest_directory}", str(config.ingest_directory) ),
                ('\u2514\u2500 append', './append'),
                ('(specify your own path to directory)', None)
                ],
            default=[ str(config.ingest_directory) ],
            **__kwargs
        )

        if not directory_input:
            directory_input = inquirer.path(
                message="Directory path",
                exists=True, 
                path_type=inquirer.Path.DIRECTORY
            )
        
        setattr(config, 'ingest_directory', Path(directory_input).resolve())

    directory_reader = SimpleDirectoryReader(
        input_dir=config.ingest_directory
    )
    files_to_ingest = directory_reader.input_files
        
    if config.non_interactive_mode == False:
        file_list = [f.name for f in files_to_ingest]
        file_list = inquirer.checkbox(
            message="Files to ingest",
            choices=file_list,
            default=file_list,
            **__kwargs
        )
        files_to_ingest = [Path(config.ingest_directory / fname).resolve() for fname in file_list]

    setattr(config, 'ingest_files', files_to_ingest)

def ask_to_confirm(args: Any, config: Any, message: str = "Proceed?", default: bool = True) -> bool:
    if config.non_interactive_mode == False:
        result= inquirer.confirm(
                message=message,
                default=default,
                **__kwargs
            )
        return result
    else: 
        return True