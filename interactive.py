import inquirer
from typing import Any
from pathlib import Path
from llama_index.core import SimpleDirectoryReader

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
            default=[ config.env_file ]
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
                (f"./{config.ingest_directory} — should contain all documents", str(config.ingest_directory) ),
                ('./append — only new items to be added to vector storage', './append'),
                ('specify your own path to ingest directory', None)
                ],
            default=[ str(config.ingest_directory) ],
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
            message="Choose file to include or exclude",
            choices=file_list,
            default=file_list
        )
        files_to_ingest = [Path(config.ingest_directory / fname).resolve() for fname in file_list]

    setattr(config, 'ingest_files', files_to_ingest)