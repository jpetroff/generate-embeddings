import argparse
from typing import Any
from pathlib import Path

parser = argparse.ArgumentParser(prog="__main__.py")
parser.add_argument(
    "--defaults", '-d',
    action="store_true",
    help="Run in default mode without interactive queries",
    default=False
)
parser.add_argument(
    "--folder", '-f',
    type=str,
    help="Folder to ingest",
    default=None
)
parser.add_argument(
    "--ignore", "-i",
    nargs="*",
    action="extend", 
    type=str,
    help="List of files/directories to ignore",
    default=[],
)
parser.add_argument(
    "--config", '-c',
    type=str,
    help="Specific embedding configuration to run",
    default=None,
)

args = parser.parse_args()

def set_arg_config(config: Any):
    if args.config is not None:
        setattr(config, 'env_file', Path(args.config))

    if args.folder is not None:
        setattr(config, 'ingest_directory', Path(args.folder))
    
    if args.ignore is not None:
        setattr(config, 'ignored_files', args.ignore)

    setattr(config, 'non_interactive_mode', args.defaults)

    