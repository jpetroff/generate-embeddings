from typing import Any
from pathlib import Path
from dotenv import dotenv_values

def set_env_config(config: Any):
    env = dotenv_values( config.env_file )
    for key, value in env.items():
        setattr(config, key, value)

    return env