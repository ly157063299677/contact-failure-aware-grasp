from pathlib import Path
import tomllib


def load_config(path):
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    return config
