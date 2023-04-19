from pathlib import Path

import click
from tfl_training_anomaly_detection.config import default_remote_storage, get_config


@click.command()
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Whether to overwrite existing files on the remote on conflicts with local files. "
    "Without this option, will raise an error in case of a conflict and not upload anything.",
)
def upload_data(force: bool):
    c = get_config()
    data_path = Path(c.data)
    storage = default_remote_storage()
    storage.push(f"{data_path}/*", local_path_prefix=data_path.parent, force=force)


if __name__ == "__main__":
    upload_data()
