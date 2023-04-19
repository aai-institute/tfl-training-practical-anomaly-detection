from pathlib import Path

import click
from tfl_training_anomaly_detection.config import default_remote_storage, get_config


@click.command()
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Whether to overwrite existing local files on conflicts with files on the remote. "
    "Without this option, will raise an error in case of a conflict and not download anything.",
)
def download_data(force: bool):
    c = get_config()
    data_path = Path(c.data)
    storage = default_remote_storage()
    storage.pull(data_path.name, local_base_dir=data_path.parent, force=force)


if __name__ == "__main__":
    download_data()
