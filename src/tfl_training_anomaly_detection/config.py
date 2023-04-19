from pathlib import Path

from accsr.config import ConfigProviderBase, DefaultDataConfiguration
from accsr.remote_storage import RemoteStorage, RemoteStorageConfig

root_dir = Path(__file__).parent.parent.parent.absolute()


class __Configuration(DefaultDataConfiguration):
    @property
    def remote_storage(self):
        return RemoteStorageConfig(**self._get_non_empty_entry("remote_storage_config"))

    @property
    def data(self):
        return self._get_existing_path("data")

    @property
    def housing_data(self):
        return self._get_existing_path("housing_data")


class ConfigProvider(ConfigProviderBase[__Configuration]):
    pass


_config_provider = ConfigProvider()


def get_config(reload=True, ignore_local=False) -> __Configuration:
    """
    :param ignore_local: if True, the local configuration file will be ignored.
        Since only the config.yml file is delivered to participants with the zip package,
        setting this to True is useful for inspecting the delivered configuration prior to building the package.
    :param reload: if True, the configuration will be reloaded from the json files
    :return: the configuration instance
    """
    config_files = ["config.yml"]
    if not ignore_local:
        config_files.append("config_local.yml")
    return _config_provider.get_config(
        config_directory=root_dir,
        config_files=config_files,
        reload=reload,
    )


__remote_storage_instance = None


def default_remote_storage():
    global __remote_storage_instance
    if __remote_storage_instance is None:
        __remote_storage_instance = RemoteStorage(get_config().remote_storage)
    return __remote_storage_instance
