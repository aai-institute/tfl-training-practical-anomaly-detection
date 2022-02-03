import os
import shutil


def pytest_sessionstart():
    shutil.rmtree(os.path.join("docs", "rise.css"), ignore_errors=True)
    shutil.rmtree(os.path.join("docs", "images"), ignore_errors=True)
    os.makedirs("docs/_static", exist_ok=True)
    shutil.copy(os.path.join("notebooks", "rise.css"), "docs/_static")
    shutil.copy(os.path.join("notebooks", "rise.css"), "docs")
    shutil.copytree(os.path.join("notebooks", "images"), os.path.join("docs", "images"), dirs_exist_ok=True)
