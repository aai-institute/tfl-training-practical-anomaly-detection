import logging
import os

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOKS_DIR = "notebooks"
DOCS_DIR = "docs"
resources = {"metadata": {"path": NOTEBOOKS_DIR}}

log = logging.getLogger(__name__)

OMITTED_NOTEBOOKS = [
    # Should not be run by pipeline
    "00-setup-and-info.ipynb",
    # HTML(camera.animate().to_html5_video())
    # Requested MovieWriter (ffmpeg) not available
    "extreme_value_theory_for_anomaly_detection.ipynb",
    # FileNotFoundError: [Errno 2] No such file or directory: '../data/numenta/NAB/data/realKnownCause/nyc_taxi.csv'
    "anomaly_detection_on_time_series.ipynb",
    # Timeout
    "anomaly_detection_approaches.ipynb"
]

@pytest.mark.parametrize(
    "notebook",
    [
        file
        for file in os.listdir(NOTEBOOKS_DIR)
        if file.endswith(".ipynb") and file not in OMITTED_NOTEBOOKS
    ],
)
def test_notebook(notebook):
    notebook_path = os.path.join(NOTEBOOKS_DIR, notebook)
    log.info(f"Reading jupyter notebook from {notebook_path}")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, resources)

    # saving the executed notebook to docs
    output_path = os.path.join(DOCS_DIR, notebook)
    log.info(f"Saving executed notebook to {output_path} for documentation purposes")
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
