# Intro to Bayesian ML - developing the training

## Setup 

### Docker based

As the docker image is being created automatically and contains all
requirements, you can simply use it for local development (e.g. using a docker
interpreter in your IDE). The package is installed in editable mode in the
docker image, so if you mount your `src` directory when running the container,
everything should go smoothly.

In summary, to get a running container you should do something like
```shell
docker pull europe-west3-docker.pkg.dev/tfl-prod-ea3b/tfl-docker/tfl_training_anomaly_detection:latest
docker run -it --rm -p 8888:8888 \
  -v $(pwd)/src:/home/jovyan/src \ 
  -v $(pwd)/data:/home/jovyan/data \
  docker.aai.sh/tl/trainings/intro_bayesian_ml /bin/bash
```
If you don't override the entrypoint (the `/bin/bash` in the end), the container
will start a notebook instead.

### Virtual environment based

```shell
bash build_scripts/install-presentation-requirements.sh
pip install -r requirements-text.txt
pip install -e .
```

## Structuring the code and testing

Source code should be in the `src` directory, tests belong to `tests` (same
structure as in `src`) and notebooks belong to `notebooks` (no surprises there).

To perform tests locally use `bash build_scripts/run-tests-render-notebooks.sh`,
to build docu use `bash build_scripts/build-docs.sh`. Note that you *have to
render notebooks first* if you want them included in the docu. In fact, the docu
build might fail if you don't render them before. Your documentation will be in
`docs/_build/html`.

## Code formatting
The repository defines pre-commit hooks for black, isort, nbstripout, and flake.
Please install pre-commit before committing anything to the repository to ensure
that your code meets out formatting guidelines.
```shell
pre-commit install
```

You can run the formatting pipeline by typing
```shell
pre-commit run --all-files
```
### Notebooks
You can apply black formatting to your notebooks through the jupyter-black extension, which
is installed by the installation script. You can apply the formatting to selected cells through 
an icon in jupyter's web interface.

## Data
The data belongs to the `data` directory. However, in order to not explode in
size, data is *not included in the repository*. Data should generally be stored
in a common cloud storage for the project and downloaded with
[accsr](https://github.com/appliedAI-Initiative/accsr/). The repository is
already setup for the use of accsr with a default gcp bucket. See the existing
notebooks for usage examples. Please contact the repository owner for
information on how to upload data to the bucket.

## Configuration and styles

There are two places for configuration: the [rise.css](rise.css) file and the
notebook's metadata. You can edit the latter by either opening the notebook as
raw document or by going to `Edit->Edit Notebook Metadata` in jupyter's
interface. They already contain sensible defaults conforming to aAI's CI.

## Export to PDF

Follow the instructions on the [rise
documentation](https://rise.readthedocs.io/en/stable/exportpdf.html).
