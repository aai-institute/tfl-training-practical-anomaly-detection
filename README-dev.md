# tfl_training_anomaly_detection - developing the training

**NOTE**: You might want to delete this file before handing the repo to the participants! 

This repository was created from an aAI internal [template](https://github.com/appliedAI-Initiative/thesane) 
for building presentations with jupyter and 
[rise](https://rise.readthedocs.io/en/stable/).

## Features

The CI pipeline will build a docker image and push it to a registry in gcc, namely to
[this path](https://console.cloud.google.com/artifacts/docker/tfl-prod-ea3b/europe-west3/tfl-docker/tfl_training_anomaly_detection?project=tfl-prod-ea3b). 
This image will be used in subsequent CI steps and for the training itself in jupyterhub.
In order for the code to be in the participant's home directory after starting a jhub server, it is
moved to it during the entrypoint (see the [Dockerfile](Dockerfile) for details).

The notebooks will be executed as part of the build, rendered to docu with `nbsphinx` and published as
gitlab pages (**Note**: currently not available). Currently, the theme
we use in nbsphinx obliterates the custom styles that we use in the presentation, so the pages will not look
as pretty as the presentation or the notebooks in jupyterhub. Still, they are useful for seeing 
whether notebooks look as you would want them to do (and whether they run through).

_IMPORTANT_: Your repository includes a branch called `solutions`, notebooks on that branch should 
be executable, i.e. without code to be filled out by participants. The docker image with code for the participants
will be built off the master branch. The CI pipeline for the solutions branch will then use this image for
executing tests. This means that you might need to update the master branch (e.g. to include new requirements)
and rerun the CI pipeline for the solutions branch after the image has been rebuilt for the tests to run through.

### Docker builds with heavy requirements

You might run into problems when trying to build an image with heavy packages, e.g. pytorch. The filesystem changes
can take too much RAM (16 GB was not enough on several occasions) and cause the build to fail.

In that case, you should create a separate file called `requirements-light.txt` and reference that one in
the Dockerfile instead of the full `requirements.txt`. The participants will need to perform a local installation
of the package anyway and can then install the heavier requirements into the container, without needing to build
an image. It is not a great solution, but it is enough for now.

## Setup

### For participants

Participants will start a jhub server based on the docker image built from the master branch. The first time
they start the server, the code will be copied to their home directory. After that they will need to install
the package (and possibly some missing requirements) with

```shell
pip install -e .
```

The way how we will distribute data is not entirely decided yet (accsr, NFS mounts, something else). If you want
to use accsr, you should distribute a _local configuration file_ called `config_local.json` with a key to
google cloud storage to the participants. Ask Mischa for more details if you don't know how to get the key. 
The accsr based solution has the benefit that data can easily be used in CI and that participants can pull data
selectively from a local setup.


### Docker based

As the docker image is being created automatically and contains all requirements, you can simply use it for
local development (e.g. using a docker interpreter in your IDE). The package is not installed in the docker image
for technical reasons (might lead to memory issues). This means you and the participants will need to run
`pip install -e .` after starting a session. Since all/most requirements already have been installed into the image, 
this should be finished very fast.

In summary, to get a running container you should do something like
```shell
docker pull europe-west3-docker.pkg.dev/tfl-prod-ea3b/tfl-docker/tfl_training_anomaly_detection
docker run -it --rm -p 8888:8888 \
  -v $(pwd)/src:/home/jovyan/src \ 
  -v $(pwd)/data:/home/jovyan/data \
  docker.aai.sh/tl/trainings/tfl_training_anomaly_detection /bin/bash
pip install -e .
```
If you don't override the entrypoint (the `/bin/bash` in the end), the container will start a notebook instead.

**NOTE**: You will need to authenticate into the docker registry before pulling. This can be done with
```shell
docker login -u _json_key -p "$(cat <credentials>.json)" https://europe-west3-docker.pkg.dev
```
Retrieve the credentials from the 
[thesan_output secrets](https://github.com/appliedAI-Initiative/thesan_output/settings/secrets/actions) 
or create a new key for yourself
in the [tfl-docker service account](https://console.cloud.google.com/iam-admin/serviceaccounts/details/113881962864414904292?project=tfl-prod-ea3b).

### Virtual environment based

If you don't want to use docker for local development, you don't have to. 
Some requirements don't work well with pip, so you will need conda.
Create a new `conda` environment, activate it and install rise-related dependencies with

```shell
bash build_scripts/install-presentation-requirements.sh
pip install -r requirements-text.txt
pip install -e .
```

## Structuring the code and testing

Source code should be in the `src` directory, tests belong inside `tests` (follow the same folder structure as `src`) and
notebooks belong in `notebooks`.

To perform tests locally use `bash build_scripts/run-tests-render-notebooks.sh`, to build the documentation use
`bash build_scripts/build-docs.sh`. Note that you *have to render notebooks first* if you want them included in the
docs. In fact, the documentation build might fail if you don't render them before. Your docs will be in
`docs/_build/html`.

The data belongs in the `data` directory, which is part of git LFS. Since data is needed for the notebooks to run, it must 
be committed. However, in order not to explode in size, data is *not included in the docker image* 
(which is why you had to mount it for the local setup). For the actual training, the data will be mounted into
the participants' containers as well. 

_IMPORTANT_: Since all participants use the same volume, the volume must be mounted read-only during
the training. Be careful with this!

## Using accsr for data

TODO. We still need to decide how we want to manage data, maybe NFS mounts on jhub will finally be fixed soon.
If you want to use accsr, you should create a github secret in the repo with the called CONFIG_LOCAL that contains the
_local_configuration_ (i.e. the key to the storage).

## Configuration and styles

There are two places for configuration: the [rise.css](rise.css) file and the notebook's metadata. You can
edit the latter either by opening the notebook as a raw document or by going to `Edit->Edit Notebook Metadata`
in jupyter's interface. They already contain sensible, aAI-conforming defaults.

**Do not use any of the other configuration options provided by rise!** (unless you really know what you are doing).

## Export to PDF
Follow the instructions on the [rise docs](https://rise.readthedocs.io/en/stable/exportpdf.html).

**NOTE**: at the moment of writing, none of the export possibilities worked on Windows WSL1.
I will try alternatives.
