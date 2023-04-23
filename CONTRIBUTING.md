# Practical Anomaly Detection - developing the training

This repository was created from an aAI internal
[template](https://github.com/appliedAI-Initiative/thesan)


## Overview of the features

When you create a new repository from the template, you will get the following
features:

1. An example notebook, fully set up for presentation and participants.
It contains a lot of documentation, so you can start by having a look at it.
2. A CI pipeline that will build a docker image and push it to our registry.
3. The source code is installable, you can run `pip install -e ".[test]"`.
4. There are bash scripts for helping with various build steps in 
`build_scripts`. They are all documented, so feel free to call
them with the `-h` flag. For a local setup, call 
`bash build_scripts/install_presentation_requirements.sh`
5. The notebooks are rendered to documentation and published as GH pages.
They are also rendered to a jupyter-book. The rendering happens as part of the
`build_scripts/build_docs.sh` script.
6. Data is downloaded with accsr and stored in the `data` directory. This
happens on demand locally, and as part of the entrypoint in the docker container.

## Writing new content

Start by having a thorough read through `nb_10_PrsentationTemplate.ipynb`.
Since IDEs don't support all the features of jupyter, it is
recommended to view it through `jupyter notebook`.

For creating a new notebook, the easiest way is to copy the `nb_20_CopyMe.ipynb` notebook
and to modify it. You should remember the following things:

1. Keep the naming consistent, i.e. `nb_10_...`, `nb_20_...`, etc.
2. Make sure to set up the slideshow metadata in the cells correctly.
3. You can remove cell input and output or remove entire cells from the 
rendered html using tags.
Note that removed cells are not executed during the render, so if you want
to remove something like an import cell, you can add the tags
`remove-input` and `remove-output` (but not `remove-cell`).
4. You can declare cells as initialization cells, and you can hide them
from the viewer, see the example notebook for details.
5. The tag `hide-cell` is ignored by `nbocnvert` but used by jupyter-book.
It is useful for import cells. You can also consider using it for the
hint-loading cells.
6. Extend `notebooks/intro.md` as you see fit. It is part of the jupyter-book.

## Rendering and committing notebooks

Notebooks are a bit tricky. We don't want to execute them in CI, because that
would take too long. We also don't want to commit the output, because it would
make the repository too big. So we have to render them locally and commit the
rendered html. On top of that, dynamically loading code cells with the `%load`
magic command makes things even more complicated.

To render the notebooks, you can use 
`build_scripts/build_docs.sh --execute`.
It will create files in `docs/_build/html` (with sphinx)
as well as in `notebooks/_build/html` (with jupyter-book).

**IMPORTANT 1**: With the `--execute` option, the above script will also create 
files inside `docs/_static`. These files have to be **committed!** Don't worry
about them blowing up the repository, they are stored in git lfs.

**IMPORTANT 2**: We use the `%load` magic command for loading code
cells dynamically. Unfortunately, this collides with rendering.
The current workaround is to first execute the cell containing `%load`
before rendering the notebook. Note however, that the 
executed cells should **not be committed!** This means, for rendering
and committing the html in `docs/_static`, you have to first execute
the `%load` cells, then render the notebooks, and then revert the
changes to the notebooks. The notebooks are validated before render and before
the docker image is built, so you will get an error if you forget to do this.

## Before the training

Make sure everything is committed as described above
and pushed to the remote repository. Ask the operations team to
create a jupyterhub server and to include the docker image that was
built in CI into the list.

Start an instance and check whether:

1. All notebooks can be executed.
2. All slideshows look good.

Note that the data is pulled in the background when you first open
the training instance, so it might take a while. This is usually not a
problem because in the first minutes of the training there is a presentation
anyway.

## Preparing the package

After the training is finished, you should distribute the code to the
participants.
We distribute a zip file containing source code, 
a read access key for the data, and the rendered html files.

To create the package, you should:

1. Replace the reference to the env var by the actual value in the `secret`
entry in `config.yml`. **IMPORTANT**: only give out the read access key to the
participants!
2. Execute all `%load` cells in the notebooks.
3. Render the docs as described above. Have a look at the rendered html files,
both in `docs/_build/html` and `notebooks/_build/html`. If you are happy with
the result, proceed.
4. Run `build_scripts/prepare_zip_package.sh`. It will create a zip file.
5. Send the package to the participants per e-mail.

## Notes on CI Pipelines

The CI pipeline will build a docker image and push it to a registry in gcc,
namely to [this
path](https://console.cloud.google.com/artifacts/docker/tfl-prod-ea3b/europe-west3/tfl-docker/thesan_output?project=tfl-prod-ea3b).
This image will be used in subsequent CI steps and for the training itself in
jupyterhub. In order for the code to be in the participant's home directory
after starting a jhub server, it is moved to it during the entrypoint (see the
[Dockerfile](Dockerfile) for details).

The notebooks will be executed as part of the build, rendered to documentation
and published as gitlab pages (**Note**: currently not available). Currently,
the theme we use in nbsphinx obliterates the custom styles that we use in the
presentation, so the pages will not look as pretty as the presentation or the
notebooks in jupyterhub. Still, they are useful for seeing whether notebooks
look as you would want them to do (and whether they run through).

### Docker builds with heavy requirements

You might run into problems when trying to build an image with heavy packages,
e.g. pytorch. The filesystem changes can take too much RAM (16 GB was not enough
on several occasions) and cause the build to fail.

In that case, you should create a separate file called `requirements-light.txt`
and reference that one in the Dockerfile instead of the full `requirements.txt`.
The participants will need to perform a local installation of the package anyway
and can then install the heavier requirements into the container, without
needing to build an image. It is not a great solution, but it is enough for now.

## Setup

### Accessing and pushing data

The configuration is already set up to work out of the box, you
just need to set the env var `TFL_STORAGE_ADMIN_SECRET`. You can upload
data by calling `python scripts/upload_data.py`.

### Virtual environment based

Some requirements don't work well with pip, so you will need conda. Create a new
`conda` environment, activate it and install dependencies with

```shell 
bash build_scripts/install_presentation_requirements.sh 
pip install -e ".[test]"  
```

### Docker based

As the docker image is being created automatically and contains all
requirements, you can simply use it for local development (e.g. using a docker
interpreter in your IDE).

In summary, to get a running container you could do something like 
```shell
docker pull europe-west3-docker.pkg.dev/tfl-prod-ea3b/tfl-docker/thesan_output
docker run -it -p 8888:8888 --entrypoint bash \ 
-v $(pwd):/home/jovyan/thesan_output \
docker.aai.sh/tl/trainings/thesan_output  
``` 

**NOTE**: You will need to authenticate into the docker registry before pulling.
This can be done with
```shell 
docker login -u _json_key -p "$(cat <credentials>.json)" https://europe-west3-docker.pkg.dev 
``` 
Ask somebody for the credentials.

You can also build the image locally. Set the `PARTICIPANT_BUCKET_READ_SECRET`
environment variable and run

```shell
docker build --build-arg PARTICIPANT_BUCKET_READ_SECRET=$PARTICIPANT_BUCKET_READ_SECRET -t thesan_output .
```

### For participants

Participants will start a jhub server based on the docker image built from the
master branch. For them, everything should work out of the box.

## Structuring the code and testing

Source code should be in the `src` directory, tests belong inside `tests`
(follow the same folder structure as `src`) and notebooks belong in `notebooks`.


## Configuration and styles

There are two places for configuration: the [rise.css](rise.css) file and the
notebook's metadata. You can edit the latter either by opening the notebook as a
raw document or by going to `Edit->Edit Notebook Metadata` in jupyter's
interface. They already contain sensible, aAI-conforming defaults.

**Do not use any of the other configuration options provided by rise!** (unless
you really know what you are doing).

