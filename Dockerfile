FROM jupyter/minimal-notebook:python-3.9.7

ARG cfg_local


USER root
RUN apt-get update -yq && apt-get -yq install ffmpeg

# pandoc needed for docu, see https://nbsphinx.readthedocs.io/en/0.7.1/installation.html?highlight=pandoc#pandoc
# gh-pages action uses rsync
RUN apt-get -y install pandoc git-lfs rsync

USER ${NB_UID}

# Start of HACK: the home directory is overwritten by a mount when a jhub server is started off this image
# Thus, we create a jovyan-owned directory to which we copy the code and then move it to the home dir as part
# of the entrypoint
ENV CODE_DIR=/tmp/code

RUN mkdir $CODE_DIR

COPY --chown=${NB_UID}:${NB_GID} entrypoint.sh $CODE_DIR

RUN chmod +x /tmp/code/entrypoint.sh
ENTRYPOINT ["/tmp/code/entrypoint.sh"]

# End of HACK
WORKDIR /tmp
COPY build_scripts build_scripts
RUN bash build_scripts/install-presentation-requirements.sh

COPY requirements-test.txt .
RUN pip install -r requirements-test.txt

# NOTE: this breaks down when requirements contain pytorch (file system too large to fit in RAM, even with 16GB)
# If pytorch is a requirement, the suggested solution is to keep a requirements-docker.txt and only install
# the lighter requirements. The install of the remaining requirements then has to happen at runtime
# instead of build time...
COPY requirements.txt .
RUN pip install -r requirements.txt

WORKDIR "${HOME}"

COPY --chown=${NB_UID}:${NB_GID} . $CODE_DIR

RUN echo ${CFG_LOCAL} > ./config_local.json
RUN echo ${CFG_LOCAL}
