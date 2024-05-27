# # FROM jupyter/minimal-notebook:python-3.10.11
# FROM python:3.10-bullseye

# # keep env var name in sync with config_local.yml
# ARG PARTICIPANT_BUCKET_READ_SECRET
# ENV PARTICIPANT_BUCKET_READ_SECRET=${PARTICIPANT_BUCKET_READ_SECRET}

# RUN if [ -z "$PARTICIPANT_BUCKET_READ_SECRET" ]; \
#       then echo "The build arg PARTICIPANT_BUCKET_READ_SECRET must be set to non-zero, e.g. \
#       by passing the flag --build-arg PARTICIPANT_BUCKET_READ_SECRET=$PARTICIPANT_BUCKET_READ_SECRET. " &&\
#       echo "If running in CI, this variable should have been included as GH secret in the repository settings." &&\
#       echo "If you are building locally and the env var is not set,  \
#       you might find the corresponding value inside config.yml under the 'secret' key." &&\
#       exit 1; \
#       fi

# # USER root
# RUN apt-get update && apt-get upgrade -y

# # pandoc needed for docs, see https://nbsphinx.readthedocs.io/en/0.7.1/installation.html?highlight=pandoc#pandoc
# # gh-pages action uses rsync
# RUN apt-get -y install \
#       build-essential \
#       libblas-dev \
#       liblapack-dev \
#       gfortran \
#       pandoc \
#       git-lfs \
#       rsync

# # USER ${NB_UID}

# # WORKDIR /tmp
# # COPY build_scripts build_scripts
# # RUN bash build_scripts/install_presentation_requirements.sh

# COPY requirements-test.txt .
# RUN pip install -r requirements-test.txt

# # NOTE: this breaks down when requirements contain pytorch (file system too large to fit in RAM, even with 16GB)
# # NOTE: this might break down when requirements contain pytorch (file system too large to fit in RAM, even with 16GB)
# # If pytorch is a requirement, the suggested solution is to keep a requirements-docker.txt and only install
# # the lighter requirements. The install of the remaining requirements then has to happen at runtime
# # instead of build time (usually as part of the entrypoint)
# COPY requirements.txt .
# RUN pip install -r requirements.txt


# # Start of HACK: the home directory is overwritten by a mount when a jhub server is started off this image
# # Thus, we create a jovyan-owned directory to which we copy the code and then move it to the home dir as part
# # of the entrypoint
# # ENV CODE_DIR=/tmp/code

# # RUN mkdir $CODE_DIR

# # COPY --chown=${NB_UID}:${NB_GID} entrypoint.sh $CODE_DIR

# RUN chmod +x "${CODE_DIR}/"entrypoint.sh
# # Unfortunately, we cannot use ${CODE_DIR} in the ENTRYPOINT directive, so we have to hardcode it
# # Keep in sync with the value of CODE_DIR above
# ENTRYPOINT ["/tmp/code/entrypoint.sh"]

# # End of HACK

# WORKDIR "${HOME}"

# # COPY --chown=${NB_UID}:${NB_GID} . $CODE_DIR
# COPY . 

FROM python:3.10-slim-bullseye
LABEL org.opencontainers.image.source="https://github.com/aai-institute/tfl-training-practical-anomaly-detection"

# keep env var name in sync with config_local.yml
ARG PARTICIPANT_BUCKET_READ_SECRET
ENV PARTICIPANT_BUCKET_READ_SECRET=${PARTICIPANT_BUCKET_READ_SECRET}

RUN if [ -z "$PARTICIPANT_BUCKET_READ_SECRET" ]; \
      then echo "The build arg PARTICIPANT_BUCKET_READ_SECRET must be set to non-zero, e.g. \
      by passing the flag --build-arg PARTICIPANT_BUCKET_READ_SECRET=$PARTICIPANT_BUCKET_READ_SECRET. " &&\
      echo "If running in CI, this variable should have been included as GH secret in the repository settings." &&\
      echo "If you are building locally and the env var is not set,  \
      you might find the corresponding value inside config.yml under the 'secret' key." &&\
      exit 1; \
      fi

RUN apt update -y && \ 
    apt install -y \
    pkg-config libhdf5-dev build-essential \
    gobjc gfortran gnat python-dev pandoc \
    git-lfs rsync libblas-dev liblapack-dev

COPY . /tmp

WORKDIR /tmp

RUN pip install -r requirements.txt && \
    pip install click cryptography 

RUN bash ./build_scripts/install_presentation_requirements.sh
