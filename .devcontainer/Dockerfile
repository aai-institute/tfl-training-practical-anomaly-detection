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

COPY . ${HOME}

WORKDIR ${HOME}

RUN pip install -r requirements.txt && \
    pip install click cryptography 

RUN bash ./build_scripts/install_presentation_requirements.sh
