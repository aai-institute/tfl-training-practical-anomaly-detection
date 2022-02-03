#!/bin/bash

shopt -s dotglob

if [ ! -d "${HOME}"/tfl_training_anomaly_detection ]; then
  echo "Code not found in ${HOME}, copying it during entrypoint. With jupyterhub this should happen only once"
  mv "${CODE_DIR}"/* "${HOME}"
fi

# original entrypoint, see https://github.com/jupyter/docker-stacks/blob/master/base-notebook/Dockerfile#L150
# need -s option for tini to work properly when started not as PID 1
tini -s -g -- "$@"
