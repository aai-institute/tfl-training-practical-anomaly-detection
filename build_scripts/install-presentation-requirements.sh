#!/usr/bin/env bash

set -euo pipefail

function usage() {
  cat > /dev/stdout <<EOF
Usage:
  install-presentation-requirements.sh [FLAGS]

  Installs and enables rise, spellchecker and other tools required for the presentation

  Optional flags:
    -h, --help              Show this information and exit
EOF
}

while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
      -h|--help)
        usage
        exit 0
      ;;
      -*)
        >&2 echo "Unknown option: $1"
        usage
        exit 255
      ;;
      *)
        >&2 echo "This script takes no positional arguments but got: $1"
        exit 255
      ;;  esac
done

BUILD_DIR=$(dirname "$0")

(
  cd "${BUILD_DIR}/.." || (echo "Unknown error, could not find directory ${BUILD_DIR}" && exit 255)
  conda install -y -c conda-forge notebook rise jupyter_contrib_nbextensions
  #python build_scripts/configure_spellcheck_dict.py
  jupyter nbextension enable spellchecker/main
  jupyter  nbextension install https://github.com/drillan/jupyter-black/archive/master.zip --user
  jupyter nbextension enable jupyter-black-master/jupyter-black
  jupyter nbextension enable equation-numbering/main
  jupyter nbextension install --py hide_code
  jupyter nbextension enable --py hide_code
  jupyter serverextension enable --py hide_code
)


