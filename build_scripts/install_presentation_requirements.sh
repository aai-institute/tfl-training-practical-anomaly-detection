#!/usr/bin/env bash

set -euo pipefail

function usage() {
  cat > /dev/stdout <<EOF
Usage:
  install_presentation_requirements.sh [FLAGS]

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
  conda install -y -c conda-forge notebook==6.5.6 jupyter rise==5.7.1 sphinx sphinx_rtd_theme jupyter-book jupyter_contrib_nbextensions jupyter_nbextensions_configurator
  python build_scripts/configure_spellcheck_dict.py
  jupyter contrib nbextension install --user
  jupyter nbextensions_configurator enable --user
  jupyter nbextension enable spellchecker/main
  jupyter nbextension enable equation-numbering/main
  jupyter nbextension enable toc2/main
  jupyter nbextension enable hinterland/hinterland
  jupyter nbextension enable hide_input/main
  jupyter nbextension enable init_cell/main
  jupyter nbextension enable exercise2/main
)
