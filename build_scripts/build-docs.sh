#!/usr/bin/env bash

set -euo pipefail

function usage() {
  cat > /dev/stdout <<EOF
Usage:
  build-docs.sh [FLAGS]

  Updates and builds the documentation. In order to include the notebooks into the docu,
  it is recommended to execute the build script run-all-tests.sh first.

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
      ;;
  esac
done

BUILD_DIR=$(dirname "$0")

(
  cd "${BUILD_DIR}/.." || (echo "Unknown error, could not find directory ${BUILD_DIR}" && exit 255)
  python build_scripts/update_docs.py
  sphinx-build -b html -d "temp/doctrees" docs "docs/_build/html"
  sphinx-build -b doctest -d "temp/doctrees" docs "docs/_build/doctest"
)
