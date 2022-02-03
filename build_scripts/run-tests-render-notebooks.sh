#!/usr/bin/env bash

set -euo pipefail

function usage() {
  cat > /dev/stdout <<EOF
Usage:
  run-tests-render-notebooks.sh [FLAGS]

  Runs unit tests and executes notebooks, moves rendered notebooks to docs

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
  pytest
# IMPORTANT: this is flaky, sometimes the parallel execution can cause a KernelDied error
# this is due to the following unresolved issue: https://github.com/jupyter/nbconvert/issues/1066
# The current "solution" is to just restart the test execution / CI pipeline
  pytest -n auto notebooks
)
