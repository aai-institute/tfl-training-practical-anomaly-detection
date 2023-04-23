#!/usr/bin/env bash

set -euo pipefail

function usage() {
  cat > /dev/stdout <<EOF
Usage:
  build_docs.sh [FLAGS]

  Updates and builds the documentation. In order to include the notebooks into the docu,
  it is recommended to execute the build script run-all-tests.sh first.

  Optional flags:
    -h, --help              Show this information and exit
    --execute               Execute the notebooks before building the docs. If this flag is not set,
                            the notebooks are converted to html as is.
    --cleanup               Clean up html files in the docs/_static directory before building the docs.
                            Use with caution, these files are typically committed to the repository.
    --skip-nbconvert        Skip the jupyter notebook conversion step. This is useful if you only want to update the
                            docs rst files and not the notebook html files.
    --skip-jupyter-book     Skip the jupyter book build step
    --skip-nb-validation    Skip the validation of the notebooks. You should only use this if you know what you
                            are doing.
EOF
}


EXECUTE_FLAG=""
CLEANUP=false
RUN_NBCONVERT=true
RUN_JUPYTER_BOOK=true
VALIDATE_NB=true

while [[ $# -gt 0 ]]
do
  key="$1"
  case $key in
      --execute)
        EXECUTE_FLAG="--execute"
        shift
      ;;
      --cleanup)
        CLEANUP=true
        shift
      ;;
      --skip-nbconvert)
        RUN_NBCONVERT=false
        shift
      ;;
      --skip-jupyter-book)
        RUN_JUPYTER_BOOK=false
        shift
      ;;
      --skip-nb-validation)
        VALIDATE_NB=false
        shift
      ;;
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
DOCS_STATIC_DIR="docs/_static"

(
  cd "${BUILD_DIR}/.." || (echo "Unknown error, could not find directory ${BUILD_DIR}" && exit 255)
  source build_scripts/utils.sh

  echo "Generating .rst files from source code modules"
  python build_scripts/update_docs.py

  if [ "$CLEANUP" = true ] ; then
    echo "Cleaning up html files in ${DOCS_STATIC_DIR}"
    rm -r $DOCS_STATIC_DIR/*.html 2>/dev/null || true
  fi

  if [ "${RUN_NBCONVERT}" = true ] ; then

    if [ "$VALIDATE_NB" = true ] ; then
      # from utils.sh
      check_notebooks_for_non_executed_load
    else
      echo "Skipping notebook validation step"
    fi

    echo "Generating static html files from notebooks with nbconvert..."
    if [ "$EXECUTE_FLAG" = "--execute" ] ; then
      echo "Executing notebooks before converting them to html."
    fi
    mkdir -p $DOCS_STATIC_DIR
    jupyter nbconvert --to html notebooks/*.ipynb ${EXECUTE_FLAG}\
    --output-dir $DOCS_STATIC_DIR \
    --TagRemovePreprocessor.enabled True \
    --TagRemovePreprocessor.remove_cell_tags remove-cell-nbconv \
    --TagRemovePreprocessor.remove_input_tags remove-input-nbconv \
    --TagRemovePreprocessor.remove_all_outputs_tags remove-output-nbconv \
    --no-prompt \
    --template classic
    echo "...done, your notebooks are now available as html in the docs/_static directory."
    echo "Note that the images are not copied over there but instead copied during the docs
build in the next step, so the html files are potentially not self-contained."
  else

    echo "Skipping nbconvert step. Validating html files..."
      NUM_NBS=$(find notebooks/*.ipynb | wc -l)
      NUM_RENDERED_NBS=$(find docs/_static/*.html | wc -l)

      if [ "$NUM_NBS" -ne "$NUM_RENDERED_NBS" ]; then
        echo "The number of notebooks in notebooks/ does not match the number
of rendered notebooks in docs/_static/: ${NUM_NBS} != ${NUM_RENDERED_NBS}.
Consider removing the --skip-nbconvert flag and adding --execute to create
rendered notebooks."
        exit 255
      fi
  fi

  if [ "$RUN_JUPYTER_BOOK" = true ] ; then
    echo "Building jupyter book"
    if [ "$VALIDATE_NB" = true ] ; then
      check_notebooks_for_non_executed_load
    else
      echo "Skipping notebook validation step"
    fi

    jupyter-book build notebooks
  else
    echo "Skipping jupyter book build step"
  fi

  echo "Searching for built jupyter book"
  if [ -f "notebooks/_build/html/index.html" ]; then
    echo "Moving the built jupyter book to the docs directory"
    rm -r docs/_static/jupyter_book 2>/dev/null || true
    mkdir -p docs/_static/jupyter_book
    cp -r notebooks/_build/html/* docs/_static/jupyter_book/
  else
    echo "Could not find the built jupyter book in notebooks/_build/html"
    echo "This is normal in CI, as the built files may be committed inside the docs"
  fi

  echo "Building documentation with sphinx"
  sphinx-build -W -b html -d "temp/doctrees" docs "docs/_build/html"
  echo "Running doctests with sphinx"
  sphinx-build -b doctest -d "temp/doctrees" docs "docs/_build/doctest"
)
