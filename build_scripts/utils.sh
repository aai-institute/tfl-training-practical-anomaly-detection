function check_notebooks_for_non_executed_load() {
  for notebook in notebooks/*.ipynb; do
    if grep -q '^[^#]*%load ' "$notebook"; then
      echo "$notebook contains a non-executed %load statement.
This is not allowed in notebooks that are supposed to be rendered!
Please make sure that all load statements are executed before calling this script.
However, also be careful to not commit the executed load statements to the repository!"
      exit 255
    fi
  done
}
