import os
import shutil

# IS_TEST_CI_PIPELINE is the environment variable that we use at test time to load batches of training data and to test our notebooks.
# At inference/during the workshop, our notebooks perform intensive training and often load very big datasets.
# During tests we only want to make sure that the code runs through, and so we only run a few training steps with a minimum datasets.
# So, whenever the ci pipeline initiates the tests, we define IS_TEST_CI_PIPELINE to be True, so that all the data loading and training methods will run in test-mode and
# save computation and time.
os.environ["IS_TEST_CI_PIPELINE"] = "True"


def pytest_sessionstart():
    shutil.rmtree(os.path.join("docs", "rise.css"), ignore_errors=True)
    shutil.rmtree(os.path.join("docs", "images"), ignore_errors=True)
    os.makedirs("docs/_static", exist_ok=True)
    shutil.copy(os.path.join("notebooks", "rise.css"), "docs/_static")
    shutil.copy(os.path.join("notebooks", "rise.css"), "docs")
    shutil.copytree(
        os.path.join("notebooks", "images"),
        os.path.join("docs", "images"),
        dirs_exist_ok=True,
    )
