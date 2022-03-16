import os

from tfl_training_anomaly_detection.sample_module import SampleClass

os.environ["is_test"] = "True"

# the suggested naming convention for unit tests is test_method_name_testDescriptionInCamelCase
# this leads to a nicely readable output of pytest
def test_sample_class_attributes_greeterSaysHello():
    greeter = SampleClass()
    assert greeter.hello == "hello "
