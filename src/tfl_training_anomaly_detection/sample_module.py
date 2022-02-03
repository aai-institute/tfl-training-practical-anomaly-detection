"""
This is a top-level module
"""


class SampleClass:
    """
    Sample docstring. Note that init docstrings should be underneath the class and not the init method itself
    (this looks prettier in sphinx). They still will be rendered correctly in pycharm's quick documentation.

    :param param: some parameter
    """

    def __init__(self, param: str = None):
        self.hello = "hello "
        self.param = param

    def sample_method(self, name: str):
        """
        >>> from tfl_training_anomaly_detection.sample_module import SampleClass
        >>>
        >>> greeter = SampleClass()
        >>> greeter.sample_method("fellow TfL trainer!")
        'hello fellow TfL trainer!'

        :param name:
        :return:
        """
        return self.hello + name
