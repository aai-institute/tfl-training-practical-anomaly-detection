from setuptools import find_packages, setup


def read_requirements(filename: str):
    return [line for line in open(filename).readlines() if not line.startswith("--")]


setup(
    name="tfl_training_anomaly_detection",
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    version="1.0.0",
    description="Transferlab training",
    install_requires=read_requirements("requirements.txt"),
    author="appliedAI Transferlab",
)
