from setuptools.config import read_configuration
from setuptools import setup, find_packages

setup(
    name="rbm",
    version="1.0.0",
    packages=find_packages(),
    install_requires=['python_version >= "3.9.7"'],
)
