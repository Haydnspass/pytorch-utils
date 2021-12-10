import setuptools
from setuptools import setup

requirements = []  # for now conda only

setup(
    name="pytorch_utils",
    version="0.2.0dev0",  # DO NOT MODIFY BY HAND; see DEVELOPER.md
    packages=setuptools.find_packages(),
    install_requires=requirements,
    zip_safe=False,
)
