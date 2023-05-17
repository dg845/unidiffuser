import os

from setuptools import setup, find_packages

setup(
    name="unidiffuser",
    version="1.0",
    description="",
    author="thuml",
    packages=find_packages(),
    install_requires=[
        "accelerate == 0.12.0",
        "absl-py",
        "ml_collections",
        "einops",
        "ftfy == 6.1.1",
        "transformers == 4.23.1",
    ],
    include_package_data=True,
)