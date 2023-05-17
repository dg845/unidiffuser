import os

from setuptools import setup, find_packages

setup(
    name="unidiffuser",
    py_modules=["unidiffuser"],
    version="1.0",
    description="",
    author="thuml",
    package_dir={"": "src"},
    packages=find_packages("src"),
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