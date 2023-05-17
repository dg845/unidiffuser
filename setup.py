import os

from setuptools import setup, find_packages

setup(
    name="unidiffuser",
    version="1.0",
    description="",
    author="thuml",
    packages=find_packages(),
    python_requires="==3.9",
    install_requires=[
        "accelerate == 0.12.0",
        "absl-py",
        "ml_collections",
        "einops",
        "ftfy == 6.1.1",
        "transformers == 4.23.1",
        "torch",
        "torchvision",
        "clip @ git+https://github.com/openai/CLIP.git@main#egg=clip",
    ],
    include_package_data=True,
)