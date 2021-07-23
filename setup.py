#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="keras-video-generators",
    version="1.1.0",
    description="Keras sequence generators for video data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    licence_file="LICENSE",
    author="Patrice Ferlet",
    author_email="metal3d@gmail.com",
    url="https://github.com/metal3d/keras-video-generators",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["tensorflow>=2.5", "numpy", "opencv-python", "matplotlib"],
)
