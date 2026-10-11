#!/usr/bin/env python

from setuptools import find_packages, setup

# read the contents of your README file
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="pinnstorch",
    version="0.2.0",
    description="An implementation of PINNs in PyTorch using typed problem definitions and Lightning.",
    author="Reza Akbarian Bafghi",
    author_email="reza.akbarianbafghi@coloardo.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AridanWarlock/pinn-runner",
    license="BSD-3-Clause",
    python_requires=">=3.10",
    install_requires=[
        "torch==2.1.0",
        "numpy==1.26.4",
        "lightning==2.1.0",
        "pytorch-lightning==2.1.0",
        "torchmetrics==1.2.0",
        "PyYAML==6.0.3",
        "scipy==1.11.4",
        "pyDOE==0.3.8",
        "setuptools==80.9.0",
    ],
    packages=find_packages(include=["pinnstorch", "pinnstorch.*"]),
    # use this to customize global commands available in the terminal after installing the package
    include_package_data=True,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.10",
    ],
)
