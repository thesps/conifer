from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="conifer",
    version="0.0.0",
    author="Sioni Summers",
    author_email="sioni@cern.ch",
    description="BDT Inference for FPGAs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thesps/conifer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    include_package_data=True,
)
