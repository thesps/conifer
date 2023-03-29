from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="conifer",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
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
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'pybind11',
        'packaging',
    ],
    include_package_data=True,
)
