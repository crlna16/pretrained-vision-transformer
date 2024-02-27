# setup.py
#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="src",
    version="alpha",
    description="Pytorch Lightning Parallelism Demo",
    author="Caroline Arnold",
    author_email="carnold@posteo.com",
    url="https://github.com/crlna16",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["lightning"],
    packages=find_packages(),
)
