#!/usr/bin/env python

from setuptools import setup

setup(
    # Package name
    name="tf_idf",

    # Release version
    version="1.0",

    # Description
    description="Simple TF-IDF lib",
    long_description="A simple TF-IDF implementation for quick and easy \
    experimentation",

    # Author/Maintainer info
    author="José Mendonça",
    author_email="jnl.mendonca@gmail.com",
    maintainer="José Mendonça",
    maintainer_email="jnl.mendonca@gmail.com",

    # Package information
    py_modules=["tf_idf"],

    # Extra information
    url="https://github.com/jnlmendonca/tf-idf",
    license="MIT"
)
