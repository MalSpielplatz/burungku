from setuptools import find_packages, setup

NAME = "rcfx"
VERSION = "0.0.1"
AUTHOR = "Yassine Alouini"
DESCRIPTION = """The repo for the RCFX audio app."""
EMAIL = "yassinealouini@outlook.com"
URL = ""

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    # Some metadata
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    url=URL,
    license="MIT",
    keywords="kaggle machine-learning audio deep-learning",
)
