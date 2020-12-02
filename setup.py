"""Install packages as defined in this file into the Python environment."""
from setuptools import setup, find_namespace_packages

# The version of this tool is based on the following steps:
# https://packaging.python.org/guides/single-sourcing-package-version/
VERSION = {}

with open("./src/hff_predictor/__init__.py") as fp:
    # pylint: disable=W0122
    exec(fp.read(), VERSION)

setup(
    name="hff_predictor",
    author="Cornelis Vletter",
    author_email="vletter.data.advies@gmail.com",
    description="HFF prediction.",
    version=VERSION.get("__version__", "1.0.0"),
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "setuptools>=45.0",
    ],
    entry_points={
        "console_scripts": [
            "hff=hff_predictor.__main__:main",
        ]
    },
)
