#!/bin/sh

PACKAGE=mlo
CONTAINER=learners

# Clean up artifacts.
./clean.sh

# Run tests.
pytest

# Create a Docker image.
docker build -t $CONTAINER .

# Generate the distribution and install the package locally [OPTIONAL].
python3 -m pip install --upgrade build
pip3 uninstall -y $PACKAGE
python3 -m build
python3 -m pip install --user .

# Push to PyPi: https://packaging.python.org/en/latest/tutorials/packaging-projects/
