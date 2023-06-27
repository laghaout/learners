#!/bin/sh

# Set the environment variables.
. ./.env

# Run all the hooks.
pre-commit run

# Run tests.
pytest

# Clean up artifacts.
./clean.sh

# Dockerize.
docker compose build

# Generate the distribution and install the package locally.
python3 -m pip install --upgrade build
pip3 uninstall -y $PACKAGE_NAME
python3 -m build
python3 -m pip install --user .

# Push to PyPi: https://packaging.python.org/en/latest/tutorials/packaging-projects/

# Clean up artifacts.
./clean.sh
