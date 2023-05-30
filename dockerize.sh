#!/bin/sh

PACKAGE=mlo
CONTAINER=learners

# Clean up artifacts.
./clean.sh

# Format the code as per PEP8.
{
echo "Cleaning up the code with autopep8..."
autopep8 --in-place --aggressive --aggressive ./*.py
autopep8 --in-place --aggressive --aggressive ./$PACKAGE/*.py
} ||
{
echo "WARNING: pip3 install autopep8 if you would like to format your code."
}

# Create a Docker image.
docker build -t $CONTAINER .

# Generate the distribution and install the package locally [OPTIONAL].
python3 -m pip install --upgrade build
pip3 uninstall -y $PACKAGE
python3 -m build
python3 -m pip install --user .

# Push to PyPi: https://packaging.python.org/en/latest/tutorials/packaging-projects/
