IMAGE_NAME=learners

# Clean up artifacts.
./clean.bat

# Format the code as per PEP8.
{
echo "Cleaning up the code with autopep8..."
autopep8 --in-place --aggressive --aggressive ./*.py
autopep8 --in-place --aggressive --aggressive ./learners/*.py
} ||
{
echo "WARNING: pip3 install autopep8 if you would like to format your code."
}

# Generate the distribution and install the package locally [OPTIONAL].
pip3 install --user --upgrade setuptools wheel
pip3 uninstall -y learners
python3 setup.py install --user

# Create a Docker image.
docker build --tag=$IMAGE_NAME .
