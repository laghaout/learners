IMAGE_NAME=learners

# Clean up artifacts from emacs.
./clean.bat

# Format the code as per PEP8.
{
echo "Cleaning up the code with autopep8..."
autopep8 --in-place --aggressive --aggressive ./*.py
autopep8 --in-place --aggressive --aggressive ./learners/*.py
} ||
{
echo "WARNING: pip install autopep8 if you would like to format your code."
}

# Stop and remove all containers.
docker stop $(docker container ls -q -a)
docker rm $(docker container ls -q -a)

# Remove all dangling images.
docker rmi $(docker images -f dangling=true -q)

# Generate the distribution.
pip install --user --upgrade setuptools wheel
python setup.py sdist bdist_wheel

# Install the package locally. [OPTIONAL]
#pip install dist/*

# Create a Docker image.
docker build --tag=$IMAGE_NAME .
