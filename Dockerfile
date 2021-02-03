# Inherit from python 3.8 image.
FROM python:3.8-slim

# Declare the working directory.
WORKDIR /learners

# Update to the latest version of pip.
RUN pip install --upgrade pip

# Install requirements.txt.
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt update
RUN apt -y install emacs-nox
RUN apt -y install less
RUN apt -y install tk
RUN apt -y install tree

# Install the package.
COPY setup.py .
COPY learners/ learners
RUN pip install .
COPY main.py .

# Default command upon running the container.
#CMD ["bash", ". <some launcher>.bat"]
