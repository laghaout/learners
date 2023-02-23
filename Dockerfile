# Inherit from the Python image.
FROM python:3.10-slim

# Declare the working directory.
WORKDIR /learners

# Install Linux utilities.
RUN apt -y update
RUN apt -y upgrade
RUN apt -y install emacs-nox
RUN apt -y install less
RUN apt -y install tk
RUN apt -y install tree

# Install Python packages.
RUN python3 -m pip install --upgrade pip
RUN pip3 install --user --upgrade pip
RUN pip3 install tensorflow==2.11.0
RUN pip3 install autopep8
RUN pip3 install seaborn==0.12.2
RUN pip3 install matplotlib==3.6.2
RUN pip3 install numpy==1.23.5
RUN pip3 install pandas==1.5.2
RUN pip3 install scikit-learn==1.2.0
RUN pip3 install py-cpuinfo
#RUN pip3 install sequana --upgrade
RUN pip3 install dcor==0.6

# Install the package.
COPY setup.py .
COPY learners/ learners
RUN pip3 install .
ENV INSIDE_DOCKER_CONTAINER Yes
