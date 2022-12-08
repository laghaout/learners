# Inherit from the Python 3.9 image.
FROM python:3.9-slim

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
RUN pip3 install tensorflow==2.10.0
RUN pip3 install autopep8
RUN pip3 install seaborn
RUN pip3 install matplotlib==3.5.3
RUN pip3 install numpy==1.23.4
RUN pip3 install pandas==1.5.1
RUN pip3 install scikit-learn==1.1.2
RUN pip3 install py-cpuinfo
#RUN pip3 install sequana --upgrade
RUN pip3 install dcor==0.5.7

# Install the package.
COPY setup.py .
COPY learners/ learners
RUN pip3 install .
ENV INSIDE_DOCKER_CONTAINER Yes
