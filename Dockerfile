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
RUN pip3 install tensorflow
RUN pip3 install autopep8
RUN pip3 install seaborn
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install py-cpuinfo
#RUN pip3 install sequana --upgrade
RUN pip3 install dcor

# Install the package.
COPY setup.py .
COPY learners/ learners
RUN pip3 install .
