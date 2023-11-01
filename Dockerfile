FROM continuumio/miniconda3 AS nmr-respredict-ms

ENV PYTHON_VERSION=3.10
ENV RDKIT_VERSION=2023.09.1

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    apt-get update -y && \
    apt-get install -y curl && \
    conda update -n base -c defaults conda

RUN conda install -c conda-forge python>=PYTHON_VERSION
# RUN conda install -c conda-forge rdkit>=RDKIT_VERSION
RUN python3 -m pip install -U pip

RUN pip3 install rdkit

RUN conda install scikit-learn
RUN conda install numba
RUN conda install numpy
RUN conda install tqdm
RUN conda install networkx
RUN conda install click
RUN conda install pandas
RUN conda install pyarrow
RUN conda install pytorch

COPY . /nmr-respredict
WORKDIR /nmr-respredict