FROM continuumio/miniconda3

ENV PYTHON_VERSION=3.7
ENV RDKIT_VERSION=2020.09.1.0

RUN conda install -c conda-forge python=$PYTHON_VERSION
RUN conda install -c conda-forge rdkit=RDKIT_VERSION
RUN conda install -c conda-forge pytorch=1.9.1
RUN conda install -c conda-forge scikit-learn=0.24.2
RUN conda install -c conda-forge numba=0.53.1
RUN conda install -c conda-forge numpy=1.21.1
RUN conda install -c conda-forge tqdm=4.62.2
RUN conda install -c conda-forge networkx=2.6.3
RUN conda install -c conda-forge click=8.0.1
RUN conda install -c conda-forge pandas=1.3.3
RUN conda install -c conda-forge pyarrow=3.0.0

COPY . /lw-reg
WORKDIR /lw-reg

RUN pip install .
