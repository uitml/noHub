FROM bitnami/pytorch:1.12.1-debian-11-r5


USER root

# Install required packages
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

# Install faiss with conda
RUN conda install -y -c pytorch faiss-gpu

WORKDIR /storage
CMD [ "/bin/bash" ]
