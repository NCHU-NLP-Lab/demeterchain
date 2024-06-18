FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu20.04

RUN apt update && \
    apt install -y \
        wget build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
        libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /temp

# download python
RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar -xvf Python-3.10.12.tgz

# install python
RUN cd Python-3.10.12 && \
    ./configure --enable-optimizations && \
    make && \
    make install

WORKDIR /workspace

RUN rm -r /temp && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip

# install pytorch
RUN pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 && \
    rm -r /root/.cache/pip

# install demeterchain
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y openjdk-11-jdk
RUN pip install demeterchain==1.0.2 faiss-cpu==1.7.4 pyserini==0.22.1 fastapi uvicorn[standard]

# install flash_attention_2
RUN apt install -y git
RUN pip install wheel setuptools 
RUN pip install flash-attn --no-build-isolation

#
WORKDIR /app
COPY data /data
COPY app /app
ENTRYPOINT ["python", "app.py"]