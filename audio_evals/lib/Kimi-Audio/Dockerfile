FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /app

COPY ./requirements.txt /app/
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    curl \
    sox \
    openssh-server \
    ffmpeg \
    libgl1-mesa-glx \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 安装 pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.10 get-pip.py \
    && rm get-pip.py
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation

# alias python3 as python
RUN ln -s /usr/bin/python3 /usr/bin/python

CMD ["/bin/bash"]
