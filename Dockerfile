FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV APP_ROOT /app
ENV WORK_DIR /app/workspace

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update -y && apt-get install -y \
    sudo \
    git \
    wget \
    curl \
    cmake \
    unzip \
    gcc \
    g++ \
    mecab \
    libmecab-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc
RUN git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git /tmp/neologd \
    && /tmp/neologd/bin/install-mecab-ipadic-neologd -n -y \
    && rm -rf /tmp/neologd
RUN ls /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd
RUN pip3 install -U pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY *.py /app/
COPY *.sh /app/

RUN mkdir -p /app/workspace/data
RUN mkdir -p /app/workspace/models

CMD ["bash", "/app/run.sh"]