FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python3.8 python3.8-dev python3.8-distutils python3-pip

ENV APP /app
RUN mkdir $APP
WORKDIR $APP

COPY requirements.txt .
RUN python3.8 -m pip install -U pip setuptools wheel
RUN python3.8 -m pip install -r requirements.txt

COPY pretrained pretrained
COPY src src
COPY xmlc xmlc
COPY experiments experiments

RUN dvc init --no-scm

EXPOSE 5000
