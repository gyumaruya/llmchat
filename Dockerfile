FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

RUN DEBIAN_FRONTEND="noninteractive" apt-get update \
  && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
  python3-pip python3 python3-dev python3-tk \
  wget vim curl zip unzip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN pip3 install -U --no-cache-dir \
  pip wheel setuptools \
  && pip3 install --no-cache-dir \
  torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 \
  && pip3 install  --no-cache-dir \
  transformers==4.26.1 sentencepiece==0.1.97 huggingface-hub==0.13.2 \
  bitsandbytes==0.37.1 accelerate==0.17.1 \
  pandas numpy scipy matplotlib ipython streamlit
