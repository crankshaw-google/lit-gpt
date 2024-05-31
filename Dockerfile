# syntax=docker/dockerfile:experimental

FROM nvcr.io/nvidia/pytorch:24.02-py3

RUN pip install -U torch>=2.3.0 torchvision torchaudio

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
# libavdevice-dev rerquired for latest torchaudio
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
  libsndfile1 sox \
  libfreetype6 \
  swig \
  ffmpeg \
  libavdevice-dev && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/

RUN pip install nvidia-dlprof-pytorch-nvtx nvidia-pyindex nvidia-dlprof

COPY . .

RUN pip install -e '.[all]'

# Check install
RUN python -c "from litgpt.model import GPT, Block" && \
  litgpt pretrain --help && \
  python -c "import lightning as L" && \
  python -c "from lightning.fabric.strategies import FSDPStrategy"


