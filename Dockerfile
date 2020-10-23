FROM python:3.6

ARG USER_ID
ARG USER_NAME

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    mesa-utils \
    binutils kmod \
  && rm -rf /var/lib/apt/lists/*# Env vars for the nvidia-container-runtime.

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

ADD NVIDIA-DRIVER.run .
RUN chmod +x NVIDIA-DRIVER.run && ./NVIDIA-DRIVER.run -a --ui=none --no-kernel-module && rm ./NVIDIA-DRIVER.run

RUN pip install jupyter notebook

RUN useradd --uid $USER_ID -m $USER_NAME
USER audran
WORKDIR /work

ADD requirements.txt .
RUN pip install --user -r requirements.txt


ENTRYPOINT jupyter notebook --NotebookApp.ip=0.0.0.0
