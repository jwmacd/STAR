#For Unraid, add --gpus all --shm-size=1g in Extra Parameters when creating the container.
ARG UBUNTVERSION=22.04
ARG CUDA_VERSION=11.8.0

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTVERSION}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    libsm6 \
    libxext6 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

# Explicitly install everything in the runtime stage.
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

ENV PYTHONPATH=/app:$PYTHONPATH
#ENV XFORMERS_FORCE_DISABLE_TRITON="1"
ENV ATTENTION=flash
ENV FORCE_CUDA="1"

WORKDIR /app/video_super_resolution

ENTRYPOINT ["bash", "scripts/inference_sr.sh"]