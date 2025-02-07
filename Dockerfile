ARG UBUNTVERSION=22.04
ARG CUDA_VERSION=12.5.1

# Build stage - for compiling if needed
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTVERSION} as builder

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
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

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r cogvideox-based/sat/requirements.txt

# Replace transformer.py in SAT package
RUN cp cogvideox-based/transformer.py /usr/local/lib/python3.10/dist-packages/sat/model/transformer.py

# Runtime stage - minimal CUDA environment
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTVERSION}

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    libsm6 \
    libxext6 \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /app /app

# Ensure Python path includes copied dependencies
ENV PYTHONPATH=/usr/local/lib/python3.10/dist-packages:/app

WORKDIR /app/video_super_resolution
ENTRYPOINT ["bash", "scripts/inference_sr.sh"]