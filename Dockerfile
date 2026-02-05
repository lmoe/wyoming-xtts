# CUDA variant: cu126 (Pascal-Ada) or cu128 (Volta-Blackwell)
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-devel-ubuntu22.04
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

FROM ${CUDA_IMAGE} AS builder

ARG TORCH_INDEX_URL
ARG TORCH_CUDA_ARCH_LIST
ARG VERSION=0.0.0.dev0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${VERSION}
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml /tmp/
COPY wyoming_xtts /tmp/wyoming_xtts/

ENV DS_BUILD_TRANSFORMER_INFERENCE=1
ENV DS_BUILD_STOCHASTIC_TRANSFORMER=1
RUN pip install --no-cache-dir --extra-index-url ${TORCH_INDEX_URL} /tmp/ && \
    pip install --no-cache-dir 'transformers>=4.39,<4.40'

FROM ${CUDA_IMAGE}

ARG TORCH_CUDA_ARCH_LIST
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    libgomp1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY wyoming_xtts /app/wyoming_xtts

VOLUME /data

ENV XTTS_ASSETS=/data

EXPOSE 10200

ENTRYPOINT ["python", "-m", "wyoming_xtts"]
CMD ["--uri", "tcp://0.0.0.0:10200"]
