FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libgl1 \
    libegl1 \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
    > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /workspace

COPY requirements.txt .
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

ENV MUJOCO_GL=egl
