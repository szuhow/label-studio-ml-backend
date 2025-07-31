# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS python-base
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8 \
    MODEL_PATH=/app/best_attention_resunet_dice_16_50_v1.pth \
    MODEL_TYPE=attention_resunet \
    RESOLUTION=384 \
    THRESHOLD=0.5 \
    HOST=0.0.0.0 \
    DISPLAY=:99 \
    QT_QPA_PLATFORM=offscreen \
    OPENCV_IO_ENABLE_OPENEXR=1

# Update the base OS and install system dependencies for OpenCV
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libfontconfig1 \
        libxcb1 \
        libegl1-mesa \
        libdbus-1-3 \
        libxi6 \
        libxkbcommon-x11-0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libxtst6 \
        libxss1 \
        libasound2 \
        ffmpeg \
        libavcodec-extra; \
    apt-get autoremove -y

# Create requirements files if they don't exist
RUN touch requirements-base.txt requirements-test.txt

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    [ -s requirements-base.txt ] && pip install -r requirements-base.txt || echo "No base requirements"

# install custom requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      [ -s requirements-test.txt ] && pip install -r requirements-test.txt || echo "No test requirements"; \
    fi

# Test OpenCV installation
RUN python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" || \
    (echo "OpenCV installation failed" && exit 1)

# Test torch installation
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')" || \
    (echo "PyTorch installation failed" && exit 1)

COPY best_model.pth .

COPY . .

# Skopiuj model (jeśli istnieje lokalnie)
# COPY best_model.pth ./

EXPOSE 9090

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi_multi:app
