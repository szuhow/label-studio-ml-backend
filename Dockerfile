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
    apt-get install --no-install-recommends -y  \
        git \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1 \                    
        libfontconfig1 \
        libxcb1 \
        libegl1 \                  
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

# Test transformers installation
RUN python3 -c "from transformers import __version__; print(f'Transformers version: {__version__}')" || \
    (echo "Transformers installation failed" && exit 1)

# Pre-download Segformer models from Hugging Face during build
# This avoids downloading during container startup and prevents lock conflicts
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python3 -c "\
import os; \
os.makedirs('/root/.cache/huggingface', exist_ok=True); \
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor; \
models_to_download = [ \
    'nvidia/segformer-b0-finetuned-ade-256-256', \
    'nvidia/segformer-b0-finetuned-ade-512-512', \
    'nvidia/segformer-b1-finetuned-ade-512-512', \
    'nvidia/segformer-b2-finetuned-ade-512-512', \
    'nvidia/segformer-b3-finetuned-ade-512-512', \
    'nvidia/segformer-b4-finetuned-ade-512-512', \
    'nvidia/segformer-b5-finetuned-ade-640-640' \
]; \
print('Downloading Segformer models from Hugging Face...'); \
for model_name in models_to_download: \
    try: \
        print(f'Downloading {model_name}...'); \
        SegformerForSemanticSegmentation.from_pretrained(model_name); \
        SegformerImageProcessor.from_pretrained(model_name); \
        print(f'✓ {model_name} downloaded'); \
    except Exception as e: \
        print(f'⚠ Warning: Failed to download {model_name}: {e}'); \
print('Segformer models pre-download complete'); \
" || echo "Warning: Some Segformer models may not have been pre-downloaded"

COPY . .

EXPOSE 9090

CMD gunicorn --preload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 _wsgi_multi_endpoints:app
