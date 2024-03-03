# --- Model cache stage --- #
FROM ubuntu:22.04 AS model

# Install git lfs support
RUN apt-get update && apt-get install -y git-lfs && rm -rf /var/lib/apt/lists/*
RUN git lfs install
# Download the LLava model
RUN git clone https://huggingface.co/panoyo9829/llava-1.6-gptq-4bit /tmp/model && rm -rf /tmp/model/.git
# Download CLIP image encoder, but ignore LFS file tf_model.h5
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/openai/clip-vit-large-patch14-336 /tmp/mm_vision_tower && \
    git -C /tmp/mm_vision_tower lfs fetch --exclude="tf_model.h5" && \
    git -C /tmp/mm_vision_tower lfs checkout && \
    rm -rf /tmp/mm_vision_tower/.git



# --- Main stage --- #
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Copy the downloaded models to the working directory
COPY --from=model /tmp/model /app/models/llava
COPY --from=model /tmp/mm_vision_tower /app/models/mm_vision_tower

# Patch LLava's config.json to use the downloaded mm_vision_tower
RUN sed -i 's|openai/clip-vit-large-patch14-336|/app/models/mm_vision_tower|' /app/models/llava/config.json

# Set common environment variables
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    && apt-get install -y clang-15 \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install poetry

# Set the working directory
WORKDIR /app

# Copy pyproject.toml and install dependencies
COPY poetry.lock pyproject.toml .
RUN --mount=type=cache,target=/tmp/poetry_cache poetry install -v --without dev --no-root

# Copy handler.py to the working directory
COPY handler.py .

# Link clang-15 to clang
RUN ln -s /usr/bin/clang-15 /usr/bin/clang

ENTRYPOINT ["poetry", "run", "python", "handler.py"]
