# FROM ghcr.io/astral-sh/uv:debian
FROM ghcr.io/astral-sh/uv:0.11.15-debian-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && useradd -m -u ${UID} -g ${GID} appuser

RUN mkdir -p /workspace/.venv && chown -R ${UID}:${GID} /workspace/.venv

USER appuser

# Install python in the virtual environment based on the .python-version file
RUN uv python install