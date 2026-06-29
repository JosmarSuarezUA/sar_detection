FROM ghcr.io/astral-sh/uv:debian

RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*

ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} appuser && useradd -m -u ${UID} -g ${GID} appuser
