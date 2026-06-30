# SAR Detection

## Docker Setup

This project uses Docker for a consistent development environment with Python, MongoDB, and all required dependencies.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Building and Running

#### Build the Docker image

```bash
docker compose build
```

#### Start the services (MongoDB + uv container)

```bash
docker compose up -d
```

This will start:
- **MongoDB** service on port 27017
- **uv** container with Python environment

#### Enter the container shell

```bash
docker compose exec uv bash
```

#### Run Python scripts

From inside the container:

```bash
uv run <script_name>.py
```

Or directly from your host:

```bash
docker compose exec uv uv run <script_name>.py
```

#### Stop services

```bash
docker compose down
```

#### View running containers

```bash
docker compose ps
```

#### Run as root

To run a one-off root shell in the `uv` service:

```bash
docker compose exec --user root uv bash
```

To run a command as root directly:

```bash
docker compose exec --user root uv uv run <script_name>.py
```

Use this only when you need root privileges for installation or debugging. For normal development, keep the container running as `appuser`.

### Environment

- FiftyOne is configured to connect to MongoDB automatically via `FIFTYONE_DATABASE_URI=mongodb://mongo:27017`
- All workspace files are mounted into `/workspace` inside the container
- Changes made locally are reflected immediately in the container

### Container user and permissions

This setup creates a local container user based on your host UID and GID so files created inside the container remain writable by your host user.

- `Dockerfile` uses build args `UID` and `GID` to create the `appuser` account
- `docker-compose.yaml` passes `UID` and `GID` from `.env`
- The container runs as `appuser`, not root

#### Setup for a new developer or machine

1. Copy the template:

```bash
cp .env.example .env
```

2. Set your UID/GID values:

```bash
echo "UID=$(id -u)" > .env
echo "GID=$(id -g)" >> .env
```

3. Build and start as normal:

```bash
docker compose build
docker compose up -d
```

#### Notes

- `.env` is ignored by Git, so each developer can use their own values safely
- If another user on another machine has a different UID/GID, they should update their local `.env`
- If you see permission errors, rebuild the container after updating `.env`
