FROM python:3.12.9-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy only the essential files first for Docker layer caching.
COPY pyproject.toml .

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Copy the project into the image
COPY . .

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Expose the port your application uses
EXPOSE 8080

# Declare a build argument; default to "ROOT"
ARG DEPLOYMENT_TARGET=ROOT
ENV DEPLOYMENT_TARGET=${DEPLOYMENT_TARGET}

# Copy an entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Use the entrypoint script to select the API
ENTRYPOINT ["/entrypoint.sh"]
