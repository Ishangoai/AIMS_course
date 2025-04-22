# specify the base image for the build.
# the -slim tag means it's a smaller variant of the official Python image
FROM python:3.12.9-slim

# temporarily use another image hosted on GitHub Container Registry (ghcr.io)
# copy the uv and uvx executables from the uv image into the /bin/ directory of the current image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# sets the working directory inside the container to /app
# any subsequent RUN, CMD, ENTRYPOINT, COPY, ADD instructions will be executed relative to this directory.
WORKDIR /app

# Copy only the essential files first for Docker layer caching.
COPY pyproject.toml .

# mount a persistent cache directory specific to uv. 
# this allows downloaded packages to be reused across builds,
# significantly speeding up dependency installation if they haven't changed.
RUN --mount=type=cache,target=/root/.cache/uv \

# temporarily mounts the uv.lock file into the container at /app/uv.lock 
# for this command without adding it permanently to this layer.
# uv.lock contains the exact pinned versions of all dependencies.
    --mount=type=bind,source=uv.lock,target=uv.lock \

# similar to the lock file, mounts the pyproject.toml file from the build context.
# while it was already copied, using a bind mount ensures this command uses the 
# absolute latest version from the context for dependency resolution.
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \

# syncing is the process of installing a subset of 
# packages from the lockfile into the project environment.
    uv sync --frozen --no-install-project

# copies all files and directories from the build context (your local project directory, the first .) 
# into the current working directory (/app, the second .)
COPY . .

# runs uv sync --frozen again, but without --no-install-project.
# now that the project code (.) has been copied into /app, 
# this command will install the project itself (e.g., in editable mode if configured that way in pyproject.toml), 
# making it importable or executable within the container environment.
# It still uses the build cache (--mount=type=cache) for efficiency.
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
