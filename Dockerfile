# specify the base image for the build.
# the -slim tag means it's a smaller variant of the official Python image
FROM python:3.12.9-slim

# temporarily use another image hosted on GitHub Container Registry (ghcr.io)
# copy the uv and uvx executables from the uv image into the /bin/ directory of the current image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# sets the working directory inside the container to /app
# any subsequent RUN, CMD, ENTRYPOINT, COPY, ADD instructions will be executed relative to this directory.
WORKDIR /app

# copies all files and directories from the build context (your local project directory, the first .) 
COPY . .

# this command will install the project itself (e.g., in editable mode if configured that way in pyproject.toml), 
# making it importable or executable within the container environment.
RUN uv sync --frozen

# Expose the port your application uses
EXPOSE 8080

# Use the entrypoint script to select the API
ENTRYPOINT ["/app/entrypoint.sh"]
