# The agent harness. Deliberately thin: it needs Python, the OpenAI SDK, and
# access to the project it is working on. Everything else is a shell command
# the agent issues.
#
# Build:   docker build -t agent-harness .
# Run:     docker compose run --rm agent "your task"
#
# The project is mounted READ-ONLY at /project by default. Writable paths are
# mounted separately so the agent cannot silently rewrite the project:
#   /workspace  - staging area for scratch work
#   /out        - transcripts
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ripgrep curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /harness
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy every module rather than naming them. An explicit list is a trap of
# exactly the kind that has bitten this project before (the engine image
# crash-looped because a new module was missing from its COPY list): a new
# .py file works locally and the container dies on import.
COPY *.py ./

# Non-root. The container is a worker, not an owner of the host.
RUN useradd -m agent && chown -R agent:agent /harness
USER agent

# /project is the code the agent works on. Mounted read-only in compose.
WORKDIR /workspace
ENTRYPOINT ["python", "/harness/run.py"]
