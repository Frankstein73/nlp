FROM debian:bookworm-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY . ./workshop1
WORKDIR /workshop1
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* && \
    uv sync --no-cache

CMD uv run app.py