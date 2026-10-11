FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /bin/uv /bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY models ./models
ENV PATH="/app/.venv/bin:$PATH"
