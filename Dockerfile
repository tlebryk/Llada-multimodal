# syntax=docker/dockerfile:1.4
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Configure the Python directory so it is consistent
# And only use the managed Python version
ENV UV_PYTHON_INSTALL_DIR=/python UV_PYTHON_PREFERENCE=only-managed

# Install Python before the project for caching
# RUN uv python install 3.12
# COPY fetch_data.sh /app/
# RUN bash fetch_data.sh

COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --no-dev


COPY . /app
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
CMD ["uv", "run", "llada_train.py"]


