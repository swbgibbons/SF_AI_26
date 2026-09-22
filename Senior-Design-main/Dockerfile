#using slim python 3.11
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ---- System deps (curl for uv installer) ----
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash curl libgomp1 libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install uv ----
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:/app/.venv/bin:$PATH"

# ---- Copy dependency metadata ----
COPY pyproject.toml ./

# ---- Install Python deps ----
RUN uv sync --no-dev --no-cache

# ---- App code ----
COPY backend /app/backend
WORKDIR /app/backend

# copy entrypoint script
COPY entry.bash /entry.bash

COPY /backend/exported_models /app/backend/exported_models

RUN chmod +x /entry.bash \
    && sed -i 's/\r$//' /entry.bash
# run the bash
CMD ["/entry.bash"]
