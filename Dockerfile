# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .streamlit/ ./.streamlit/
COPY streamlit_app.py ./
# Keep the editable source layout: default data paths are relative to this checkout.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/experiments \
    && chown appuser:appuser /app /app/data /app/experiments
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3); urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"
CMD ["python", "scripts/run_dashboard.py", "--ui-host", "0.0.0.0"]
