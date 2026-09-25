# Multi-stage build for the Fine-Tuning Platform control plane (API + dashboard).
#
# This image runs the *platform*, not GPU training. The default (sim) trainer
# needs no GPU. To run real LoRA training you'd build with the [ml] extra and run
# on a GPU node / separate worker image — see deploy/README.md.

# ---- builder: resolve deps into a venv with uv ----------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# Copy only what's needed to resolve + install first, for layer caching.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# ---- runtime: slim image with just the venv + app -------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    FTP_DATA_DIR=/data \
    PYTHONUNBUFFERED=1

# Non-root user for the container.
RUN useradd -m -u 10001 ftp && mkdir -p /data && chown ftp:ftp /data

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY frontend ./frontend
COPY pyproject.toml README.md ./

USER ftp
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "ftp.app:app", "--host", "0.0.0.0", "--port", "8000"]
