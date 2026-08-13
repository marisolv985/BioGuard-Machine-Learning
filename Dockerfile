# Build Stage
FROM python:3.14-slim AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Production Hardened Runtime Stage (Bank DevSecOps Standard)
FROM python:3.14-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TMPDIR=/tmp

WORKDIR /app

# Non-root unprivileged user (UID 10001) for strict CIS Docker Benchmark compliance
RUN groupadd -g 10001 bioguard \
    && useradd -u 10001 -g bioguard -s /sbin/nologin -d /app bioguard \
    && mkdir -p /app/data /tmp \
    && chown -R bioguard:bioguard /app /tmp

COPY --from=builder /install /usr/local
COPY --chown=bioguard:bioguard app ./app

USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
