FROM python:3.12-slim

RUN groupadd --gid 10001 harness \
    && useradd --uid 10001 --gid harness --create-home --shell /usr/sbin/nologin harness

WORKDIR /app
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /app/corpus /app/.cache /app/traces \
    && chown -R harness:harness /app

COPY --chown=harness:harness pyproject.toml README.md ./
COPY --chown=harness:harness harness ./harness

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

USER harness

EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --start-period=20s --retries=8 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5).read()"

CMD ["harnessd", "--host", "0.0.0.0", "--port", "8080", "--config", "/app/harness.yaml"]
