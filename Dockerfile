FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/invincible-jha/agent-marketplace"
LABEL org.opencontainers.image.description="agent-marketplace: Agent capability registry and discovery"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.vendor="AumOS"

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir ".[server]" 2>/dev/null || pip install --no-cache-dir . && rm -rf /root/.cache

RUN useradd -m -s /bin/bash aumos
USER aumos

ENV PYTHONUNBUFFERED=1

EXPOSE 8085

CMD ["python", "-m", "agent_marketplace.server.app", "--host", "0.0.0.0", "--port", "8085"]
