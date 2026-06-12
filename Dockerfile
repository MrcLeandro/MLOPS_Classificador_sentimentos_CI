FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements-mlserver.txt .
RUN pip install --no-cache-dir -r requirements-mlserver.txt
COPY . .
EXPOSE 8080 8081 8082
HEALTHCHECK --interval=10s --timeout=5s --retries=5 CMD curl -f http://localhost:8080/health || exit 1
CMD ["mlserver", "start", "--model-settings-path", "/app/mlserver/model-settings.json"]
