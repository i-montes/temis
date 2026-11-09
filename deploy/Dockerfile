# Temis API: inference only (ONNX Runtime on CPU). The model is mounted at /models, not baked in,
# so a new version is a file copy plus a restart, not a rebuild.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY decisor/ decisor/
RUN useradd --system --uid 10001 temis
USER temis
EXPOSE 8000
# nginx in front sets X-Forwarded-*; only nginx can reach this port.
CMD ["uvicorn", "decisor.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
