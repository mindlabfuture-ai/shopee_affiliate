FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY ugc ./ugc
COPY app ./app
ENV UGC_DATA_DIR=/data UGC_REQUIRE_PASSWORD=1 UGC_SECURE_COOKIE=1 PYTHONUNBUFFERED=1
CMD ["sh", "-c", "python -m ugc serve --host 0.0.0.0 --port ${PORT:-8000}"]
