FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[api,mqtt,plot]" \
    && python -c "import cv2, onnxruntime, roomsense" \
    && roomsense models download nano --dir /app/models

EXPOSE 8080
ENTRYPOINT ["roomsense"]
CMD ["simulate", "--qrc-mock", "--out", "/app/out/sim"]
