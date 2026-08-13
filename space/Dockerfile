FROM python:3.11-slim

# opencv-python-headless still needs libgl/libglib present at import time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user

WORKDIR /app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# 0.0.0.0 so the platform can reach it; --no-browser because there is no display.
CMD ["python", "app.py", "--weights", "model.pt", \
     "--host", "0.0.0.0", "--port", "7860", "--no-browser", "--threads", "2"]
