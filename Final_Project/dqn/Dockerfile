# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./

# Just install what's in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY phase3.py ./

ENTRYPOINT ["python", "/app/phase3.py"]
