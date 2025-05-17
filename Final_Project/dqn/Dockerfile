# Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Install both Ray+DQN deps and your own requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ray[default] torch matplotlib \
    && pip install --no-cache-dir -r requirements.txt

# Copy your code in
COPY phase3.py .

# Default entrypoint just runs with your built-in defaults:
CMD ["python", "/app/phase3.py"]
