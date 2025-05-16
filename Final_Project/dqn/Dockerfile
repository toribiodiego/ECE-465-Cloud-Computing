FROM python:3.10-slim

# 1) set workdir
WORKDIR /app

# 2) copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) copy your code
COPY . .

# 4) entrypoint so args pass through
ENTRYPOINT ["python", "/app/phase3.py"]
CMD []
