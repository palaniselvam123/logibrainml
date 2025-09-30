FROM python:3.10-slim

# minimal apt deps for many Python wheels
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Ensure pip latest and install pinned deps (numpy pinned <1.25 to match sklearn 1.1.3)
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# add unprivileged user and switch
RUN useradd -m -u 1000 appuser || true
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
