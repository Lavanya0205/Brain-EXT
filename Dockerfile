FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies FIRST
RUN pip install --no-cache-dir -r requirements.txt

# THEN install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy rest of project
COPY . .

# Use dynamic port from Render
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]