FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY benchmarks/ benchmarks/

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Default command runs the distributed benchmark simulation
# Ideally, this would run the "real" cluster benchmark if we had one implemented.
# For now, we package the simulation and the needle benchmark.
CMD ["python", "benchmarks/needle_benchmark.py"]
