FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . /app/

# Install the mandala-rl package
RUN pip install --no-cache-dir -e .

# Install workflow requirements
RUN pip install --no-cache-dir -r workflows/requirements.txt

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Flyte will override the entrypoint
CMD ["python"]
