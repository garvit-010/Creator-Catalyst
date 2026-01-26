# Use python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# ADDED: ffmpeg (Critical for video processing)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose port 5000
EXPOSE 5000

# Run Streamlit on port 5000
CMD ["streamlit", "run", "app/app.py", "--server.port=5000", "--server.address=0.0.0.0"]