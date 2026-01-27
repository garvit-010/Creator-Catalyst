# 1. Base Image
FROM python:3.9-slim

# 2. Setup System Dependencies (ffmpeg is crucial here)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Work Directory
WORKDIR /app

# 4. Copy Requirements & Install (Cache Layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. COPY THE APP CODE (This was the critical fix in the other repo)
COPY . .

# 6. Expose Port
EXPOSE 5000

# 7. Run Command (Keep Streamlit!)
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]