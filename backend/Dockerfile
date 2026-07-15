# Use a lightweight Python 3.10 image
FROM python:3.10-slim

# Install system dependencies (needed for OpenCV and image processing)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Start the server (using 0.0.0.0 is mandatory inside Docker)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]