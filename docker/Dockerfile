FROM python:3.11-slim

LABEL maintainer="ParkVision ANPR"
LABEL description="Indian Smart Parking & Number Plate Recognition System"

# Install system dependencies for OpenCV + EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create data directories
RUN mkdir -p data/output_images data/output_videos data/crops data/sample_plates

# Expose FastAPI and Streamlit ports
EXPOSE 8000 8501

# Default: start FastAPI backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
