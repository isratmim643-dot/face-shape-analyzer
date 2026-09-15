FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install exact versions to avoid conflicts
RUN pip install --no-cache-dir \
    tensorflow-cpu==2.15.0 \
    "gradio==3.50.2" \
    "huggingface_hub==0.19.4" \
    opencv-python-headless \
    dlib \
    scikit-learn \
    numpy \
    Pillow \
    joblib

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]