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

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir "huggingface_hub==0.19.4"
RUN pip install --no-cache-dir "gradio==3.50.2"
RUN pip install --no-cache-dir "tensorflow-cpu==2.15.0"
RUN pip install --no-cache-dir opencv-python-headless dlib scikit-learn numpy Pillow joblib

COPY . .

EXPOSE 7860
CMD ["python", "app.py"]