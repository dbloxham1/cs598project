# Use official Python 3.11 slim
FROM python:3.11-slim

WORKDIR /app

# 1) Pull in any Ubuntu libs TF wheels need (they’re self-contained, but this covers edge cases)
RUN apt-get update && apt-get install -y \
    git curl nano unzip \
    && rm -rf /var/lib/apt/lists/*

# 2) Copy and install everything except TF/tf-keras
COPY requirements.txt .
RUN grep -vE '^(tensorflow|tf-keras)' requirements.txt > base-reqs.txt \
 && pip install --upgrade pip \
 && pip install --no-cache-dir -r base-reqs.txt

# 3) Now explicitly install TensorFlow and the shim in one line
RUN pip install --no-cache-dir tensorflow==2.12.0 tf-keras

# 4) Verify it’s there (this will fail the build early if TF didn’t install)
RUN python -c "import tensorflow as tf; print('TensorFlow OK, version=' + tf.__version__)"

# 5) Copy the rest of your code and dirs
COPY . .

# 6) Create data/scripts dirs if you want
RUN mkdir -p /app/scripts /app/data

CMD ["bash"]