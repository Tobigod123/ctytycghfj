FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY app.py .
COPY train_ckd.py .
COPY train_diabetes.py .
COPY datasets/ ./datasets/
COPY templates/ ./templates/
COPY static/ ./static/

# Create models directory
RUN mkdir -p models

# Train both models at build time
RUN python train_ckd.py
RUN python train_diabetes.py

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run Flask app
CMD ["python", "app.py"]
