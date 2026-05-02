FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY train_ckd.py .
COPY train_diabetes.py .
COPY datasets/ ./datasets/
COPY templates/ ./templates/

RUN mkdir -p models

RUN python train_ckd.py
RUN python train_diabetes.py

EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

CMD ["python", "app.py"]
