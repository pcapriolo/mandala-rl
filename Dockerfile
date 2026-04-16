FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD gunicorn serve:app --bind 0.0.0.0:${PORT:-8080} --timeout 30 --workers 1
