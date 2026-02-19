FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

WORKDIR /app/src

EXPOSE 7860

CMD ["sh", "-c", "uvicorn run:app --host 0.0.0.0 --port ${PORT:-7860}"]
