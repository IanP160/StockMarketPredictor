FROM python:3.10-bullseye

WORKDIR /app

COPY requirements.txt requirements.txt

# Install all the dependencies in one layer
RUN apt-get update && apt-get install -y build-essential && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Command to run the application
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
