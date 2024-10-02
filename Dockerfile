# Start with the base Python image
FROM python:3.10-slim

# Set up environment
WORKDIR /app

# Copy requirements file before installing dependencies
COPY requirements.txt /app/requirements.txt

# Install virtualenv and create a virtual environment
RUN pip install --no-cache-dir virtualenv && virtualenv venv

# Activate virtual environment and install dependencies
RUN ./venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Command to run your app
CMD ["./venv/bin/gunicorn", "-b", "0.0.0.0:8000", "app:app"]
