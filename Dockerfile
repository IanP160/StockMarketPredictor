# Start with the base Python image
FROM python:3.10-slim

# Set up environment
WORKDIR /app

# Copy requirements file before installing dependencies
COPY requirements.txt /app/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose the port your app runs on
EXPOSE 10000

# Command to run your app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app"]
