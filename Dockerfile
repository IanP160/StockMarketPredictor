FROM python:3.10-bullseye

# Set up environment
WORKDIR /app

# Install virtualenv
RUN pip install --no-cache-dir virtualenv

# Create a virtual environment
RUN virtualenv venv

# Activate virtual environment and install dependencies
RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Activate the virtual environment when running commands
CMD ["/app/venv/bin/gunicorn", "-b", "0.0.0.0:8000", "app:app"]
