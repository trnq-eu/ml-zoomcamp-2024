FROM python:3.8.12-slim

# Set the working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ping.py .

# Expose the application port
EXPOSE 9696

# Set the entry point for the container
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "ping:app"]
