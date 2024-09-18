# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Install NLTK and download required data
RUN python -m nltk.downloader punkt wordnet stopwords

# Copy the Django application code
COPY . /app/

# Copy the entry point script
COPY entrypoint.sh /app/

# Make the entry point script executable
RUN chmod +x /app/entrypoint.sh

# Expose port 8000 to the outside world
EXPOSE 8000

# Set the entry point
CMD  ["sh","/app/entrypoint.sh"]
