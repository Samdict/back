# FROM python:3.11-slim

# WORKDIR /app

# # Install system dependencies including build tools
# RUN apt-get update && apt-get install -y \
#     libsndfile1 \
#     ffmpeg \
#     gcc \
#     g++ \
#     make \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements and install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Remove build tools to keep image small
# RUN apt-get remove -y gcc g++ make && apt-get autoremove -y

# # Copy application code
# COPY app/ ./app/
# COPY main.py .

# # Create uploads directory
# RUN mkdir -p uploads

# # Expose port
# EXPOSE 8000

# # Run the application
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY main.py .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]