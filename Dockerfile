#  Base image
FROM python:3.10-slim

#  Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Default command (test pipeline)
CMD ["python", "src/test.py"]
