# Multi-Service Container for AI-Ready Data Dispute System
FROM python:3.12-slim

# Install system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source trees
COPY backend/ ./backend/
COPY webinar/ ./webinar/
COPY tests/ ./tests/
COPY "Food images/" ./"Food images"/

# Set environment
ENV PYTHONPATH=/app/backend:/app

EXPOSE 8000 8501

# Default startup: launches Streamlit interactive workbench
CMD ["streamlit", "run", "webinar/webinar_demo_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
