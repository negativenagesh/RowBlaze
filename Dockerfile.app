# ---- Builder Stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

# Create a virtual environment
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy and install dependencies into the virtual environment
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt


# ---- Final Stage ----
FROM python:3.11-slim AS final

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH"

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /app/venv /app/venv

# Copy project files
COPY app/ /app/app/
COPY pyproject.toml /app/

# Create directory for file uploads
RUN mkdir -p /app/uploads && chmod 777 /app/uploads

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Add this line before the CMD instruction
ENV PYTHONPATH="/app"

# Command to run the app
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
