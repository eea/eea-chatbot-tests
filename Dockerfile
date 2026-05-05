# Use the official Playwright Python image as base
# This image comes with Python and all necessary system dependencies for Playwright
FROM mcr.microsoft.com/playwright/python:v1.49.1-jammy

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure Playwright browsers are installed
# The base image already has them, but we make sure chromium is present
RUN playwright install chromium

# Copy the rest of the application code
COPY . .

# Install the project in editable mode so 'chatbot_tests' command is available
RUN pip install -e .

# Create reports directory and ensure it's writable
RUN mkdir -p reports && chmod 777 reports

# Set default environment variables
ENV CHATBOT_BASE_URL=https://www.eea.europa.eu
ENV CHATBOT_PATH=/en/chatbot
ENV HEADLESS=true
ENV REPORTS_DIR=/app/reports

# Default command to run basic and question tests with color output
CMD ["chatbot_tests", "run", "-m", "basic,question", "--color"]
