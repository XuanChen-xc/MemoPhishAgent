FROM mcr.microsoft.com/playwright/python:v1.44.0

# Set working directory
WORKDIR /work

# Copy all files from current directory to container
COPY . /work

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install browser drivers (Chromium, Firefox, WebKit)
RUN playwright install