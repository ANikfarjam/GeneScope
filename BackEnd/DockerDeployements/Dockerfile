FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your backend files
COPY ../Marimo_server/. /app

# Install dependencies
COPY ../Marimo_server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "API:app", "--host", "0.0.0.0", "--port", "8000"]

#CMD ["uvicorn", "Marimo_server.API:app", "--host", "0.0.0.0", "--port", "8000"]
# 
# ===== Stage 1: Builder =====
# FROM python:3.13-slim

# WORKDIR /build

# # # Install system dependencies
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     gcc \
#     libffi-dev \
#     libssl-dev \
#     && rm -rf /var/lib/apt/lists/*
# # Install Python dependencies in isolated stage
# COPY ./Marimo_server/requirements.txt .
# RUN pip install --user --no-cache-dir -r requirements.txt

# # ===== Stage 2: Final Image =====

# # Copy core scripts and configs
# COPY ./Marimo_server/API.py .
# COPY ./Marimo_server/app.py .
# COPY ./Marimo_server/DMM.png .
# COPY ./Marimo_server/EDA.py .
# COPY ./Marimo_server/head1.csv .
# COPY ./Marimo_server/major_findings.py .
# COPY ./Marimo_server/manifest.json .
# COPY ./Marimo_server/MLs.py .
# COPY ./Marimo_server/requirements.txt .
# COPY ./Marimo_server/site.webmanifest .

# # Copy folders
# COPY ./Marimo_server/assets ./assets
# COPY ./Marimo_server/data ./data
# COPY ./Marimo_server/img ./img
# COPY ./Marimo_server/AHPresults ./AHPresults
# COPY ./Marimo_server/DNNResults ./DNNResults
# COPY ./Marimo_server/layouts ./layouts
