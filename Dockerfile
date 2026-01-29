# Build Frontend
FROM node:22-alpine AS builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend ./
RUN npm run build

# Runtime
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y tzdata && rm -rf /var/lib/apt/lists/*

# Set timezone
ENV TZ=Asia/Shanghai
ENV DB_PATH=/app/data/lottery.db

RUN mkdir -p /app/data

# Copy requirements
COPY backend/requirements.txt ./backend/

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend ./backend

# Copy frontend assets from builder
COPY --from=builder /app/frontend/dist ./frontend/dist

RUN find /app -maxdepth 3 -name "*.db" -delete || true

# Expose port
EXPOSE 8888

# Run application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8888"]
