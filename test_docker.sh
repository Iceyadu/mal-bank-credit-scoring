#!/bin/bash
# Quick test script to verify Docker setup

echo "=========================================="
echo "Testing Docker Setup"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    echo "Please install Docker: https://www.docker.com/get-started"
    exit 1
fi

echo "✓ Docker is installed"

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

echo "✓ Docker Compose is available"

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile not found"
    exit 1
fi

echo "✓ Dockerfile exists"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found"
    exit 1
fi

echo "✓ docker-compose.yml exists"

# Check if data directory exists
if [ ! -d "malbank_case_data" ] && [ ! -d "data" ]; then
    echo "⚠ Warning: Data directory not found"
    echo "   Make sure to mount data as volume in docker-compose.yml"
else
    echo "✓ Data directory found"
fi

echo ""
echo "=========================================="
echo "Docker Setup Test Complete"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Build image: docker-compose build"
echo "2. Start container: docker-compose up -d"
echo "3. View logs: docker-compose logs -f"
echo "4. Access Jupyter: http://localhost:8888"
echo ""

