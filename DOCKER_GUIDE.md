# Docker Guide for Credit Risk Modeling Project

## Quick Start

### Prerequisites
- Docker installed ([Get Docker](https://www.docker.com/get-started))
- Docker Compose (usually included with Docker Desktop)

### Option 1: Using Docker Compose (Recommended)

**Start the container:**
```bash
docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f
```

**Access Jupyter Lab:**
- Open browser: http://localhost:8888
- Look for the token in the logs (or use `docker-compose logs`)

**Stop the container:**
```bash
docker-compose down
```

### Option 2: Using Docker Directly

**Build the image:**
```bash
docker build -t credit-risk-modeling .
```

**Run the container:**
```bash
docker run -d \
  --name mal-bank-credit-risk \
  -p 8888:8888 \
  -v $(pwd)/malbank_case_data:/app/data:ro \
  -v $(pwd)/notebooks:/app/notebooks \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/plots:/app/plots \
  credit-risk-modeling
```

**Get Jupyter token:**
```bash
docker logs mal-bank-credit-risk | grep token
```

**Stop and remove:**
```bash
docker stop mal-bank-credit-risk
docker rm mal-bank-credit-risk
```

## Data Setup

### Option A: Use Existing Data Directory
The docker-compose.yml mounts `./malbank_case_data` to `/app/data` in the container.

**In the notebook, use:**
```python
data_dir = "data"  # Points to /app/data in container
```

### Option B: Copy Data into Container
If you want data inside the image:

1. Copy data to `data/` directory:
```bash
cp -r malbank_case_data/* data/
```

2. Update Dockerfile to copy data:
```dockerfile
COPY data/ /app/data/
```

3. Rebuild image

## Running the Notebook

1. **Start container** (see above)

2. **Access Jupyter Lab:**
   - Open http://localhost:8888
   - Navigate to `credit_risk_model.ipynb`

3. **Update data path in notebook:**
   ```python
   data_dir = "data"  # Container path
   ```

4. **Run all cells** or run sequentially

## Volumes Explained

The docker-compose.yml mounts several volumes:

- **`./malbank_case_data:/app/data:ro`** - Data files (read-only)
- **`./notebooks:/app/notebooks`** - Notebooks (editable)
- **`./src:/app/src`** - Source code (editable)
- **`./models:/app/models`** - Saved models (persistent)
- **`./plots:/app/plots`** - Generated plots (persistent)
- **`./docs:/app/docs`** - Documentation (editable)

**Benefits:**
- Changes to notebooks/code persist on host
- Models and plots saved to host
- Data remains on host (not copied)

## Development Workflow

### Making Code Changes

1. Edit files on host (notebooks, src/)
2. Changes are immediately available in container (via volumes)
3. Restart kernel in Jupyter if needed

### Installing New Packages

**Option 1: Update requirements.txt and rebuild**
```bash
# Edit requirements.txt
docker-compose build
docker-compose up -d
```

**Option 2: Install in running container**
```bash
docker-compose exec credit-risk-modeling pip install package-name
```

**Option 3: Interactive shell**
```bash
docker-compose exec credit-risk-modeling bash
pip install package-name
```

## Troubleshooting

### Port Already in Use
```bash
# Change port in docker-compose.yml
ports:
  - "8889:8888"  # Use 8889 instead
```

### Permission Issues
```bash
# Fix permissions
sudo chown -R $USER:$USER models plots notebooks
```

### Container Won't Start
```bash
# Check logs
docker-compose logs

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up
```

### Jupyter Token Not Found
```bash
# Get token from logs
docker-compose logs | grep -i token

# Or access without token (set password)
docker-compose exec credit-risk-modeling jupyter lab password
```

### Out of Memory
```bash
# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory
# Set to at least 4GB
```

## Production Deployment

### Build for Production
```dockerfile
# Use multi-stage build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "-m", "notebooks.credit_risk_model"]
```

### Run as Service
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  credit-risk-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models
    command: uvicorn api:app --host 0.0.0.0 --port 8000
```

## Alternative: Jupyter Notebook (Not Lab)

To use classic Jupyter Notebook instead:

**Update Dockerfile:**
```dockerfile
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app/notebooks"]
```

## Environment Variables

Set custom environment variables:

```yaml
# docker-compose.yml
environment:
  - PYTHONPATH=/app/src
  - JUPYTER_ENABLE_LAB=yes
  - DATA_DIR=/app/data
  - MODEL_DIR=/app/models
```

## Clean Up

**Remove containers and volumes:**
```bash
docker-compose down -v
```

**Remove images:**
```bash
docker rmi credit-risk-modeling
```

**Full cleanup:**
```bash
docker system prune -a
```

## Tips

1. **Use .dockerignore** to exclude unnecessary files
2. **Layer caching**: Copy requirements.txt first for faster rebuilds
3. **Volume mounts**: Keep data on host, not in image
4. **Multi-stage builds**: For smaller production images
5. **Health checks**: Add to docker-compose.yml for production

## Example: Running Tests in Container

```bash
# Run setup test
docker-compose exec credit-risk-modeling python test_setup.py

# Run specific Python script
docker-compose exec credit-risk-modeling python -m src.data_loading
```

