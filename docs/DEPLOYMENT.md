# RAG+ Deployment Guide

Complete guide for deploying RAG+ in production environments.

## Table of Contents
- [Production Considerations](#production-considerations)
- [Deployment Options](#deployment-options)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Scaling](#scaling)
- [Security](#security)

---

## Production Considerations

### Performance Requirements

Before deploying, determine your requirements:

| Metric | Consideration |
|--------|---------------|
| **Latency** | Target response time (e.g., < 2 seconds) |
| **Throughput** | Requests per second |
| **Corpus Size** | Number of knowledge items |
| **Concurrency** | Simultaneous users |
| **Availability** | Uptime requirements (e.g., 99.9%) |

### Cost Estimation

RAG+ costs depend on:
- **LLM API calls**: Per token pricing
- **Embedding API calls**: Per token pricing
- **Infrastructure**: Compute, storage, bandwidth
- **Monitoring**: Logging and analytics

**Example Monthly Cost Estimate:**
```
Corpus building (one-time):
- 1000 knowledge items × 1 application each
- ~500 tokens per generation
- Cost: ~$5-10 (one-time)

Query processing (recurring):
- 10,000 queries/month
- ~1000 tokens per query (retrieval + generation)
- Cost: ~$100-200/month
```

---

## Deployment Options

### Option 1: Simple API Server

Deploy as a REST API using Flask/FastAPI.

#### Create API Server

```python
# app.py
from flask import Flask, request, jsonify
from rag_plus import RAGPlus, OpenAILLM, OpenAIEmbeddingModel
import os

app = Flask(__name__)

# Initialize RAG+ (load corpus at startup)
llm = OpenAILLM(model="gpt-3.5-turbo")
embedding_model = OpenAIEmbeddingModel(model="text-embedding-3-small")
rag_plus = RAGPlus(llm, embedding_model, top_k=3)

# Load pre-built corpus
rag_plus.load_corpus("knowledge.json", "applications.json")

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('query')
    task_type = data.get('task_type', 'general')

    if not question:
        return jsonify({'error': 'No query provided'}), 400

    try:
        answer = rag_plus.generate(question, task_type=task_type)
        return jsonify({
            'query': question,
            'answer': answer,
            'task_type': task_type
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### Run Server

```bash
# Development
python app.py

# Production with Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### Test API

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the derivative of x^3?", "task_type": "math"}'
```

### Option 2: Docker Deployment

Containerize for consistent deployment.

#### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn flask

# Copy application code
COPY rag_plus.py .
COPY app.py .
COPY knowledge.json .
COPY applications.json .

# Set environment variables
ENV OPENAI_API_KEY=""
ENV PORT=5000

# Expose port
EXPOSE 5000

# Run with Gunicorn
CMD gunicorn -w 4 -b 0.0.0.0:$PORT app:app
```

#### Build and Run

```bash
# Build image
docker build -t rag-plus:latest .

# Run container
docker run -d \
  -p 5000:5000 \
  -e OPENAI_API_KEY="your-key" \
  --name rag-plus-api \
  rag-plus:latest

# Check logs
docker logs rag-plus-api
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  rag-plus-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - PORT=5000
    volumes:
      - ./knowledge.json:/app/knowledge.json
      - ./applications.json:/app/applications.json
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Run with Docker Compose
docker-compose up -d

# Scale to multiple instances
docker-compose up -d --scale rag-plus-api=3
```

### Option 3: Cloud Deployment

#### AWS Elastic Beanstalk

```bash
# Install EB CLI
pip install awsebcli

# Initialize
eb init -p python-3.9 rag-plus-api

# Create environment
eb create rag-plus-prod

# Deploy
eb deploy

# Set environment variables
eb setenv OPENAI_API_KEY=your-key
```

#### Google Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/rag-plus

# Deploy
gcloud run deploy rag-plus \
  --image gcr.io/PROJECT_ID/rag-plus \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=your-key \
  --allow-unauthenticated
```

#### Azure Container Instances

```bash
# Create container
az container create \
  --resource-group rag-plus-rg \
  --name rag-plus-api \
  --image rag-plus:latest \
  --dns-name-label rag-plus \
  --ports 5000 \
  --environment-variables OPENAI_API_KEY=your-key
```

### Option 4: Kubernetes Deployment

#### Deployment Configuration

```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-plus-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-plus
  template:
    metadata:
      labels:
        app: rag-plus
    spec:
      containers:
      - name: rag-plus
        image: rag-plus:latest
        ports:
        - containerPort: 5000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-plus-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-plus-service
spec:
  selector:
    app: rag-plus
  ports:
  - protocol: TCP
    port: 80
    targetPort: 5000
  type: LoadBalancer
```

#### Deploy to Kubernetes

```bash
# Create secret for API key
kubectl create secret generic rag-plus-secrets \
  --from-literal=openai-api-key=your-key

# Apply deployment
kubectl apply -f k8s-deployment.yml

# Check status
kubectl get pods
kubectl get services

# Scale deployment
kubectl scale deployment rag-plus-deployment --replicas=5
```

---

## Configuration

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="your-openai-key"

# Optional
export RAG_PLUS_TOP_K=3                    # Number of retrieved pairs
export RAG_PLUS_MODEL="gpt-3.5-turbo"      # LLM model
export RAG_PLUS_EMBEDDING_MODEL="text-embedding-3-small"
export RAG_PLUS_MAX_TOKENS=2048            # Max generation tokens
export RAG_PLUS_TEMPERATURE=0.0            # Generation temperature
```

### Configuration File

```python
# config.py
import os

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    # Model Configuration
    LLM_MODEL = os.getenv('RAG_PLUS_MODEL', 'gpt-3.5-turbo')
    EMBEDDING_MODEL = os.getenv('RAG_PLUS_EMBEDDING_MODEL', 'text-embedding-3-small')

    # Retrieval Configuration
    TOP_K = int(os.getenv('RAG_PLUS_TOP_K', '3'))

    # Generation Configuration
    MAX_TOKENS = int(os.getenv('RAG_PLUS_MAX_TOKENS', '2048'))
    TEMPERATURE = float(os.getenv('RAG_PLUS_TEMPERATURE', '0.0'))

    # Corpus Paths
    KNOWLEDGE_PATH = os.getenv('KNOWLEDGE_PATH', 'knowledge.json')
    APPLICATIONS_PATH = os.getenv('APPLICATIONS_PATH', 'applications.json')

    # Server Configuration
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', '5000'))
    WORKERS = int(os.getenv('WORKERS', '4'))

# Usage in app
from config import Config

llm = OpenAILLM(
    api_key=Config.OPENAI_API_KEY,
    model=Config.LLM_MODEL
)
```

---

## Monitoring

### Logging

```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_plus.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Log in application
@app.route('/query', methods=['POST'])
def query():
    logger.info(f"Received query: {request.json.get('query')}")
    try:
        answer = rag_plus.generate(query, task_type=task_type)
        logger.info(f"Generated answer: {len(answer)} characters")
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
```

### Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# Define metrics
query_counter = Counter('rag_plus_queries_total', 'Total queries processed')
query_duration = Histogram('rag_plus_query_duration_seconds', 'Query processing time')
error_counter = Counter('rag_plus_errors_total', 'Total errors')

@app.route('/query', methods=['POST'])
def query():
    start_time = time.time()
    query_counter.inc()

    try:
        answer = rag_plus.generate(question, task_type=task_type)
        query_duration.observe(time.time() - start_time)
        return jsonify({'answer': answer})
    except Exception as e:
        error_counter.inc()
        raise

@app.route('/metrics')
def metrics():
    return generate_latest()
```

### Health Checks

```python
@app.route('/health', methods=['GET'])
def health():
    checks = {
        'status': 'healthy',
        'corpus_loaded': len(rag_plus.retriever.knowledge_corpus) > 0,
        'llm_available': check_llm_health(),
        'embedding_available': check_embedding_health()
    }

    status_code = 200 if all(checks.values()) else 503
    return jsonify(checks), status_code

def check_llm_health():
    try:
        llm.generate("test", max_tokens=1)
        return True
    except:
        return False
```

---

## Scaling

### Horizontal Scaling

Load balance across multiple instances:

```
         ┌──────────────┐
         │ Load Balancer│
         └──────┬───────┘
                │
      ┌─────────┼─────────┐
      │         │         │
   ┌──▼──┐   ┌─▼───┐  ┌──▼──┐
   │API 1│   │API 2│  │API 3│
   └─────┘   └─────┘  └─────┘
```

#### With Nginx

```nginx
# nginx.conf
upstream rag_plus_backend {
    least_conn;
    server rag-plus-1:5000;
    server rag-plus-2:5000;
    server rag-plus-3:5000;
}

server {
    listen 80;

    location / {
        proxy_pass http://rag_plus_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Caching

Implement caching for repeated queries:

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cached_generate(query_hash, task_type):
    return rag_plus.generate(query_hash, task_type=task_type)

@app.route('/query', methods=['POST'])
def query():
    question = request.json.get('query')
    task_type = request.json.get('task_type', 'general')

    # Create cache key
    query_hash = hashlib.md5(question.encode()).hexdigest()

    # Use cached result if available
    answer = cached_generate(query_hash, task_type)
    return jsonify({'answer': answer})
```

### Database Integration

Store queries and responses:

```python
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Query(Base):
    __tablename__ = 'queries'

    id = Column(Integer, primary_key=True)
    query = Column(String)
    answer = Column(String)
    task_type = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine('postgresql://user:pass@localhost/ragplus')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

@app.route('/query', methods=['POST'])
def query():
    question = request.json.get('query')
    task_type = request.json.get('task_type', 'general')

    answer = rag_plus.generate(question, task_type=task_type)

    # Store in database
    session = Session()
    query_record = Query(
        query=question,
        answer=answer,
        task_type=task_type
    )
    session.add(query_record)
    session.commit()

    return jsonify({'answer': answer})
```

---

## Security

### API Authentication

```python
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.getenv('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/query', methods=['POST'])
@require_api_key
def query():
    # ... process query
    pass
```

### Rate Limiting

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

@app.route('/query', methods=['POST'])
@limiter.limit("10 per minute")
def query():
    # ... process query
    pass
```

### Input Validation

```python
from pydantic import BaseModel, ValidationError

class QueryRequest(BaseModel):
    query: str
    task_type: str = "general"

@app.route('/query', methods=['POST'])
def query():
    try:
        request_data = QueryRequest(**request.json)
    except ValidationError as e:
        return jsonify({'error': str(e)}), 400

    answer = rag_plus.generate(
        request_data.query,
        task_type=request_data.task_type
    )
    return jsonify({'answer': answer})
```

---

## Best Practices

1. **Pre-build Corpus**: Build corpus offline, load at startup
2. **Use Caching**: Cache repeated queries and embeddings
3. **Monitor Costs**: Track API usage and costs
4. **Set Timeouts**: Prevent hanging requests
5. **Implement Retries**: Handle transient API failures
6. **Load Balance**: Distribute traffic across instances
7. **Use Secrets Management**: Store API keys securely
8. **Enable HTTPS**: Encrypt data in transit
9. **Log Appropriately**: Balance detail with privacy
10. **Test Thoroughly**: Load test before production

---

## Troubleshooting

### High Latency
- Reduce `top_k` value
- Use faster LLM model
- Implement caching
- Scale horizontally

### High Costs
- Use gpt-3.5-turbo instead of gpt-4
- Reduce `max_tokens`
- Implement query deduplication
- Cache aggressively

### Memory Issues
- Use smaller embedding model
- Limit corpus size
- Implement lazy loading
- Use disk-based vector storage

---

## See Also

- [API Reference](API.md)
- [Architecture](ARCHITECTURE.md)
- [Quick Start](QUICKSTART.md)
