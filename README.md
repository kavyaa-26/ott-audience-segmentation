# Containerized Audience Segmentation & Personalization Service

A lightweight, production-ready OTT audience intelligence and personalization service that learns behavioral audience groups from viewer activity using unsupervised machine learning (KMeans), predicts audience segments for new viewer profiles, and serves explainable content recommendations.

---

## Architecture Overview

The system strictly consists of three backend services running in isolated containers with a shared models volume:

```
[ USER ACTIVITY DATA ]
           │
           ▼
     ┌───────────┐
     │  TRAINER  │  Validation -> Preprocessing -> StandardScaler -> KMeans (Candidate K evaluation)
     └─────┬─────┘
           │
           ▼
    [ PERSISTED MODEL ] (Shared volume: audience_pipeline.joblib, segment_metadata.json)
           │
           ▼
     ┌───────────┐
     │    API    │  FastAPI: /health, /recommend, /api/segments, /api/status, and Embedded Frontend
     └─────┬─────┘
           │
           ▼
     ┌───────────┐
     │ EVALUATOR │  Independent HTTP client: health, profile tests, 7 edge cases -> metrics.json
     └───────────┘
```

1. **`trainer`**: Validates schema, cleans missing/duplicate data, extracts behavioral and multi-hot genre features, benchmarks candidate K values (2, 3, 4, 5, 6) using silhouette scores, fits the final KMeans model (`random_state=42`), generates cluster statistics, and persists model artifacts.
2. **`api`**: Fast, lightweight API built with FastAPI that loads the persisted pipeline into memory, handles viewer profile inputs, computes Euclidean distance to the predicted cluster centroid, selects catalog recommendations with transparent explanation reasons, and serves the neutral, responsive user interface.
3. **`evaluator`**: Independent HTTP test suite that waits for API readiness, executes recommendation queries against seed profiles, validates edge-case robustness, reads measured model metrics, and automatically writes `metrics.json`.

---

## Quickstart (Docker Compose)

Prerequisites: Docker Engine and Docker Compose.

To build and run the entire pipeline cleanly:

```bash
docker compose up --build
```

### Execution Lifecycle
1. The **trainer** container initializes, processes `data/synthetic_users.csv`, tests candidate K values, and writes `audience_pipeline.joblib` and `segment_metadata.json` to the shared `models_data` volume.
2. The **api** container starts, checks model availability, passes health checks, and begins serving the HTTP API and frontend at `http://localhost:8000`.
3. The **evaluator** container waits for the API health check to pass, sends verification test profiles and robustness edge cases over HTTP, and generates `metrics.json`.

---

## Local Development (Without Docker)

You can also run each component directly using Python:

### 1. Train Model
```bash
python trainer/train.py
```
Outputs:
- `models/audience_pipeline.joblib`
- `models/segment_metadata.json`

### 2. Start API & Frontend
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Open `http://localhost:8000` in your web browser.

### 3. Run Evaluator Test Suite
```bash
python evaluator/evaluate.py
```
Outputs:
- `metrics.json`

---

## API Specification

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response (Ready)**:
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```
- **Response (Not Ready)**:
  ```json
  {
    "status": "not_ready",
    "model_loaded": false
  }
  ```

### 2. Personalization & Recommendation
- **Endpoint**: `POST /recommend`
- **Request Body**:
  ```json
  {
    "user_id": "DEMO-ACTION-001",
    "watch_time_hours": 42.0,
    "avg_session_mins": 105.0,
    "visits_per_week": 28,
    "top_genres": ["Action", "Thriller"]
  }
  ```
- **Response Body**:
  ```json
  {
    "user_id": "DEMO-ACTION-001",
    "segment_id": 4,
    "segment_name": "High-Engagement Action Viewers",
    "recommendations": [
      {
        "title": "The Dark Knight",
        "genre": "Action",
        "reason": "Strong match for Action preference and high engagement"
      }
    ],
    "distance_to_centroid": 0.42
  }
  ```

### 3. Segments Metadata
- **Endpoint**: `GET /api/segments`
- Returns full cluster telemetry, candidate K evaluation metrics, and segment profiles.

### 4. System Status
- **Endpoint**: `GET /api/status`
- Returns active model configuration, preprocessor type, cluster count, and silhouette score.

---

## Custom Domain Deployment

This application is architected to operate behind any reverse proxy, load balancer, or custom domain without code modifications:
- All frontend API calls use **relative paths** (`/recommend`, `/health`, `/api/segments`, `/api/status`) rather than hardcoded URLs.
- To bind a custom domain in production:
  1. Route your domain (e.g. `ott-intelligence.yourdomain.com`) to the host running the container or cloud instance (e.g., via DNS A / CNAME records).
  2. Terminate TLS via reverse proxy (e.g., Nginx, Caddy, Traefik, or Cloudflare).
  3. Proxy requests to container port `8000`:
     ```nginx
     server {
         server_name ott-intelligence.yourdomain.com;
         location / {
             proxy_pass http://localhost:8000;
             proxy_set_header Host $host;
             proxy_set_header X-Real-IP $remote_addr;
         }
     }
     ```

---

## Privacy & Data Handling

- **Synthetic Data**: Training data and seed demonstration profiles are synthetically generated for demonstration. No real viewer personal data is processed or collected.
- **In-Memory Inference**: Recommendation inference executes strictly in-memory without logging or persisting viewer profiles to disk.
- **Zero External AI Services**: No paid external APIs, LLMs, or proprietary cloud endpoints are utilized.
