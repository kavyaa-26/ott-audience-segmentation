import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator
import joblib

# Determine directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(PROJECT_DIR, "models"))
CONTENT_PATH = os.path.join(BASE_DIR, "content", "movies.json")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(
    title="OTT Audience Segmentation & Personalization Service",
    description="Lightweight OTT audience intelligence and personalization service using KMeans clustering.",
    version="1.0.0"
)

# Custom exception handler to return clean user-safe error messages without stack traces
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    msg = first_error.get("msg", "Invalid request input")
    loc = ".".join(str(l) for l in first_error.get("loc", []) if l != "body")
    detail_msg = f"Validation error on '{loc}': {msg}" if loc else msg
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": detail_msg}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc)}
    )

# Model state holders
pipeline_artifact = None
segment_metadata = None
movie_catalog = []

def load_catalog():
    global movie_catalog
    if os.path.exists(CONTENT_PATH):
        try:
            with open(CONTENT_PATH, "r", encoding="utf-8") as f:
                movie_catalog = json.load(f)
            print(f"Loaded {len(movie_catalog)} catalog items from {CONTENT_PATH}")
        except Exception as e:
            print(f"Error loading content catalog: {e}")
            movie_catalog = []

def try_load_model():
    global pipeline_artifact, segment_metadata
    pipeline_path = os.path.join(MODELS_DIR, "audience_pipeline.joblib")
    metadata_path = os.path.join(MODELS_DIR, "segment_metadata.json")

    if os.path.exists(pipeline_path) and os.path.exists(metadata_path):
        try:
            pipeline_artifact = joblib.load(pipeline_path)
            with open(metadata_path, "r", encoding="utf-8") as f:
                segment_metadata = json.load(f)
            print("Successfully loaded model pipeline and segment metadata.")
            return True
        except Exception as e:
            print(f"Error loading persisted model: {e}")
            return False
    return False

# Attempt initial load on startup
@app.on_event("startup")
def startup_event():
    load_catalog()
    try_load_model()

# Helper for genre normalization
def normalize_genres(raw_genres: Union[list, str]) -> list:
    if isinstance(raw_genres, list):
        items = raw_genres
    elif isinstance(raw_genres, str):
        items = raw_genres.split(",")
    else:
        return []
    
    cleaned = []
    seen = set()
    for g in items:
        clean_g = str(g).strip().title()
        if clean_g and clean_g.lower() not in seen:
            seen.add(clean_g.lower())
            cleaned.append(clean_g)
    return cleaned

# Seed / Demo users specification
DEMO_USERS = [
    {
        "user_id": "DEMO-ACTION-001",
        "name": "High Engagement Action",
        "watch_time_hours": 42.0,
        "avg_session_mins": 105.0,
        "visits_per_week": 28,
        "top_genres": ["Action", "Thriller"],
        "expected_behavior": "High engagement, long sessions, frequent activity"
    },
    {
        "user_id": "DEMO-CASUAL-001",
        "name": "Casual Viewer",
        "watch_time_hours": 5.0,
        "avg_session_mins": 18.0,
        "visits_per_week": 5,
        "top_genres": ["Comedy", "Family"],
        "expected_behavior": "Low watch time, short sessions, low frequency"
    },
    {
        "user_id": "DEMO-FOCUSED-001",
        "name": "Genre-Focused",
        "watch_time_hours": 24.0,
        "avg_session_mins": 60.0,
        "visits_per_week": 17,
        "top_genres": ["Drama", "Drama"],
        "expected_behavior": "Genre concentration, moderate/high engagement"
    },
    {
        "user_id": "DEMO-LOW-001",
        "name": "Low Activity",
        "watch_time_hours": 2.0,
        "avg_session_mins": 12.0,
        "visits_per_week": 2,
        "top_genres": ["Family"],
        "expected_behavior": "Very low activity"
    },
    {
        "user_id": "DEMO-EXPLORER-001",
        "name": "Genre Explorer",
        "watch_time_hours": 30.0,
        "avg_session_mins": 55.0,
        "visits_per_week": 22,
        "top_genres": ["Action", "Comedy", "Drama", "Sci-Fi"],
        "expected_behavior": "High activity, diverse preferences"
    },
    {
        "user_id": "DEMO-DRAMA-001",
        "name": "Drama Viewer",
        "watch_time_hours": 21.0,
        "avg_session_mins": 65.0,
        "visits_per_week": 15,
        "top_genres": ["Romance", "Drama"],
        "expected_behavior": "Moderate/high drama/romance engagement"
    },
    {
        "user_id": "DEMO-THRILLER-001",
        "name": "Thriller Viewer",
        "watch_time_hours": 29.0,
        "avg_session_mins": 72.0,
        "visits_per_week": 19,
        "top_genres": ["Thriller", "Mystery"],
        "expected_behavior": "High thriller/mystery engagement"
    },
    {
        "user_id": "DEMO-FAMILY-001",
        "name": "Family Viewer",
        "watch_time_hours": 10.0,
        "avg_session_mins": 35.0,
        "visits_per_week": 8,
        "top_genres": ["Family", "Animation"],
        "expected_behavior": "Moderate/low family viewing"
    }
]

# Request & Response schemas
class RecommendRequest(BaseModel):
    user_id: str = Field(..., description="Unique viewer identifier")
    watch_time_hours: float = Field(..., description="Total watch time in hours")
    avg_session_mins: float = Field(..., description="Average session duration in minutes")
    visits_per_week: float = Field(..., description="Weekly visit frequency")
    top_genres: List[str] = Field(..., description="List of preferred genres")

    @field_validator("watch_time_hours")
    def validate_watch_time(cls, v):
        if v < 0:
            raise ValueError("watch_time_hours cannot be negative")
        return v

    @field_validator("avg_session_mins")
    def validate_session(cls, v):
        if v < 0:
            raise ValueError("avg_session_mins cannot be negative")
        return v

    @field_validator("visits_per_week")
    def validate_visits(cls, v):
        if v < 0:
            raise ValueError("visits_per_week cannot be negative")
        return v

class MovieRecommendation(BaseModel):
    title: str
    genre: str
    reason: str

class RecommendResponse(BaseModel):
    user_id: str
    segment_id: int
    segment_name: str
    recommendations: List[MovieRecommendation]
    distance_to_centroid: float

# Required Endpoint 1: GET /health
@app.get("/health")
def health_check():
    # If not loaded, attempt lazy load in case trainer recently completed
    if pipeline_artifact is None or segment_metadata is None:
        try_load_model()

    if pipeline_artifact is not None and segment_metadata is not None:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ok", "model_loaded": True}
        )
    else:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "model_loaded": False}
        )

# Required Endpoint 2: POST /recommend
@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(payload: RecommendRequest):
    # Ensure model is available
    if pipeline_artifact is None or segment_metadata is None:
        if not try_load_model():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not ready. Trainer may still be executing."
            )

    scaler = pipeline_artifact["scaler"]
    kmeans = pipeline_artifact["kmeans"]
    all_genres = pipeline_artifact["all_genres"]
    feature_cols = pipeline_artifact["feature_cols"]

    # Normalize genres safely (including repeated genres like ['Drama', 'Drama'] or whitespace)
    cleaned_genres = normalize_genres(payload.top_genres)
    
    # Feature extraction matching trainer
    watch_time = float(payload.watch_time_hours)
    avg_session = float(payload.avg_session_mins)
    visits = float(payload.visits_per_week)
    
    safe_visits = visits if visits > 0 else 1.0
    session_intensity = watch_time / safe_visits
    genre_diversity = float(len(cleaned_genres))

    features_dict = {
        "watch_time_hours": watch_time,
        "avg_session_mins": avg_session,
        "visits_per_week": visits,
        "session_intensity": session_intensity,
        "genre_diversity": genre_diversity
    }

    for g in all_genres:
        col_name = f"genre_{g.lower()}"
        features_dict[col_name] = 1.0 if g.title() in cleaned_genres else 0.0

    # Build ordered vector
    feature_vector = np.array([[features_dict[col] for col in feature_cols]], dtype=float)

    # Standardize using fitted scaler
    scaled_vector = scaler.transform(feature_vector)

    # Predict cluster (do NOT retrain inside POST /recommend)
    cluster_id = int(kmeans.predict(scaled_vector)[0])

    # Calculate Euclidean distance to predicted cluster centroid
    centroid = kmeans.cluster_centers_[cluster_id]
    dist_to_centroid = float(np.linalg.norm(scaled_vector - centroid))
    dist_rounded = round(dist_to_centroid, 2)

    # Look up segment metadata
    clusters_info = {c["cluster_id"]: c for c in segment_metadata.get("clusters", [])}
    cluster_profile = clusters_info.get(cluster_id, {
        "segment_name": f"Segment {cluster_id}",
        "dominant_genres": []
    })
    segment_name = cluster_profile.get("segment_name", f"Segment {cluster_id}")
    dominant_genres = cluster_profile.get("dominant_genres", [])

    # Recommendation Engine: Rank movies
    scored_movies = []
    viewer_genres_lower = {g.lower() for g in cleaned_genres}
    dominant_genres_lower = {g.lower() for g in dominant_genres}

    for movie in movie_catalog:
        movie_genres = [str(g) for g in movie.get("genre", [])]
        movie_genres_lower = {g.lower() for g in movie_genres}

        # 1. Viewer genre match score
        viewer_match_count = len(viewer_genres_lower.intersection(movie_genres_lower))
        score = viewer_match_count * 3.0

        # 2. Cluster dominant genre match score
        cluster_match_count = len(dominant_genres_lower.intersection(movie_genres_lower))
        score += cluster_match_count * 1.5

        # 3. Session duration alignment bonus (up to 1.0 point)
        duration = float(movie.get("duration_mins", 120))
        diff = abs(duration - avg_session)
        max_dim = max(duration, avg_session, 1.0)
        session_fit = max(0.0, 1.0 - (diff / max_dim))
        score += session_fit * 0.8

        # 4. Content popularity weighting
        popularity = float(movie.get("popularity", 0.8))
        score += popularity * 1.2

        # Formulation of transparent, explainable reason
        matched_viewer_genres = [g for g in movie_genres if g.lower() in viewer_genres_lower]
        matched_cluster_genres = [g for g in movie_genres if g.lower() in dominant_genres_lower]

        primary_movie_genre = movie_genres[0] if movie_genres else "General"

        if matched_viewer_genres:
            genre_str = ", ".join(matched_viewer_genres)
            if watch_time >= 25.0:
                reason = f"Strong match for {genre_str} preference and high engagement"
            else:
                reason = f"Direct match for your {genre_str} preference and {segment_name} profile"
        elif matched_cluster_genres:
            genre_str = ", ".join(matched_cluster_genres)
            reason = f"Recommended based on popular {genre_str} titles in the {segment_name} segment"
        elif cleaned_genres:
            reason = f"Popular feature aligned with your {segment_name} viewing habits"
        else:
            reason = f"Broad appeal catalog title aligned with the {segment_name} audience segment"

        scored_movies.append({
            "title": movie.get("title", "Untitled"),
            "genre": primary_movie_genre,
            "reason": reason,
            "score": score
        })

    # Sort movies descending by score and pick top 4
    scored_movies.sort(key=lambda m: m["score"], reverse=True)
    top_recommendations = [
        MovieRecommendation(
            title=m["title"],
            genre=m["genre"],
            reason=m["reason"]
        )
        for m in scored_movies[:4]
    ]

    return RecommendResponse(
        user_id=payload.user_id,
        segment_id=cluster_id,
        segment_name=segment_name,
        recommendations=top_recommendations,
        distance_to_centroid=dist_rounded
    )

# Additional API endpoints for frontend metadata & state
@app.get("/api/segments")
def get_segments():
    if segment_metadata is None:
        try_load_model()
    if segment_metadata is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Segment metadata unavailable")
    return segment_metadata

@app.get("/api/status")
def get_system_status():
    model_loaded = (pipeline_artifact is not None and segment_metadata is not None)
    if not model_loaded:
        model_loaded = try_load_model()

    return {
        "model": "KMeans",
        "preprocessing": "StandardScaler",
        "clusters": segment_metadata.get("num_clusters") if segment_metadata else None,
        "silhouette_score": segment_metadata.get("silhouette_score") if segment_metadata else None,
        "inertia": segment_metadata.get("inertia") if segment_metadata else None,
        "model_status": "Loaded" if model_loaded else "Not Ready",
        "api_status": "Operational"
    }

@app.get("/api/demo-users")
def get_demo_users():
    return DEMO_USERS

# Static & Frontend routes
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "OTT Audience Segmentation API is running."}

@app.get("/favicon.svg")
def serve_favicon():
    favicon_path = os.path.join(STATIC_DIR, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    raise HTTPException(status_code=404, detail="Favicon not found")

@app.get("/privacy")
def serve_privacy():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Privacy Policy"}

@app.get("/terms")
def serve_terms():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Terms & Conditions"}
