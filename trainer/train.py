import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

ALL_GENRES = ["Action", "Thriller", "Sci-Fi", "Drama", "Romance", "Comedy", "Family", "Animation", "Mystery"]
NUMERICAL_COLS = ["watch_time_hours", "avg_session_mins", "visits_per_week", "session_intensity", "genre_diversity"]

def normalize_genres(genre_val) -> list:
    """Safely normalizes genre input from string or list, handling repeated entries and whitespace."""
    if isinstance(genre_val, list):
        raw_list = genre_val
    elif isinstance(genre_val, str):
        raw_list = [g.strip() for g in genre_val.split(",") if g.strip()]
    elif pd.isna(genre_val):
        return []
    else:
        return []
    
    cleaned = []
    seen = set()
    for g in raw_list:
        clean_g = str(g).strip().title()
        if clean_g and clean_g.lower() not in seen:
            seen.add(clean_g.lower())
            cleaned.append(clean_g)
    return cleaned

def generate_synthetic_data(output_path: str, n_samples: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate realistic viewer activity dataset without target segment labels."""
    rng = np.random.default_rng(seed)
    
    # 5 natural behavioral groups
    n1 = 120
    wt1 = rng.normal(40.0, 5.0, n1).clip(28.0, 58.0)
    sd1 = rng.normal(100.0, 12.0, n1).clip(75.0, 135.0)
    vf1 = rng.normal(26.0, 3.5, n1).clip(18.0, 34.0)
    genres1 = [", ".join(rng.choice(["Action", "Thriller", "Sci-Fi"], size=rng.integers(1, 3), replace=False)) for _ in range(n1)]

    n2 = 120
    wt2 = rng.normal(6.5, 1.8, n2).clip(3.0, 11.0)
    sd2 = rng.normal(20.0, 4.0, n2).clip(12.0, 30.0)
    vf2 = rng.normal(5.5, 1.5, n2).clip(3.0, 10.0)
    genres2 = [", ".join(rng.choice(["Comedy", "Family", "Animation"], size=rng.integers(1, 3), replace=False)) for _ in range(n2)]

    n3 = 120
    wt3 = rng.normal(22.0, 3.5, n3).clip(15.0, 29.0)
    sd3 = rng.normal(62.0, 7.0, n3).clip(45.0, 78.0)
    vf3 = rng.normal(16.0, 2.5, n3).clip(11.0, 21.0)
    genres3 = [", ".join(rng.choice(["Drama", "Romance"], size=rng.integers(1, 3), replace=False)) for _ in range(n3)]

    n4 = 120
    wt4 = rng.normal(2.5, 0.9, n4).clip(0.5, 4.8)
    sd4 = rng.normal(14.0, 2.5, n4).clip(8.0, 20.0)
    vf4 = rng.normal(2.2, 0.8, n4).clip(1.0, 4.0)
    genres4 = [rng.choice(["Family", "Drama", "Comedy", "Animation"]) for _ in range(n4)]

    n5 = n_samples - (n1 + n2 + n3 + n4)
    wt5 = rng.normal(30.0, 4.0, n5).clip(22.0, 39.0)
    sd5 = rng.normal(56.0, 6.0, n5).clip(42.0, 70.0)
    vf5 = rng.normal(21.0, 2.5, n5).clip(15.0, 27.0)
    genres5 = [", ".join(rng.choice(ALL_GENRES, size=rng.integers(3, 5), replace=False)) for _ in range(n5)]

    watch_times = np.concatenate([wt1, wt2, wt3, wt4, wt5])
    session_durations = np.concatenate([sd1, sd2, sd3, sd4, sd5])
    visits = np.concatenate([vf1, vf2, vf3, vf4, vf5])
    all_genres_list = genres1 + genres2 + genres3 + genres4 + genres5

    # Introduce realistic variations: repeated genres (e.g. "Drama, Drama") to test preprocessing
    for i in range(0, len(all_genres_list), 25):
        primary = all_genres_list[i].split(",")[0].strip()
        all_genres_list[i] = f"{primary}, {primary}"

    # Shuffle dataset
    indices = np.arange(len(watch_times))
    rng.shuffle(indices)

    user_ids = [f"USER-{1000 + i:04d}" for i in range(len(watch_times))]

    df = pd.DataFrame({
        "user_id": user_ids,
        "watch_time_hours": np.round(watch_times[indices], 1),
        "avg_session_mins": np.round(session_durations[indices], 1),
        "visits_per_week": np.round(visits[indices]).astype(int),
        "top_genres": [all_genres_list[idx] for idx in indices]
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated synthetic dataset with {len(df)} users at {output_path}")
    return df

def extract_features(df_or_record, all_genres=ALL_GENRES) -> tuple[np.ndarray, list[str]]:
    """Transform raw viewer activity inputs into numerical feature matrix."""
    if isinstance(df_or_record, dict):
        df = pd.DataFrame([df_or_record])
    else:
        df = df_or_record.copy()
    
    # 1. Behavioral features
    watch_time = df["watch_time_hours"].astype(float)
    avg_session = df["avg_session_mins"].astype(float)
    visits = df["visits_per_week"].astype(float)
    
    # Session intensity: hours per visit
    safe_visits = np.where(visits > 0, visits, 1.0)
    session_intensity = watch_time / safe_visits
    
    # Normalized genres
    norm_genres = df["top_genres"].apply(normalize_genres)
    genre_diversity = norm_genres.apply(len).astype(float)
    
    feature_dict = {
        "watch_time_hours": watch_time,
        "avg_session_mins": avg_session,
        "visits_per_week": visits,
        "session_intensity": session_intensity,
        "genre_diversity": genre_diversity
    }
    
    # Genre multi-hot columns
    for g in all_genres:
        col_name = f"genre_{g.lower()}"
        feature_dict[col_name] = norm_genres.apply(lambda gl: 1.0 if g.title() in gl else 0.0)
        
    features_df = pd.DataFrame(feature_dict)
    feature_cols = list(features_df.columns)
    return features_df.values, feature_cols

def generate_segment_name(avg_watch_time: float, avg_session: float, avg_visits: float, dominant_genres: list) -> str:
    """Generate human-readable segment names from actual cluster characteristics."""
    # Determine engagement tier
    if avg_watch_time >= 32.0 and avg_visits >= 20.0:
        tier = "High-Engagement"
    elif avg_watch_time <= 4.0 or (avg_visits <= 3.0 and avg_session <= 18.0):
        tier = "Low-Activity"
    elif avg_session <= 24.0 and avg_watch_time <= 10.0:
        tier = "Casual Short-Session"
    elif avg_watch_time >= 22.0:
        tier = "Active"
    else:
        tier = "Moderate"

    # Determine genre focus
    top_two = dominant_genres[:2] if dominant_genres else []
    if "Action" in top_two and ("Thriller" in top_two or "Sci-Fi" in top_two):
        genre_label = "Action"
    elif "Action" in top_two:
        genre_label = "Action"
    elif "Thriller" in top_two or "Mystery" in top_two:
        genre_label = "Thriller & Mystery"
    elif "Drama" in top_two and "Romance" in top_two:
        genre_label = "Drama & Romance"
    elif "Drama" in top_two:
        genre_label = "Drama Focused"
    elif "Animation" in top_two:
        genre_label = "Animation & Family"
    elif "Family" in top_two or "Comedy" in top_two:
        genre_label = "Family & Comedy"
    elif len(dominant_genres) >= 4:
        genre_label = "Genre Explorers"
    elif top_two:
        genre_label = " & ".join(top_two)
    else:
        genre_label = "General"

    if tier == "Casual Short-Session":
        return f"Casual Short-Session {genre_label} Viewers"
    elif tier == "High-Engagement":
        return f"High-Engagement {genre_label} Viewers"
    elif tier == "Low-Activity":
        return f"Low-Activity {genre_label} Viewers"
    elif genre_label == "Genre Explorers":
        return "Active Genre Explorers"
    else:
        return f"{tier} {genre_label} Viewers"

def train():
    parser = argparse.ArgumentParser(description="Train OTT audience clustering model")
    parser.add_argument("--data-path", default=None, help="Path to synthetic_users.csv")
    parser.add_argument("--output-dir", default=None, help="Directory to save model artifacts")
    parser.add_argument("--random-state", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    # Determine base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = args.data_path or os.path.join(base_dir, "data", "synthetic_users.csv")
    output_dir = args.output_dir or os.path.join(base_dir, "models")
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Starting Trainer Service ===")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")

    # 1. Load or generate synthetic dataset
    if not os.path.exists(data_path):
        print(f"Synthetic dataset not found at {data_path}. Generating now...")
        df = generate_synthetic_data(data_path, n_samples=600, seed=args.random_state)
    else:
        print(f"Loading synthetic dataset from {data_path}...")
        df = pd.read_csv(data_path)

    # 2. Validate schema
    required_columns = {"user_id", "watch_time_hours", "avg_session_mins", "visits_per_week", "top_genres"}
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    initial_count = len(df)
    print(f"Loaded {initial_count} records. Validating schema and data...")

    # 3. Detect and handle missing values
    df = df.dropna(subset=["user_id", "watch_time_hours", "avg_session_mins", "visits_per_week"])

    # 4. Handle invalid numeric values
    for col in ["watch_time_hours", "avg_session_mins", "visits_per_week"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["watch_time_hours", "avg_session_mins", "visits_per_week"])
    df = df[(df["watch_time_hours"] >= 0) & (df["avg_session_mins"] >= 0) & (df["visits_per_week"] >= 0)]

    # 5. Handle duplicates
    df = df.drop_duplicates(subset=["user_id"], keep="first")
    print(f"Validation complete: {len(df)} clean records available for training.")

    # 6, 7, 8: Feature extraction and genre encoding
    X_raw, feature_cols = extract_features(df, all_genres=ALL_GENRES)
    print(f"Extracted {len(feature_cols)} features: {feature_cols}")

    # 9. Standardize numerical features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 10. Test K values 2, 3, 4, 5, and 6
    k_candidates = [2, 3, 4, 5, 6]
    candidate_results = []
    print("\n--- Evaluating Candidate K Values ---")
    print(f"{'K':<4} {'Silhouette Score':<18} {'Inertia':<15}")
    print("-" * 40)

    best_k = None
    best_silhouette = -1.0
    best_inertia = None

    for k in k_candidates:
        km = KMeans(n_clusters=k, random_state=args.random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = float(silhouette_score(X_scaled, labels))
        inertia = float(km.inertia_)
        candidate_results.append({
            "k": k,
            "silhouette": round(sil, 4),
            "inertia": round(inertia, 2)
        })
        print(f"{k:<4} {sil:<18.4f} {inertia:<15.2f}")
        
        # 12. Select defensible K using actual measured evidence (highest silhouette score)
        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k
            best_inertia = inertia

    print(f"\nDefensible optimal K selected from evidence: K = {best_k} (Silhouette: {best_silhouette:.4f})")

    # 13. Train final KMeans model using random_state=42
    final_kmeans = KMeans(n_clusters=best_k, random_state=args.random_state, n_init=10)
    final_labels = final_kmeans.fit_predict(X_scaled)
    df["cluster_id"] = final_labels

    # 14. Inspect cluster sizes
    cluster_counts = df["cluster_id"].value_counts().to_dict()
    cluster_sizes = {str(c): int(count) for c, count in sorted(cluster_counts.items())}
    print(f"Cluster sizes: {cluster_sizes}")

    # 15 & 16. Generate cluster profiles and human-readable segment names
    clusters_meta = []
    for c_id in range(best_k):
        c_df = df[df["cluster_id"] == c_id]
        c_size = int(len(c_df))
        avg_wt = float(round(c_df["watch_time_hours"].mean(), 2))
        avg_sess = float(round(c_df["avg_session_mins"].mean(), 2))
        avg_vis = float(round(c_df["visits_per_week"].mean(), 2))

        # Compute genre frequencies within this cluster
        genre_freq = {}
        for gl in c_df["top_genres"].apply(normalize_genres):
            for g in gl:
                genre_freq[g] = genre_freq.get(g, 0) + 1
        
        sorted_genres = sorted(genre_freq.items(), key=lambda x: x[1], reverse=True)
        dominant_genres = [g for g, _ in sorted_genres[:4]]

        # Generate segment name from actual characteristics
        segment_name = generate_segment_name(avg_wt, avg_sess, avg_vis, dominant_genres)

        cluster_profile = {
            "cluster_id": int(c_id),
            "segment_name": segment_name,
            "cluster_size": c_size,
            "avg_watch_time": avg_wt,
            "avg_session_mins": avg_sess,
            "avg_visit_frequency": avg_vis,
            "dominant_genres": dominant_genres
        }
        clusters_meta.append(cluster_profile)
        print(f"Cluster {c_id}: '{segment_name}' | Users: {c_size} | Avg Watch: {avg_wt}h | Avg Sess: {avg_sess}m | Genres: {dominant_genres}")

    # 17. Persist complete inference preprocessing + model pipeline
    pipeline_artifact = {
        "scaler": scaler,
        "kmeans": final_kmeans,
        "feature_cols": feature_cols,
        "all_genres": ALL_GENRES,
        "best_k": best_k,
        "random_state": args.random_state
    }
    pipeline_path = os.path.join(output_dir, "audience_pipeline.joblib")
    joblib.dump(pipeline_artifact, pipeline_path)
    print(f"\nPersisted pipeline artifact to {pipeline_path}")

    # 18. Persist segment metadata
    metadata = {
        "model_type": "KMeans",
        "preprocessor": "StandardScaler",
        "num_clusters": best_k,
        "silhouette_score": round(best_silhouette, 4),
        "inertia": round(best_inertia, 2),
        "cluster_sizes": cluster_sizes,
        "candidate_evaluations": candidate_results,
        "clusters": clusters_meta
    }
    metadata_path = os.path.join(output_dir, "segment_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Persisted segment metadata to {metadata_path}")
    print("=== Training Complete Successfully ===")

if __name__ == "__main__":
    train()
