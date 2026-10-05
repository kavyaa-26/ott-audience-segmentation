import os
import sys
import time
import json
import httpx

def find_file(relative_options):
    for path in relative_options:
        if os.path.exists(path):
            return os.path.abspath(path)
    return os.path.abspath(relative_options[0])

def main():
    api_url = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    models_dir = os.environ.get("MODELS_DIR", os.path.join(base_dir, "models"))
    metadata_path = os.path.join(models_dir, "segment_metadata.json")
    
    metrics_path = os.environ.get(
        "METRICS_PATH",
        find_file([
            os.path.join(base_dir, "metrics.json"),
            os.path.join(os.getcwd(), "metrics.json")
        ])
    )

    print("==================================================")
    print("   OTT Audience Segmentation - Evaluator Service   ")
    print("==================================================")
    print(f"Target API URL:   {api_url}")
    print(f"Models Directory: {models_dir}")
    print(f"Output Metrics:   {metrics_path}")
    print("--------------------------------------------------")

    client = httpx.Client(timeout=10.0)

    # 1. Wait for API readiness (Calling /health)
    print("\n[Step 1] Waiting for API readiness at /health...")
    api_ready = False
    max_retries = 30
    retry_interval = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.get(f"{api_url}/health")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "ok" and data.get("model_loaded") is True:
                    print(f"  -> API is healthy and model is loaded (Attempt {attempt})")
                    api_ready = True
                    break
                else:
                    print(f"  -> API reachable but model not ready: {data} (Attempt {attempt})")
            else:
                print(f"  -> Health check returned HTTP {resp.status_code} (Attempt {attempt})")
        except Exception as e:
            print(f"  -> Connection waiting: {e.__class__.__name__} (Attempt {attempt}/{max_retries})")
        time.sleep(retry_interval)

    if not api_ready:
        print("ERROR: API failed to reach ready state within timeout period.")
        sys.exit(1)

    health_check_passed = True

    # 2. Test valid viewer profiles through HTTP
    print("\n[Step 2] Testing valid viewer profiles via POST /recommend...")
    test_profiles = [
        {
            "user_id": "DEMO-ACTION-001",
            "watch_time_hours": 42.0,
            "avg_session_mins": 105.0,
            "visits_per_week": 28,
            "top_genres": ["Action", "Thriller"]
        },
        {
            "user_id": "DEMO-CASUAL-001",
            "watch_time_hours": 5.0,
            "avg_session_mins": 18.0,
            "visits_per_week": 5,
            "top_genres": ["Comedy", "Family"]
        },
        {
            "user_id": "DEMO-FOCUSED-001",
            "watch_time_hours": 24.0,
            "avg_session_mins": 60.0,
            "visits_per_week": 17,
            "top_genres": ["Drama", "Drama"]  # Repeated genre check
        },
        {
            "user_id": "DEMO-EXPLORER-001",
            "watch_time_hours": 30.0,
            "avg_session_mins": 55.0,
            "visits_per_week": 22,
            "top_genres": ["Action", "Comedy", "Drama", "Sci-Fi"]
        },
        {
            "user_id": "DEMO-LOW-001",
            "watch_time_hours": 2.0,
            "avg_session_mins": 12.0,
            "visits_per_week": 2,
            "top_genres": ["Family"]
        }
    ]

    valid_requests_passed = True
    first_response = None

    for profile in test_profiles:
        try:
            resp = client.post(f"{api_url}/recommend", json=profile)
            if resp.status_code != 200:
                print(f"  FAIL: Profile {profile['user_id']} returned HTTP {resp.status_code}")
                valid_requests_passed = False
                continue

            res_json = resp.json()
            # Validate response schema
            required_keys = {"user_id", "segment_id", "segment_name", "recommendations", "distance_to_centroid"}
            if not required_keys.issubset(res_json.keys()):
                print(f"  FAIL: Response missing required keys: {required_keys - set(res_json.keys())}")
                valid_requests_passed = False
                continue

            if not isinstance(res_json["recommendations"], list) or len(res_json["recommendations"]) == 0:
                print(f"  FAIL: Recommendations list is empty for {profile['user_id']}")
                valid_requests_passed = False
                continue

            # Check recommendation structure
            rec = res_json["recommendations"][0]
            if "title" not in rec or "genre" not in rec or "reason" not in rec:
                print(f"  FAIL: Recommendation item missing title/genre/reason: {rec}")
                valid_requests_passed = False
                continue

            if first_response is None and profile["user_id"] == "DEMO-ACTION-001":
                first_response = res_json

            print(f"  PASS: {profile['user_id']} -> Segment {res_json['segment_id']} ('{res_json['segment_name']}'), dist={res_json['distance_to_centroid']}, {len(res_json['recommendations'])} recs")
        except Exception as e:
            print(f"  FAIL: Exception calling /recommend: {e}")
            valid_requests_passed = False

    # Test repeatability / stability: send DEMO-ACTION-001 again
    print("\n[Step 3] Testing repeated request stability...")
    try:
        repeat_resp = client.post(f"{api_url}/recommend", json=test_profiles[0])
        repeat_json = repeat_resp.json()
        if (repeat_resp.status_code == 200 and
            repeat_json["segment_id"] == first_response["segment_id"] and
            repeat_json["distance_to_centroid"] == first_response["distance_to_centroid"]):
            print("  PASS: Repeated request produced identical, stable output.")
        else:
            print(f"  FAIL: Repeated request output differed: {repeat_json} vs {first_response}")
            valid_requests_passed = False
    except Exception as e:
        print(f"  FAIL: Repeated request error: {e}")
        valid_requests_passed = False

    # 4. Test required robustness edge cases
    print("\n[Step 4] Testing robustness edge cases...")
    robustness_results = {}

    # 4a. Unknown genre
    try:
        resp = client.post(f"{api_url}/recommend", json={
            "user_id": "TEST-EDGE-UNKNOWN",
            "watch_time_hours": 20.0,
            "avg_session_mins": 45.0,
            "visits_per_week": 10,
            "top_genres": ["CyberpunkGenre", "NonExistentGenre"]
        })
        robustness_results["unknown_genre"] = (resp.status_code == 200 and "recommendations" in resp.json())
        print(f"  {'PASS' if robustness_results['unknown_genre'] else 'FAIL'}: Unknown genre handled gracefully (HTTP {resp.status_code})")
    except Exception as e:
        robustness_results["unknown_genre"] = False
        print(f"  FAIL: Unknown genre raised exception: {e}")

    # 4b. Empty genres
    try:
        resp = client.post(f"{api_url}/recommend", json={
            "user_id": "TEST-EDGE-EMPTY-GENRES",
            "watch_time_hours": 15.0,
            "avg_session_mins": 30.0,
            "visits_per_week": 7,
            "top_genres": []
        })
        robustness_results["empty_genres"] = (resp.status_code == 200 and "recommendations" in resp.json())
        print(f"  {'PASS' if robustness_results['empty_genres'] else 'FAIL'}: Empty genres handled gracefully (HTTP {resp.status_code})")
    except Exception as e:
        robustness_results["empty_genres"] = False
        print(f"  FAIL: Empty genres raised exception: {e}")

    # 4c. Zero watch time
    try:
        resp = client.post(f"{api_url}/recommend", json={
            "user_id": "TEST-EDGE-ZERO-WATCH",
            "watch_time_hours": 0.0,
            "avg_session_mins": 0.0,
            "visits_per_week": 0,
            "top_genres": ["Action"]
        })
        robustness_results["zero_watch_time"] = (resp.status_code == 200 and "recommendations" in resp.json())
        print(f"  {'PASS' if robustness_results['zero_watch_time'] else 'FAIL'}: Zero watch time handled cleanly (HTTP {resp.status_code})")
    except Exception as e:
        robustness_results["zero_watch_time"] = False
        print(f"  FAIL: Zero watch time raised exception: {e}")

    # 4d. Large watch time
    try:
        resp = client.post(f"{api_url}/recommend", json={
            "user_id": "TEST-EDGE-LARGE-WATCH",
            "watch_time_hours": 1000.0,
            "avg_session_mins": 500.0,
            "visits_per_week": 150,
            "top_genres": ["Sci-Fi"]
        })
        robustness_results["large_watch_time"] = (resp.status_code == 200 and "recommendations" in resp.json())
        print(f"  {'PASS' if robustness_results['large_watch_time'] else 'FAIL'}: Large watch time handled cleanly (HTTP {resp.status_code})")
    except Exception as e:
        robustness_results["large_watch_time"] = False
        print(f"  FAIL: Large watch time raised exception: {e}")

    # 4e. Missing field
    try:
        resp = client.post(f"{api_url}/recommend", json={
            # Missing user_id
            "watch_time_hours": 25.0,
            "avg_session_mins": 45.0,
            "visits_per_week": 10,
            "top_genres": ["Drama"]
        })
        robustness_results["missing_field"] = (resp.status_code == 422 and "detail" in resp.json())
        print(f"  {'PASS' if robustness_results['missing_field'] else 'FAIL'}: Missing field safely caught (HTTP {resp.status_code})")
    except Exception as e:
        robustness_results["missing_field"] = False
        print(f"  FAIL: Missing field raised unhandled exception: {e}")

    # 4f. Invalid numeric type
    try:
        resp = client.post(f"{api_url}/recommend", json={
            "user_id": "TEST-EDGE-INVALID-TYPE",
            "watch_time_hours": "non_numeric_string",
            "avg_session_mins": 45.0,
            "visits_per_week": 10,
            "top_genres": ["Drama"]
        })
        robustness_results["invalid_numeric_type"] = (resp.status_code == 422 and "detail" in resp.json())
        print(f"  {'PASS' if robustness_results['invalid_numeric_type'] else 'FAIL'}: Invalid numeric type safely caught (HTTP {resp.status_code})")
    except Exception as e:
        robustness_results["invalid_numeric_type"] = False
        print(f"  FAIL: Invalid numeric type raised unhandled exception: {e}")

    # 4g. Negative value
    try:
        resp = client.post(f"{api_url}/recommend", json={
            "user_id": "TEST-EDGE-NEGATIVE",
            "watch_time_hours": -15.0,
            "avg_session_mins": 45.0,
            "visits_per_week": 10,
            "top_genres": ["Drama"]
        })
        robustness_results["negative_value"] = (resp.status_code in [400, 422] and "detail" in resp.json())
        print(f"  {'PASS' if robustness_results['negative_value'] else 'FAIL'}: Negative value safely caught (HTTP {resp.status_code})")
    except Exception as e:
        robustness_results["negative_value"] = False
        print(f"  FAIL: Negative value raised unhandled exception: {e}")

    # 5. Read actual clustering metrics from segment_metadata.json
    print("\n[Step 5] Reading measured model metrics from segment_metadata.json...")
    if not os.path.exists(metadata_path):
        print(f"ERROR: Metadata not found at {metadata_path}")
        sys.exit(1)

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    measured_clustering = {
        "num_clusters": meta_data["num_clusters"],
        "silhouette_score": meta_data["silhouette_score"],
        "inertia": meta_data["inertia"],
        "cluster_sizes": meta_data["cluster_sizes"]
    }
    print(f"  Measured K: {measured_clustering['num_clusters']}")
    print(f"  Measured Silhouette: {measured_clustering['silhouette_score']}")
    print(f"  Measured Inertia: {measured_clustering['inertia']}")
    print(f"  Cluster Sizes: {measured_clustering['cluster_sizes']}")

    # 6. Generate metrics.json automatically matching Section 26
    metrics_output = {
        "clustering": measured_clustering,
        "api": {
            "health_check": health_check_passed,
            "valid_requests_passed": valid_requests_passed
        },
        "robustness": robustness_results,
        "reproducibility": {
            "fixed_seed": True
        }
    }

    os.makedirs(os.path.dirname(os.path.abspath(metrics_path)), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    print(f"\n[Step 6] Generated metrics.json automatically at {metrics_path}")
    print("==================================================")
    print("              Evaluator Completed                 ")
    print("==================================================")

if __name__ == "__main__":
    main()
