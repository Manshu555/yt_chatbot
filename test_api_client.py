import requests
import time
import sys

def test_api():
    base_url = "http://127.0.0.1:8000/ask"
    
    # 1. Normal Query
    payload_normal = {
        "query": "What is machine learning?",
        "video_id": "ukzFI9rgwfU",
        "validate_externally": False
    }
    print("Testing normal query...")
    t0 = time.time()
    try:
        res1 = requests.post(base_url, json=payload_normal, timeout=30)
        t1 = time.time()
        if res1.status_code == 200:
            data = res1.json()
            print(f"Normal query passed. Latency: {t1-t0:.2f}s")
        else:
            print("Normal query failed:", res1.text)
    except Exception as e:
        print("Normal query failed with exception:", e)
        
    # 2. Validated Query
    payload_validated = {
        "query": "Is this information correct?",
        "video_id": "ukzFI9rgwfU",
        "validate_externally": True
    }
    print("\nTesting validated query...")
    t0 = time.time()
    try:
        res2 = requests.post(base_url, json=payload_validated, timeout=30)
        t1 = time.time()
        if res2.status_code == 200:
            data = res2.json()
            print(f"Validated query passed. Latency: {t1-t0:.2f}s")
            print("Validation status:", data.get("validation_status"))
            print("Sources:", len(data.get("sources", [])))
        else:
            print("Validated query failed:", res2.text)
    except Exception as e:
        print("Validated query failed with exception:", e)

if __name__ == "__main__":
    test_api()
