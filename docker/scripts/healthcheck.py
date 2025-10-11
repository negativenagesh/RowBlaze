#!/usr/bin/env python3
"""
Simple health check script for Docker containers.
"""
import sys
import requests
import time

def check_health(url, max_retries=3, retry_delay=1):
    """Check if a service is healthy by making an HTTP request."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"Health check passed for {url}")
                return True
            else:
                print(f"Health check failed with status code {response.status_code}")
        except Exception as e:
            print(f"Health check attempt {attempt + 1}/{max_retries} failed: {str(e)}")
        
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    
    return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: healthcheck.py <url>")
        sys.exit(1)
    
    url = sys.argv[1]
    if check_health(url):
        sys.exit(0)
    else:
        sys.exit(1)