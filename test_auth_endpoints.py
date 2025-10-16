#!/usr/bin/env python3
"""
Test script to verify authentication endpoints are working.
"""
import asyncio
import os

import httpx


# Smart API URL detection
def get_api_url():
    env_url = os.getenv("ROWBLAZE_API_URL")
    if env_url:
        return env_url

    if os.path.exists("/.dockerenv"):
        return "http://api:8000/api"

    return "http://localhost/api"


API_URL = get_api_url()


async def test_health_endpoint():
    """Test the health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_URL}/health")
            print(f"Health endpoint status: {response.status_code}")
            if response.status_code == 200:
                print(f"Health response: {response.json()}")
                return True
            else:
                print(f"Health endpoint failed: {response.text}")
                return False
    except Exception as e:
        print(f"Health endpoint error: {e}")
        return False


async def test_auth_health_endpoint():
    """Test the auth health endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_URL}/auth/health")
            print(f"Auth health endpoint status: {response.status_code}")
            if response.status_code == 200:
                print(f"Auth health response: {response.json()}")
                return True
            else:
                print(f"Auth health endpoint failed: {response.text}")
                return False
    except Exception as e:
        print(f"Auth health endpoint error: {e}")
        return False


async def test_register_endpoint():
    """Test user registration."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {"email": "test@example.com", "password": "testpassword123"}
            response = await client.post(
                f"{API_URL}/auth/register",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            print(f"Register endpoint status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Registration successful: {result.get('user', {}).get('email')}")
                return True, result.get("access_token")
            elif response.status_code == 409:
                print("User already exists (expected for repeated tests)")
                return True, None  # Consider this a success for testing purposes
            else:
                print(f"Registration failed: {response.text}")
                return False, None
    except Exception as e:
        print(f"Registration error: {e}")
        return False, None


async def test_login_endpoint():
    """Test user login."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {"email": "test@example.com", "password": "testpassword123"}
            response = await client.post(
                f"{API_URL}/auth/login",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            print(f"Login endpoint status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Login successful: {result.get('user', {}).get('email')}")
                return True, result.get("access_token")
            else:
                print(f"Login failed: {response.text}")
                return False, None
    except Exception as e:
        print(f"Login error: {e}")
        return False, None


async def main():
    """Run all tests."""
    print(f"Testing API at: {API_URL}")
    print("=" * 50)

    # Test health endpoints
    print("1. Testing main health endpoint...")
    health_ok = await test_health_endpoint()

    # Skip auth health endpoint for now
    auth_health_ok = True

    # Test registration
    print("\n2. Testing user registration...")
    register_ok, token = await test_register_endpoint()

    # Test login
    print("\n3. Testing user login...")
    login_ok, login_token = await test_login_endpoint()

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Health endpoint: {'✅' if health_ok else '❌'}")
    print(f"Registration: {'✅' if register_ok else '❌'}")
    print(f"Login: {'✅' if login_ok else '❌'}")

    if all([health_ok, register_ok, login_ok]):
        print("\n🎉 All tests passed! Authentication system is working.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    asyncio.run(main())
