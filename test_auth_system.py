#!/usr/bin/env python3
"""
Test script for the new email-based authentication system.
Run this to verify the authentication endpoints work correctly.
"""

import asyncio
import json

import httpx

API_URL = "http://localhost:8000/api"


async def test_auth_system():
    """Test the complete authentication flow."""
    print("🧪 Testing RowBlaze Authentication System")
    print("=" * 50)

    async with httpx.AsyncClient() as client:
        # Test 1: Register a new user
        print("\n1️⃣ Testing User Registration")
        register_data = {"email": "test@example.com", "password": "TestPassword123"}

        try:
            response = await client.post(f"{API_URL}/auth/register", json=register_data)
            if response.status_code == 200:
                result = response.json()
                print("✅ Registration successful!")
                print(f"   User ID: {result['user']['id']}")
                print(f"   Email: {result['user']['email']}")
                access_token = result["access_token"]
            else:
                print(f"❌ Registration failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return

        # Test 2: Login with the same credentials
        print("\n2️⃣ Testing User Login")
        login_data = {"email": "test@example.com", "password": "TestPassword123"}

        try:
            response = await client.post(f"{API_URL}/auth/login", json=login_data)
            if response.status_code == 200:
                result = response.json()
                print("✅ Login successful!")
                print(f"   User ID: {result['user']['id']}")
                print(f"   Email: {result['user']['email']}")
                login_token = result["access_token"]
            else:
                print(f"❌ Login failed: {response.status_code}")
                print(f"   Error: {response.text}")
                return
        except Exception as e:
            print(f"❌ Login error: {e}")
            return

        # Test 3: Access protected endpoint
        print("\n3️⃣ Testing Protected Endpoint Access")
        headers = {"Authorization": f"Bearer {login_token}"}

        try:
            response = await client.get(f"{API_URL}/auth/me", headers=headers)
            if response.status_code == 200:
                result = response.json()
                print("✅ Protected endpoint access successful!")
                print(f"   User Info: {json.dumps(result, indent=2)}")
            else:
                print(f"❌ Protected endpoint access failed: {response.status_code}")
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"❌ Protected endpoint error: {e}")

        # Test 4: Try duplicate registration
        print("\n4️⃣ Testing Duplicate Registration Prevention")
        try:
            response = await client.post(f"{API_URL}/auth/register", json=register_data)
            if response.status_code == 409:
                print("✅ Duplicate registration correctly prevented!")
                print(f"   Error message: {response.json().get('detail')}")
            else:
                print(
                    f"❌ Duplicate registration not prevented: {response.status_code}"
                )
        except Exception as e:
            print(f"❌ Duplicate registration test error: {e}")

        # Test 5: Try login with wrong password
        print("\n5️⃣ Testing Invalid Login Prevention")
        wrong_login_data = {"email": "test@example.com", "password": "WrongPassword123"}

        try:
            response = await client.post(f"{API_URL}/auth/login", json=wrong_login_data)
            if response.status_code == 401:
                print("✅ Invalid login correctly prevented!")
                print(f"   Error message: {response.json().get('detail')}")
            else:
                print(f"❌ Invalid login not prevented: {response.status_code}")
        except Exception as e:
            print(f"❌ Invalid login test error: {e}")

    print("\n" + "=" * 50)
    print("🎉 Authentication system testing complete!")
    print("\n💡 Next steps:")
    print("   1. Start the API server: uvicorn main:app --reload")
    print("   2. Start the Streamlit app: streamlit run app/app.py")
    print("   3. Test the UI by registering and logging in")


if __name__ == "__main__":
    asyncio.run(test_auth_system())
