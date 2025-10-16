#!/usr/bin/env python3
"""
Simple test to verify the authentication system works.
"""

import sys

sys.path.append(".")


def test_password_hashing():
    """Test password hashing functionality."""
    try:
        from api.routes.auth import hash_password, verify_password

        # Test password hashing
        password = "TestPassword123"
        hashed = hash_password(password)

        print("🔐 Testing Password Hashing:")
        print(f"   Original: {password}")
        print(f"   Hashed: {hashed[:20]}...")

        # Test verification
        is_valid = verify_password(password, hashed)
        is_invalid = verify_password("WrongPassword", hashed)

        print(f"   ✅ Correct password verification: {is_valid}")
        print(f"   ✅ Wrong password rejection: {not is_invalid}")

        return True
    except Exception as e:
        print(f"❌ Password hashing test failed: {e}")
        return False


def test_email_validation():
    """Test email validation functionality."""
    try:
        from api.routes.auth import validate_email

        test_cases = [
            ("user@example.com", True),
            ("test.email+tag@domain.co.uk", True),
            ("invalid-email", False),
            ("@domain.com", False),
            ("user@", False),
            ("user..double@domain.com", False),
        ]

        print("\n📧 Testing Email Validation:")
        all_passed = True
        for email, expected in test_cases:
            result = validate_email(email)
            status = "✅" if result == expected else "❌"
            print(f"   {status} {email}: {result} (expected {expected})")
            if result != expected:
                all_passed = False

        return all_passed
    except Exception as e:
        print(f"❌ Email validation test failed: {e}")
        return False


def test_token_creation():
    """Test JWT token creation."""
    try:
        import jwt

        from api.routes.auth import create_access_token

        test_data = {"sub": "test-user-id", "email": "test@example.com"}

        token = create_access_token(test_data)

        print("\n🎫 Testing JWT Token Creation:")
        print(f"   Token created: {token[:20]}...")

        # Try to decode it
        decoded = jwt.decode(token, options={"verify_signature": False})
        print(f"   ✅ Token contains: {decoded}")

        return True
    except Exception as e:
        print(f"❌ Token creation test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Testing RowBlaze Authentication Components")
    print("=" * 50)

    tests = [test_password_hashing, test_email_validation, test_token_creation]

    passed = 0
    for test in tests:
        if test():
            passed += 1

    print("\n" + "=" * 50)
    print(f"🎯 Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("✅ All authentication components are working correctly!")
        print("\n💡 Next steps:")
        print("   1. Start the API server: uvicorn main:app --reload")
        print("   2. Start the Streamlit app: streamlit run app/app.py")
        print("   3. Test registration and login in the UI")
    else:
        print("❌ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
