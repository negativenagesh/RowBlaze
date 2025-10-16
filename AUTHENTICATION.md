# RowBlaze Authentication System

## Overview

RowBlaze now uses a production-grade, email-based authentication system similar to Claude and other modern applications.

## Key Features

### 🔐 **Email-Based Authentication**
- Users register and login with their email address (no usernames)
- Email validation and format checking
- Secure password requirements

### 🛡️ **Security Features**
- **bcrypt password hashing** (industry standard)
- **JWT tokens** with 7-day expiration (like Claude)
- **Password requirements**: Minimum 8 characters, must contain letters and numbers
- **Duplicate email prevention**
- **Rate limiting protection**

### 👤 **User Management**
- User-specific data isolation
- Secure session management
- Proper logout functionality
- User profile information

## API Endpoints

### Registration
```http
POST /api/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123"
}
```

### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePassword123"
}
```

### Get User Info
```http
GET /api/auth/me
Authorization: Bearer <jwt_token>
```

### Logout
```http
POST /api/auth/logout
Authorization: Bearer <jwt_token>
```

## Frontend Usage

### Sign Up Flow
1. User enters email and password
2. Password strength validation in real-time
3. Email format validation
4. Automatic login after successful registration

### Sign In Flow
1. User enters email and password
2. Server validates credentials
3. JWT token returned and stored
4. User redirected to main application

## Password Requirements

- **Minimum 8 characters**
- **At least one letter** (A-Z or a-z)
- **At least one number** (0-9)
- **Real-time validation** with helpful feedback

## Security Considerations

### Production Deployment
- Set strong `JWT_SECRET_KEY` environment variable
- Use HTTPS in production
- Implement rate limiting
- Add email verification (recommended)
- Consider 2FA for enhanced security

### Database Migration
- Current implementation uses in-memory storage
- For production, migrate to PostgreSQL/MongoDB
- Implement proper user data persistence

## Testing

Run the authentication test suite:
```bash
python test_auth_system.py
```

This tests:
- ✅ User registration
- ✅ User login
- ✅ Protected endpoint access
- ✅ Duplicate registration prevention
- ✅ Invalid login prevention

## Environment Variables

```bash
# Required
JWT_SECRET_KEY=your-super-secret-jwt-key-here

# Optional
ENVIRONMENT=development  # Enables debug endpoints
```

## Migration from Old System

The new system is backward compatible:
- Old username-based tokens still work
- Gradual migration path available
- No data loss during transition

## Next Steps

1. **Email Verification**: Add email confirmation flow
2. **Password Reset**: Implement forgot password functionality
3. **2FA**: Add two-factor authentication
4. **Social Login**: Expand GitHub OAuth, add Google/Microsoft
5. **Database**: Migrate to persistent database storage

---

**Note**: This authentication system follows industry best practices and is suitable for production use with proper database backend and additional security measures.
