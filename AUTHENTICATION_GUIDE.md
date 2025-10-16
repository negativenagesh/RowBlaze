# RowBlaze Authentication Guide

## Quick Start

### Default Credentials
- **Username:** `rowblaze`
- **Password:** `password123`

## Running the Application

### Option 1: With API Server (Full Features)
1. Start the API server:
   ```bash
   uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

3. Login with the default credentials above.

### Option 2: Local Mode (Limited Features)
If the API server is not running, the app will automatically fall back to local authentication mode.

1. Start only the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

2. Login with the default credentials above.

## Authentication Status Indicators

- 🟢 **API Server Connected**: Full functionality available
- 🔵 **API Server Offline - Local Authentication Available**: Limited functionality, local authentication only

## Troubleshooting

### Login Issues
1. Make sure you're using the correct credentials: `rowblaze` / `password123`
2. Check if the API server is running (see status indicator on login page)
3. If API server is offline, local authentication will still work for basic access

### API Server Issues
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Check that port 8000 is not in use by another application
3. Verify environment variables are set correctly (see `.env` file)

## Environment Variables

Create a `.env` file with:
```
ROWBLAZE_API_URL=http://localhost:8000/api
JWT_SECRET_KEY=your-secret-key-here
OPENAI_API_KEY=your-openai-key-here
```

## Security Notes

- The default credentials are for development only
- In production, change the JWT secret and implement proper user management
- The local authentication fallback is for development convenience only
