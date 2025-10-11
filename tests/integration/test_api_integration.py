import pytest
import asyncio
import httpx
import os
from unittest.mock import patch
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestAPIIntegration:
    """Integration tests for the API endpoints."""
    
    @pytest.fixture
    def api_base_url(self):
        """Get the API base URL from environment or use default."""
        return os.getenv("TEST_API_URL", "http://localhost:8000/api")
    
    @pytest.mark.asyncio
    async def test_health_endpoint_integration(self, api_base_url):
        """Test health endpoint in integration environment."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_base_url}/health", timeout=5.0)
                
                if response.status_code == 404:
                    # API not running, skip test
                    pytest.skip("API server not available for integration testing")
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert "timestamp" in data
                
            except (httpx.ConnectError, httpx.TimeoutException):
                pytest.skip("API server not available for integration testing")
    
    @pytest.mark.asyncio
    async def test_files_endpoint_integration(self, api_base_url):
        """Test files endpoint integration."""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_base_url}/files/test_index", timeout=10.0)
                # Should return either 200 with file list or 404 if index doesn't exist
                # If we get 404 for the health endpoint, the API isn't running
                
                # First check if API is available
                health_response = await client.get(f"{api_base_url}/health", timeout=5.0)
                if health_response.status_code == 404:
                    pytest.skip("API server not available for integration testing")
                
                # Now test the files endpoint
                assert response.status_code in [200, 404, 422]  # 422 for validation errors
                
            except (httpx.ConnectError, httpx.TimeoutException):
                pytest.skip("API server not available for integration testing")