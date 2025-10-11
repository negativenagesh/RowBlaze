import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_env():
    """Set up test environment variables."""
    original_env = os.environ.copy()
    
    # Set test-specific environment variables
    test_env_vars = {
        "OPENAI_MODEL": "gpt-4o-mini-2024-07-18",
        "CHUNK_SIZE_TOKENS": "1024",
        "CHUNK_OVERLAP_TOKENS": "512",
        "TEST_API_URL": "http://localhost:8000/api",
    }
    
    for key, value in test_env_vars.items():
        os.environ[key] = value
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


{
  "status": "healthy",
  "timestamp": "2023-10-11T12:34:56"
}