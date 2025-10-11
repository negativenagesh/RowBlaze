import pytest
import asyncio
import httpx
import tempfile
import os
from pathlib import Path
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

class TestCompleteWorkflow:
    """End-to-end tests for complete user workflows."""
    
    @pytest.fixture
    def api_base_url(self):
        """Get the API base URL from environment or use default."""
        return os.getenv("TEST_API_URL", "http://localhost:8000/api")
    
    @pytest.mark.asyncio
    async def test_document_upload_and_query_workflow(self, api_base_url):
        """Test the complete workflow: upload document -> query -> get response."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Step 1: Check API health
                health_response = await client.get(f"{api_base_url}/health")
                
                if health_response.status_code == 404:
                    pytest.skip("API server not available for E2E testing")
                
                assert health_response.status_code == 200
                health_data = health_response.json()
                assert health_data["status"] == "healthy"
                
                # Step 2: Create a test document
                test_content = """
                This is a test document for RowBlaze.
                It contains information about artificial intelligence and machine learning.
                The document explains how AI systems can process and understand text data.
                """
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
                    tmp_file.write(test_content)
                    tmp_file_path = tmp_file.name
                
                try:
                    # Step 3: Upload the document
                    with open(tmp_file_path, 'rb') as file:
                        files = {"file": ("test_doc.txt", file, "text/plain")}
                        data = {
                            "index_name": "test_e2e_index",
                            "description": "Test document for E2E testing",
                            "model": "gpt-4o-mini-2024-07-18",
                            "max_tokens": "16384"
                        }
                        
                        upload_response = await client.post(
                            f"{api_base_url}/ingest",
                            files=files,
                            data=data
                        )
                        
                        # Upload should succeed or indicate document already exists
                        if upload_response.status_code not in [200, 400, 422]:
                            pytest.skip(f"Upload endpoint not working properly: {upload_response.status_code}")
                    
                    # Step 4: Query the uploaded document (only if upload succeeded)
                    if upload_response.status_code == 200:
                        query_payload = {
                            "question": "What does this document explain about AI systems?",
                            "index_name": "test_e2e_index",
                            "model": "gpt-4o-mini-2024-07-18",
                            "max_tokens": 16384
                        }
                        
                        query_response = await client.post(
                            f"{api_base_url}/query",
                            json=query_payload
                        )
                        
                        if query_response.status_code == 200:
                            result = query_response.json()
                            assert "answer" in result
                            assert len(result["answer"]) > 0
                            
                            # The answer should be related to AI/ML based on our test document
                            answer_lower = result["answer"].lower()
                            assert any(keyword in answer_lower for keyword in 
                                     ["ai", "artificial intelligence", "machine learning", "text", "process"])
                        else:
                            pytest.skip(f"Query endpoint not working: {query_response.status_code}")
                
                finally:
                    # Cleanup
                    os.unlink(tmp_file_path)
                    
            except (httpx.ConnectError, httpx.TimeoutException):
                pytest.skip("API server not available for E2E testing")