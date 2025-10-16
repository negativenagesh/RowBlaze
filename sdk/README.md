# RowBlaze SDK

The RowBlaze SDK provides a simple interface for interacting with the RowBlaze API.

## Installation

```bash
pip install rowblaze-sdk
```

## Usage

```python
from rowblaze.sdk import RowBlazeClient

# Initialize the client
client = RowBlazeClient(api_url="http://localhost:8000/api", api_key="your_api_key")

# Upload a document
response = client.upload_document_sync(
    file_path="path/to/document.pdf",
    index_name="my_index",
    description="Important document about topic XYZ"
)
print(f"Document uploaded: {response}")

# Query the knowledge base
result = client.query_sync(
    question="What are the key points in the document?",
    index_name="my_index"
)
print(f"Answer: {result.answer}")
print(f"Sources: {result.sources}")

# Check API health
health = client.health_sync()
print(f"API Status: {health['status']}")
```

## Async Usage

The SDK also supports asynchronous operation:

```python
import asyncio
from rowblaze.sdk import RowBlazeClient

async def main():
    client = RowBlazeClient(api_url="http://localhost:8000/api")

    # Upload a document
    response = await client.upload_document(
        file_path="path/to/document.pdf",
        index_name="my_index"
    )

    # Query the document
    result = await client.query(
        question="What are the key points?",
        index_name="my_index"
    )
    print(result.answer)

asyncio.run(main())
```

## Configuration Options

The client accepts the following initialization parameters:

- `api_url`: Base URL of the RowBlaze API
- `api_key`: API key for authentication (optional)

## Error Handling

The SDK raises standard HTTP exceptions for API errors. It's recommended to handle these appropriately:

```python
import httpx

try:
    result = client.query_sync("What's in this document?")
    print(result.answer)
except httpx.HTTPStatusError as e:
    print(f"API Error: {e.response.status_code} - {e.response.text}")
except httpx.RequestError as e:
    print(f"Connection Error: {str(e)}")
```

## Development

To contribute to the SDK, clone the repository and install development dependencies:

```bash
git clone https://github.com/your-org/rowblaze.git
cd rowblaze
pip install -e ".[dev]"
```
