import abc
from typing import Any, AsyncGenerator, Dict, Generic, List, Optional, Tuple, TypeVar

# Define a generic type parameter
T = TypeVar("T")


class BaseParser:
    """Base class for all document parsers."""

    def __init__(
        self,
        aclient_openai: Optional[Any] = None,
        server_type: str = None,
        processor_ref: Optional[Any] = None,
    ):
        self.aclient_openai = aclient_openai
        self.server_type = server_type
        self.processor_ref = processor_ref

    @abc.abstractmethod
    def ingest(self, data: Any) -> Any:
        """Process the input data and return extracted content."""
        pass


class AsyncParser(Generic[T]):
    """Base class for asynchronous document parsers with generic type parameter."""

    def __init__(
        self,
        aclient_openai: Optional[Any] = None,
        server_type: str = None,
        processor_ref: Optional[Any] = None,
    ):
        self.aclient_openai = aclient_openai
        self.server_type = server_type
        self.processor_ref = processor_ref

    @abc.abstractmethod
    async def ingest(self, data: T) -> AsyncGenerator[Tuple[str, int], None]:
        """Process the input data and yield extracted content with page numbers."""
        pass
