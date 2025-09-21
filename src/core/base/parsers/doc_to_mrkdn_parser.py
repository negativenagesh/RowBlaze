import logging
from io import BytesIO
from typing import AsyncGenerator, List, Union
import tempfile
import os

import pypandoc
from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from .base_parser import AsyncParser

logger = logging.getLogger(__name__)

class DOCParser(AsyncParser[bytes]):
    """
    A parser for DOC (legacy Microsoft Word) data.
    Converts DOC to DOCX in memory, then parses the content to extract
    paragraphs and tables, yielding them as text and Markdown strings.
    """

    def __init__(self):
        # No specific initialization needed for this approach
        pass

    async def ingest(
        self, data: bytes, **kwargs
    ) -> AsyncGenerator[Union[str, List[List[str]]], None]:
        """
        Ingest DOC data, convert it, and yield a stream of content
        (text paragraphs and markdown tables).
        """
        if not isinstance(data, bytes):
            raise TypeError("DOC data must be in bytes format.")

        tmp_docx_path = None
        try:
            # Create a temporary file path for pandoc to write to
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp:
                tmp_docx_path = temp.name

            # 1. Convert .doc bytes to a temporary .docx file using pandoc
            pypandoc.convert_text(
                data, "docx", format="docx", outputfile=tmp_docx_path
            )

            # Read the converted file back into memory
            with open(tmp_docx_path, "rb") as f:
                docx_bytes = f.read()

            docx_stream = BytesIO(docx_bytes)

            # 2. Open the .docx stream with python-docx
            document: DocumentObject = Document(docx_stream)

            # 3. Iterate through document body elements (paragraphs and tables)
            for element in document._body.iter_inner_content():
                if isinstance(element, Paragraph):
                    # Yield non-empty paragraph text
                    if element.text.strip():
                        yield element.text.strip()
                elif isinstance(element, Table):
                    # Convert table to Markdown and yield it
                    table_markdown = self._convert_table_to_markdown(element)
                    if table_markdown:
                        yield table_markdown

        except Exception as e:
            logger.error(f"Error processing DOC file: {str(e)}", exc_info=True)
            if "pandoc" in str(e).lower() and "not found" in str(e).lower():
                 raise RuntimeError(
                    "Pandoc not found. Please install pandoc and ensure it is in your system's PATH."
                ) from e
            raise ValueError(f"Error processing DOC file: {str(e)}") from e
        finally:
            # Ensure the temporary file is always cleaned up
            if tmp_docx_path and os.path.exists(tmp_docx_path):
                os.remove(tmp_docx_path)

    def _convert_table_to_markdown(self, table: Table) -> str:
        """Convert a python-docx Table object to a Markdown string."""
        if not table.rows:
            return ""

        # Extract header from the first row
        header_cells = [cell.text.strip() for cell in table.rows[0].cells]
        
        # Skip empty tables
        if not any(header_cells):
            return ""

        markdown = "| " + " | ".join(header_cells) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(header_cells)) + " |\n"

        # Process remaining rows
        for row in table.rows[1:]:
            row_cells = [cell.text.strip() for cell in row.cells]
            markdown += "| " + " | ".join(row_cells) + " |\n"
            
        return markdown