import logging
import asyncio
from io import BytesIO
from typing import AsyncGenerator

from .base_parser import AsyncParser

logger = logging.getLogger(__name__)

try:
    from pdf2image import convert_from_bytes

    PYPDF2IMAGE_INSTALLED = True
except ImportError:
    PYPDF2IMAGE_INSTALLED = False

try:
    import pytesseract

    PYTESSERACT_INSTALLED = True
except ImportError:
    PYTESSERACT_INSTALLED = False


class OCRParser(AsyncParser[bytes]):
    """A parser for OCR-based text extraction from PDF files."""

    def __init__(self):
        if not PYPDF2IMAGE_INSTALLED or not PYTESSERACT_INSTALLED:
            msg = "OCR parsing requires 'pdf2image' and 'pytesseract'. Please install them (`pip install pdf2image pytesseract`) and ensure Tesseract-OCR is in your system's PATH."
            logger.error(msg)
            raise ImportError(msg)

    async def ingest(self, data: bytes, **kwargs) -> AsyncGenerator[str, None]:
        """Ingest PDF data, perform OCR, and yield text from each page."""
        if not isinstance(data, bytes):
            raise TypeError("PDF data must be in bytes format.")

        pdf_stream = BytesIO(data)
        logger.info("Starting OCR text extraction. This may take a while...")
        try:
            images = convert_from_bytes(pdf_stream.read())
            logger.info(f"Converted {len(images)} pages to images for OCR.")
            for i, image in enumerate(images):
                page_num = i + 1
                try:
                    # Perform OCR on the image in a separate thread to avoid blocking asyncio event loop
                    page_text = await asyncio.to_thread(
                        pytesseract.image_to_string, image
                    )
                    if not page_text or not page_text.strip():
                        logger.warning(f"OCR found no text on page {page_num}.")
                    yield page_text
                except pytesseract.TesseractNotFoundError:
                    logger.error(
                        "Tesseract executable not found. Please install Tesseract-OCR and ensure it's in your system's PATH."
                    )
                    raise
                except Exception as ocr_err:
                    logger.error(
                        f"Error during OCR on page {page_num}: {ocr_err}", exc_info=True
                    )
                    yield ""
        except Exception as e:
            logger.error(f"Failed to convert PDF to images for OCR: {e}", exc_info=True)
            raise ValueError(f"Error processing PDF for OCR: {str(e)}") from e
        finally:
            pdf_stream.close()
