import asyncio
import base64
import copy
import hashlib
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Tuple

import fitz
import httpx
import requests
import yaml

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import pdfplumber
import tiktoken
from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import BulkIndexError, async_bulk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

from sdk.message import Message
from sdk.response import FunctionResponse, Messages
from src.core.base.parsers.csv_parser import CSVParser
from src.core.base.parsers.doc_parser import DOCParser
from src.core.base.parsers.docx_parser import DOCXParser
from src.core.base.parsers.image_parser import ImageParser
from src.core.base.parsers.odt_parser import ODTParser
from src.core.base.parsers.text_parser import TextParser
from src.core.base.parsers.xlsx_parser import XLSXParser

# from utils.billing import calculatePriceByApi


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

try:
    from mistralai import Mistral

    MISTRAL_SDK_AVAILABLE = True
except ImportError:
    MISTRAL_SDK_AVAILABLE = False

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = 3072

CHUNK_SIZE_TOKENS = 20000
CHUNK_OVERLAP_TOKENS = 0
FORWARD_CHUNKS = 3
BACKWARD_CHUNKS = 3
CHARS_PER_TOKEN_ESTIMATE = 4
SUMMARY_MAX_TOKENS = 1024
ELASTICSEARCH_URL = os.getenv("RAG_UPLOAD_ELASTIC_URL")
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")


def init_async_openai_client() -> Optional[AsyncOpenAI]:
    openai_api_key = OPENAI_API_KEY
    print("Open ai key:-", openai_api_key)
    if not openai_api_key:
        print(
            "❌ OPENAI_API_KEY not found in .env. OpenAI client will not be functional."
        )
        return None

    try:
        client = AsyncOpenAI(api_key=openai_api_key)
        print("✅ AsyncOpenAI client initialized.")
        return client
    except Exception as e:
        print(f"❌ Failed to initialize AsyncOpenAI client: {e}")
        return None


async def check_async_elasticsearch_connection() -> Optional[AsyncElasticsearch]:
    try:
        print("Connecting to Elsatic client ELASTICSEARCH_URL:-", ELASTICSEARCH_URL)
        es_client = None
        os.getenv("ELASTICSEARCH_API_KEY")
        es_client = AsyncElasticsearch(
            ELASTICSEARCH_URL,
            api_key=ELASTICSEARCH_API_KEY,
            request_timeout=60,
            retry_on_timeout=True,
        )

        if not await es_client.ping():
            print(
                "❌ Ping to Elasticsearch cluster failed. URL may be incorrect or server is down."
            )
            return None

        print("✅ AsyncElasticsearch client initialized.")
        return es_client
    except Exception as e:
        print(f"❌ Failed to initialize AsyncElasticsearch client: {e}")
        return None


def tiktoken_len(text: str) -> int:
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text, disallowed_special=())
        return len(tokens)
    except Exception as e:
        print(f"❌ Failed to calc tiktoken_len: {e}")
        return 0


CHUNKED_PDF_MAPPINGS = {
    "mappings": {
        "properties": {
            "chunk_text": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 3072,
                "index": True,
                "similarity": "cosine",
            },
            "metadata": {
                "properties": {
                    "file_name": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "page_number": {"type": "integer"},
                    "chunk_index_in_page": {"type": "integer"},
                    "document_summary": {"type": "text"},
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "keyword"},
                            "type": {"type": "keyword"},
                            "description": {"type": "text"},
                            "description_embedding": {
                                "type": "dense_vector",
                                "dims": 3072,
                                "index": True,
                                "similarity": "cosine",
                            },
                        },
                    },
                    "relationships": {
                        "type": "nested",
                        "properties": {
                            "source_entity": {"type": "keyword"},
                            "target_entity": {"type": "keyword"},
                            "relation": {"type": "keyword"},
                            "relationship_description": {"type": "text"},
                            "relationship_weight": {"type": "float"},
                        },
                    },
                    "hierarchies": {
                        "type": "nested",
                        "properties": {
                            "name": {"type": "keyword"},
                            "description": {"type": "text"},
                            "root_type": {"type": "keyword"},
                            "levels": {
                                "type": "nested",
                                "properties": {
                                    "id": {"type": "keyword"},
                                    "name": {"type": "keyword"},
                                    "nodes": {"type": "nested"},
                                },
                            },
                            "relationships": {"type": "nested"},
                        },
                    },
                }
            },
        }
    }
}


class PDFParser:
    """A parser for extracting tables and text from PDF files using pdfplumber."""

    def __init__(
        self,
        aclient_openai: Optional[AsyncOpenAI],
        server_type: str,
        processor_ref: Optional[Any] = None,
    ):

        self.aclient_openai = aclient_openai
        self.Server_type = os.getenv("SERVER_TYPE")
        self.processor_ref = processor_ref
        self.vision_prompt_text = self._load_vision_prompt()

    def _load_vision_prompt(self) -> str:
        try:
            prompt_file_path = (
                Path(__file__).parent.parent / "prompts" / "vision_img.yaml"
            )
            with open(prompt_file_path, "r") as f:
                prompt_data = yaml.safe_load(f)

            if (
                prompt_data
                and "vision_img" in prompt_data
                and "template" in prompt_data["vision_img"]
            ):
                template_content = prompt_data["vision_img"]["template"]
                print("Successfully loaded vision prompt template.")
                return template_content
            else:
                print(
                    f"Vision prompt template not found or invalid in {prompt_file_path}."
                )
                return "Describe the image in detail."
        except Exception as e:
            print(f"Error loading vision prompt: {e}")
            return "Describe the image in detail."

    async def _get_image_description(self, image_bytes: bytes) -> str:

        image_data = base64.b64encode(image_bytes).decode("utf-8")
        media_type = "image/png"  # PyMuPDF pixmaps default to png

        if self.Server_type == "ARMY":
            if not self.processor_ref:
                print(
                    "Processor reference not available for NVIDIA VLM call. Skipping image description."
                )
                return ""

            print("Using NVIDIA VLM for image description.")
            # Prepare the payload for our unified API caller
            messages = [
                {
                    "role": "user",
                    "content": self.vision_prompt_text,
                    "image": image_data,
                }
            ]

            description = await self.processor_ref._call_nvidia_api(
                payload_messages=messages,
                is_vision_call=True,
                max_tokens=1024,
                temperature=1.0,
            )
            return (
                f"\n[Image Description]: {description.strip()}\n" if description else ""
            )

        if not self.aclient_openai:
            print("OpenAI client not available, skipping image description.")
            return ""

        if not self.processor_ref:
            print(
                "Processor reference not available for OpenAI call. Skipping image description."
            )
            return ""

        if not self.aclient_openai:
            print("OpenAI client not available, skipping image description.")
            return ""

        print("Using OpenAI VLM for image description.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.vision_prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{image_data}"},
                    },
                ],
            }
        ]

        description = await self.processor_ref._call_openai_api(
            model_name=OPENAI_CHAT_MODEL,
            payload_messages=messages,
            is_vision_call=True,
            max_tokens=1024,
            temperature=1.0,
        )

        print(f"*** image description: {description}")
        return f"\n[Image Description]: {description.strip()}\n" if description else ""

    async def ingest(self, data: bytes) -> AsyncGenerator[Tuple[str, int], None]:
        """
        Ingest PDF data and yield a stream of content parts (text, tables, image descriptions)
        along with their corresponding page number.
        """
        if not isinstance(data, bytes):
            raise TypeError("PDF data must be in bytes format.")

        pdf_stream = BytesIO(data)
        try:
            with pdfplumber.open(pdf_stream) as pdf_plumber:
                with fitz.open(stream=data, filetype="pdf") as pdf_fitz:
                    if len(pdf_plumber.pages) != len(pdf_fitz):
                        print(
                            "Warning: Page count mismatch between pdfplumber and PyMuPDF."
                        )

                    for page_num, p_page in enumerate(pdf_plumber.pages, 1):
                        # 1. Extract text from the page. Each block is a potential paragraph.
                        page_text = p_page.extract_text()
                        if page_text:
                            yield (page_text.strip(), page_num)

                        # 2. Extract tables from the same page as separate blocks.
                        tables = p_page.extract_tables()
                        for table in tables:
                            table_markdown = self._convert_table_to_markdown(table)
                            yield (table_markdown, page_num)

                        # 3. Extract images using PyMuPDF and get descriptions as separate blocks.
                        if page_num <= len(pdf_fitz):
                            fitz_page = pdf_fitz.load_page(page_num - 1)
                            image_list = fitz_page.get_images(full=True)
                            if image_list:
                                print(
                                    f"Found {len(image_list)} images on page {page_num}."
                                )
                                for img_info in image_list:
                                    xref = img_info[0]
                                    base_image = pdf_fitz.extract_image(xref)
                                    image_bytes = base_image["image"]

                                    img_w = int(base_image.get("width", 0) or 0)
                                    img_h = int(base_image.get("height", 0) or 0)
                                    if img_w < 100 or img_h < 100:
                                        print(
                                            f"Skipping image extraction for page {page_num} due to small image size."
                                        )
                                        continue

                                    description = await self._get_image_description(
                                        image_bytes
                                    )
                                    if description:
                                        yield (description, page_num)
                        else:
                            print(
                                f"Skipping image extraction for page {page_num} due to page count mismatch."
                            )

        except Exception as e:
            print(f"Failed to process PDF: {e}")
            raise ValueError(f"Error processing PDF: {str(e)}") from e

    def _extract_tables_with_fitz(self, page):
        """Extract tables from a page using PyMuPDF."""
        tables = []
        try:
            # Method 1: Use built-in table detection
            tab = page.find_tables()
            if tab and tab.tables:
                for table_data in tab.tables:
                    headers = []
                    rows = []

                    # Convert fitz table to list of lists format
                    for i in range(table_data.rows):
                        row = []
                        for j in range(table_data.cols):
                            cell_text = (
                                table_data.cells[i][j].text
                                if table_data.cells[i][j]
                                else ""
                            )
                            row.append(cell_text.strip())

                        if i == 0:  # Assume first row is header
                            headers = row
                        else:
                            rows.append(row)

                    # Add the table with headers and rows
                    tables.append([headers] + rows)

            # Method 2: If no tables found with method 1, try structured text extraction
            if not tables:
                blocks = page.get_text("blocks")
                for block in blocks:
                    if block[6] == 1:  # Block type 1 indicates a table
                        # Extract text within this block and process as tabular
                        rect = fitz.Rect(block[:4])
                        table_text = page.get_textbox(rect)
                        if table_text:
                            # Simple heuristic to detect tabular structure
                            lines = table_text.split("\n")
                            if lines and len(lines) >= 2:
                                processed_table = []
                                for line in lines:
                                    cells = [cell.strip() for cell in line.split("|")]
                                    if len(cells) <= 1:
                                        cells = [
                                            cell.strip() for cell in line.split("\t")
                                        ]
                                    if (
                                        len(cells) > 1
                                    ):  # Only add rows that actually have multiple cells
                                        processed_table.append(cells)

                                if processed_table and len(processed_table) >= 2:
                                    tables.append(processed_table)

        except Exception as e:
            print(f"Error extracting tables with PyMuPDF: {e}")

        return tables

    def _convert_table_to_markdown(self, table: list) -> str:
        """Convert a table (list of rows) to Markdown format."""
        if not table or not table[0]:
            return ""

        header = [str(cell) if cell is not None else "" for cell in table[0]]
        markdown = "| " + " | ".join(header) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
        for row in table[1:]:
            if row:
                processed_row = [str(cell) if cell is not None else "" for cell in row]
                if len(processed_row) == len(header):
                    markdown += "| " + " | ".join(processed_row) + " |\n"
        return markdown


class OCRParser:
    """A parser for OCR-based text extraction from PDF files."""

    def __init__(self):
        if not PYPDF2IMAGE_INSTALLED or not PYTESSERACT_INSTALLED:
            msg = "OCR parsing requires 'pdf2image' and 'pytesseract'. Please install them and ensure Tesseract-OCR is in your system's PATH."
            print(f"ERROR: {msg}")
            raise ImportError(msg)

    async def ingest(self, data: bytes) -> AsyncGenerator[str, None]:
        """Ingest PDF data, perform OCR, and yield text from each page."""
        if not isinstance(data, bytes):
            raise TypeError("PDF data must be in bytes format.")

        pdf_stream = BytesIO(data)
        print("Starting OCR text extraction. This may take a while...")
        try:
            images = convert_from_bytes(pdf_stream.read())
            print(f"Converted {len(images)} pages to images for OCR.")
            for i, image in enumerate(images):
                page_num = i + 1
                try:
                    page_text = await asyncio.to_thread(
                        pytesseract.image_to_string, image
                    )
                    if not page_text or not page_text.strip():
                        print(f"OCR found no text on page {page_num}.")
                    yield page_text
                except pytesseract.TesseractNotFoundError:
                    print(
                        "Tesseract executable not found. Please install Tesseract-OCR and ensure it's in your system's PATH."
                    )
                    raise
                except Exception as ocr_err:
                    print(f"Error during OCR on page {page_num}: {ocr_err}")
                    yield ""
        except Exception as e:
            print(f"Failed to convert PDF to images for OCR: {e}")
            raise ValueError(f"Error processing PDF for OCR: {str(e)}") from e
        finally:
            pdf_stream.close()


class MistralOCRParser:
    """OCR via Mistral OCR API (mistral-ocr-latest)."""

    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("MISTRAL_API_KEY is not set.")
        if not MISTRAL_SDK_AVAILABLE:
            raise ImportError("mistralai SDK is not installed.")
        self.client = Mistral(api_key=api_key)

    async def ingest(self, data: bytes) -> AsyncGenerator[str, None]:
        if not isinstance(data, bytes):
            raise TypeError("PDF data must be in bytes format.")

        # Run blocking SDK calls in a thread
        def _run_ocr(tmp_path: str) -> List[str]:
            with open(tmp_path, "rb") as f:
                uploaded_pdf = self.client.files.upload(
                    file={"file_name": os.path.basename(tmp_path), "content": f},
                    purpose="ocr",
                )
            signed_url = self.client.files.get_signed_url(file_id=uploaded_pdf.id)
            ocr_response = self.client.ocr.process(
                model="mistral-ocr-latest",
                document={"type": "document_url", "document_url": signed_url.url},
                include_image_base64=False,
            )
            pages = []
            for p in getattr(ocr_response, "pages", []) or []:
                text = getattr(p, "markdown", None) or getattr(p, "text", "") or ""
                pages.append(text)
            return pages

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                pages_text = await asyncio.to_thread(_run_ocr, tmp.name)
            except Exception as e:
                print(f"Mistral OCR failed: {e}")
                raise

        # Now yield each page with its page number (starting from 1)
        for page_num, page_text in enumerate(pages_text, 1):
            yield (page_text, page_num)


class ChunkingEmbeddingPDFProcessor:
    def __init__(
        self,
        params: Any,
        config: Any,
        aclient_openai: Optional[AsyncOpenAI],
        file_extension: str,
    ):
        self.params = params
        self.config = config
        self.aclient_openai = aclient_openai
        self.Server_type = os.getenv("SERVER_TYPE")
        self.embedding_model = None
        self.embedding_dims = OPENAI_EMBEDDING_DIMENSIONS

        if file_extension in [".docx", ".doc", ".odt"]:
            chunk_size = 2048
            chunk_overlap = 1024
            print(
                f"Using DOCX specific chunking: size={chunk_size}, overlap={chunk_overlap}"
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=tiktoken_len,
                separators=["\n|", "\n", "|", ". ", " ", ""],
            )
        else:
            chunk_size = CHUNK_SIZE_TOKENS
            chunk_overlap = CHUNK_OVERLAP_TOKENS
            print(f"Using default chunking: size={chunk_size}, overlap={chunk_overlap}")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE_TOKENS,
                chunk_overlap=CHUNK_OVERLAP_TOKENS,
                length_function=tiktoken_len,
                separators=["\n|", "\n", "|", ". ", " ", ""],
            )
        self.enrich_prompt_template = self._load_prompt_template("chunk_enrichment")
        self.graph_extraction_prompt_template = self._load_prompt_template(
            "graph_extraction"
        )
        self.summary_prompt_template = self._load_prompt_template("summary")
        self.graph_hierarchy_prompt_template = self._load_prompt_template(
            "graph_hierarchy"
        )

    def _load_prompt_template(self, prompt_name: str) -> str:
        try:
            prompt_file_path = PROMPTS_DIR / f"{prompt_name}.yaml"
            print("prompt_file_path:----", prompt_file_path)
            with open(prompt_file_path, "r") as f:
                prompt_data = yaml.safe_load(f)

            if (
                prompt_data
                and prompt_name in prompt_data
                and "template" in prompt_data[prompt_name]
            ):
                template_content = prompt_data[prompt_name]["template"]
                print(f"Successfully loaded prompt template for '{prompt_name}'.")
                return template_content
            else:
                print(
                    f"Prompt template for '{prompt_name}' not found or invalid in {prompt_file_path}."
                )
                raise ValueError(f"Invalid prompt structure for {prompt_name}")
        except FileNotFoundError:
            print(f"Prompt file not found: {prompt_file_path}")
            raise
        except Exception as e:
            print(f"Error loading prompt '{prompt_name}': {e}")
            raise

    async def _extract_hierarchies(
        self, chunk_text: str, document_summary: str
    ) -> List[Dict[str, Any]]:
        """Extract hierarchical structures from text using LLM."""
        if self.Server_type == "ARMY":
            print("Extracting hierarchies using NVIDIA API.")
            formatted_prompt = self.graph_hierarchy_prompt_template.format(
                collection_description=document_summary, input_text=chunk_text
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert assistant that identifies hierarchical structures in text and formats them as XML according to the provided schema.",
                },
                {"role": "user", "content": formatted_prompt},
            ]
            xml_response = await self._call_nvidia_api(
                payload_messages=messages, max_tokens=4000, temperature=0.1
            )
            return self._parse_hierarchy_xml(xml_response) if xml_response else []

        if not self.aclient_openai or not self.graph_hierarchy_prompt_template:
            print(
                "OpenAI client or hierarchy extraction prompt not available. Skipping hierarchy extraction."
            )
            return []

        formatted_prompt = self.graph_hierarchy_prompt_template.format(
            collection_description=document_summary, input_text=chunk_text
        )
        print(
            f"Formatted prompt for hierarchy extraction (chunk-level, to {OPENAI_SUMMARY_MODEL}): First 200 chars: {formatted_prompt[:200]}..."
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert assistant that identifies hierarchical structures in text and formats them as XML according to the provided schema.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        xml_response_content = await self._call_openai_api(
            model_name=OPENAI_SUMMARY_MODEL,
            payload_messages=messages,
            max_tokens=4000,
            temperature=0.1,
        )

        if not xml_response_content:
            print("LLM returned empty content for hierarchy extraction.")
            return []

        print(
            f"Raw XML response from LLM for hierarchy extraction (first 500 chars):\n{xml_response_content[:500]}"
        )
        return self._parse_hierarchy_xml(xml_response_content)

    def _clean_ocr_repetitions(self, text: str) -> str:
        """Remove repetitive patterns from OCR text."""
        if not text:
            return text

        # Clean up common OCR repetition patterns
        # Look for repeated phrases of 5-15 words
        words = text.split()
        cleaned_words = []
        i = 0

        while i < len(words):
            cleaned_words.append(words[i])

            # Check for repetitions of phrases (5-15 words long)
            for phrase_len in range(5, min(16, len(words) - i)):
                phrase = words[i : i + phrase_len]
                phrase_str = " ".join(phrase)

                # Look ahead to see if this phrase repeats immediately
                next_pos = i + phrase_len
                repeat_count = 0

                while next_pos + phrase_len <= len(words):
                    next_phrase = words[next_pos : next_pos + phrase_len]
                    next_phrase_str = " ".join(next_phrase)

                    if next_phrase_str == phrase_str:
                        repeat_count += 1
                        next_pos += phrase_len
                    else:
                        break

                # If we found repetitions, skip them in the output
                if repeat_count > 0:
                    print(f"Found {repeat_count} repetitions of phrase: '{phrase_str}'")
                    i = (
                        next_pos - 1
                    )  # -1 because we'll increment i at the end of the loop
                    break

            i += 1

        return " ".join(cleaned_words)

    async def _call_openai_api(
        self,
        model_name: str,
        payload_messages: List[Dict[str, Any]],
        is_vision_call: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """A unified async method to call OpenAI text and vision models with retry logic."""
        if not self.aclient_openai:
            print("OpenAI client not configured. Cannot make API call.")
            return ""

        model_to_use = self.params.get("model", model_name)

        max_retries = 10
        base_delay_seconds = 3

        for attempt in range(max_retries):
            try:
                start_time = datetime.now(timezone.utc)

                response = await self.aclient_openai.chat.completions.create(
                    model=model_to_use,
                    messages=payload_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                content = response.choices[0].message.content

                if content:
                    print(
                        f"OpenAI API call successful. Preview: {content[:100].strip()}..."
                    )
                    return content
                else:
                    print(
                        f"OpenAI API returned empty content. Attempt {attempt + 1}/{max_retries}"
                    )

            except Exception as e:
                print(
                    f"OpenAI API call failed (Attempt {attempt + 1}/{max_retries}): {e}"
                )

            if attempt + 1 < max_retries:
                delay = base_delay_seconds * (2**attempt)
                print(f"Waiting for {delay} seconds before retrying...")
                await asyncio.sleep(delay)

        print("Max retries reached for OpenAI API call. Returning empty string.")
        return ""

    def _parse_hierarchy_xml(self, xml_string: str) -> List[Dict[str, Any]]:
        """Parse the XML response to extract hierarchical structures."""
        hierarchies = []

        cleaned_xml = self._clean_xml_string(xml_string)
        if not cleaned_xml:
            print("XML string is empty after cleaning. Cannot parse hierarchies.")
            return hierarchies

        try:

            if cleaned_xml.strip().startswith("<hierarchy"):
                # Wrap in a root element
                cleaned_xml = f"<hierarchies>{cleaned_xml}</hierarchies>"
                print("Wrapped direct <hierarchy> tag in <hierarchies> root element")

            root = ET.fromstring(cleaned_xml)

            for hierarchy_elem in root.findall(".//hierarchy"):
                hierarchy = {}

                # Extract basic hierarchy information
                name_elem = hierarchy_elem.find("name")
                desc_elem = hierarchy_elem.find("description")
                root_type_elem = hierarchy_elem.find("root_type")

                hierarchy["name"] = (
                    name_elem.text.strip()
                    if name_elem is not None and name_elem.text
                    else "Unnamed Hierarchy"
                )
                hierarchy["description"] = (
                    desc_elem.text.strip()
                    if desc_elem is not None and desc_elem.text
                    else ""
                )
                hierarchy["root_type"] = (
                    root_type_elem.text.strip()
                    if root_type_elem is not None and root_type_elem.text
                    else "Unknown"
                )

                # Process levels
                levels = []
                for level_elem in hierarchy_elem.findall(".//levels/level"):
                    level = {
                        "id": level_elem.get("id", ""),
                        "name": (
                            level_elem.get("name", "") or level_elem.find("name").text
                            if level_elem.find("name")
                            else ""
                        ),
                    }

                    # Process level description
                    level_desc_elem = level_elem.find("description")
                    if level_desc_elem is not None and level_desc_elem.text:
                        level["description"] = level_desc_elem.text.strip()

                    # Process nodes in this level
                    nodes = []
                    for node_elem in level_elem.findall(".//nodes/node"):
                        node = {
                            "id": node_elem.get("id", ""),
                        }

                        # Process node name
                        node_name_elem = node_elem.find("name")
                        if node_name_elem is not None and node_name_elem.text:
                            node["name"] = node_name_elem.text.strip()

                        # Process children references
                        children = []
                        for child_ref_elem in node_elem.findall(
                            ".//children/child_ref"
                        ):
                            children.append(
                                {
                                    "level": child_ref_elem.get("level", ""),
                                    "node_id": child_ref_elem.get("node_id", ""),
                                }
                            )

                        if children:
                            node["children"] = children

                        # Process data sources
                        data_sources_elem = node_elem.find("data_sources")
                        if data_sources_elem is not None and data_sources_elem.text:
                            node["data_sources"] = data_sources_elem.text.strip()

                        nodes.append(node)

                    if nodes:
                        level["nodes"] = nodes

                    levels.append(level)

                if levels:
                    hierarchy["levels"] = levels

                # Process relationships
                relationships = []
                for rel_elem in hierarchy_elem.findall(".//relationships/relationship"):
                    relationship = {
                        "type": rel_elem.get("type", ""),
                    }

                    # Process source and target
                    source_elem = rel_elem.find("source")
                    if source_elem is not None:
                        relationship["source"] = {
                            "node_id": source_elem.get("node_id", ""),
                            "level": source_elem.get("level", ""),
                        }

                    target_elem = rel_elem.find("target")
                    if target_elem is not None:
                        relationship["target"] = {
                            "node_id": target_elem.get("node_id", ""),
                            "level": target_elem.get("level", ""),
                        }

                    # Process description and data sources
                    desc_elem = rel_elem.find("description")
                    if desc_elem is not None and desc_elem.text:
                        relationship["description"] = desc_elem.text.strip()

                    data_sources_elem = rel_elem.find("data_sources")
                    if data_sources_elem is not None and data_sources_elem.text:
                        relationship["data_sources"] = data_sources_elem.text.strip()

                    relationships.append(relationship)

                if relationships:
                    hierarchy["relationships"] = relationships

                hierarchies.append(hierarchy)

            print(
                f"Successfully parsed {len(hierarchies)} hierarchies using ET.fromstring."
            )

        except ET.ParseError as e:
            print(f"XML parsing error for hierarchies: {e}")
            # Implement fallback parsing with regex if needed

        except Exception as e:
            print(f"An unexpected error occurred during hierarchy XML parsing: {e}")

        return hierarchies

    def _clean_xml_string(self, xml_string: str) -> str:
        """Cleans the XML string from common LLM artifacts and prepares it for parsing."""
        if not isinstance(xml_string, str):
            print(
                f"XML input is not a string, type: {type(xml_string)}. Returning empty string."
            )
            return ""

        cleaned_xml = xml_string.strip()

        if cleaned_xml.startswith("```xml"):
            cleaned_xml = cleaned_xml[len("```xml") :].strip()
        elif cleaned_xml.startswith("```"):
            cleaned_xml = cleaned_xml[len("```") :].strip()

        if cleaned_xml.endswith("```"):
            cleaned_xml = cleaned_xml[: -len("```")].strip()

        if cleaned_xml.startswith("<?xml"):
            end_decl = cleaned_xml.find("?>")
            if end_decl != -1:
                cleaned_xml = cleaned_xml[end_decl + 2 :].lstrip()

        first_angle_bracket = cleaned_xml.find("<")
        last_angle_bracket = cleaned_xml.rfind(">")

        if (
            first_angle_bracket != -1
            and last_angle_bracket != -1
            and last_angle_bracket > first_angle_bracket
        ):
            cleaned_xml = cleaned_xml[first_angle_bracket : last_angle_bracket + 1]
        elif first_angle_bracket == -1:
            print(
                f"No XML tags found in the string after initial cleaning. Original: {xml_string[:200]}"
            )
            return ""

        cleaned_xml = re.sub(
            r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)", "&amp;", cleaned_xml
        )

        common_prefixes = [
            "Sure, here is the XML:",
            "Here's the XML output:",
            "Okay, here's the XML:",
        ]
        for prefix in common_prefixes:
            if cleaned_xml.lower().startswith(prefix.lower()):
                cleaned_xml = cleaned_xml[len(prefix) :].lstrip()
                break

        cleaned_xml = re.sub(
            r"[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]",
            "",
            cleaned_xml,
        )

        return cleaned_xml

    def _parse_graph_xml(
        self, xml_string: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entities = []
        relationships = []

        cleaned_xml = self._clean_xml_string(xml_string)

        if not cleaned_xml:
            print("XML string is empty after cleaning. Cannot parse.")
            return entities, relationships

        has_known_root = False
        known_roots = ["<graph>", "<root>", "<entities>", "<response>", "<data>"]
        for root_tag_start in known_roots:
            if cleaned_xml.startswith(root_tag_start):
                root_tag_name = root_tag_start[1:-1]
                if cleaned_xml.endswith(f"</{root_tag_name}>"):
                    has_known_root = True
                    break

        string_to_parse = cleaned_xml
        if not has_known_root:
            if (
                cleaned_xml.startswith("<entity")
                or cleaned_xml.startswith("<relationship")
            ) and (
                cleaned_xml.endswith("</entity>")
                or cleaned_xml.endswith("</relationship>")
            ):
                string_to_parse = f"<root_wrapper>{cleaned_xml}</root_wrapper>"
                print(
                    "Wrapping multiple top-level entity/relationship tags with <root_wrapper>."
                )
            elif not (
                cleaned_xml.count("<") > 1
                and cleaned_xml.count(">") > 1
                and cleaned_xml.find("</") > 0
                and cleaned_xml.startswith("<")
                and cleaned_xml.endswith(">")
                and cleaned_xml[
                    1 : cleaned_xml.find(">") if cleaned_xml.find(">") > 1 else 0
                ].strip()
                == cleaned_xml[cleaned_xml.rfind("</") + 2 : -1].strip()
            ):
                string_to_parse = f"<root_wrapper>{cleaned_xml}</root_wrapper>"
                print(
                    "Wrapping content with <root_wrapper> as it doesn't appear to have a single root or matching end tag."
                )

        try:
            root = ET.fromstring(string_to_parse)

            for entity_elem in root.findall(".//entity"):
                name_val = entity_elem.get("name")
                if not name_val:
                    name_elem = entity_elem.find("name")
                    name_val = (
                        name_elem.text.strip()
                        if name_elem is not None and name_elem.text
                        else None
                    )

                ent_type_elem = entity_elem.find("type")
                ent_desc_elem = entity_elem.find("description")

                ent_type = (
                    ent_type_elem.text.strip()
                    if ent_type_elem is not None and ent_type_elem.text
                    else "Unknown"
                )
                ent_desc = (
                    ent_desc_elem.text.strip()
                    if ent_desc_elem is not None and ent_desc_elem.text
                    else ""
                )

                if name_val:
                    entities.append(
                        {
                            "name": name_val.strip(),
                            "type": ent_type,
                            "description": ent_desc,
                        }
                    )

            for rel_elem in root.findall(".//relationship"):
                source_elem = rel_elem.find("source")
                target_elem = rel_elem.find("target")
                rel_type_elem = rel_elem.find("type")
                rel_desc_elem = rel_elem.find("description")
                rel_weight_elem = rel_elem.find("weight")

                source = (
                    source_elem.text.strip()
                    if source_elem is not None and source_elem.text
                    else None
                )
                target = (
                    target_elem.text.strip()
                    if target_elem is not None and target_elem.text
                    else None
                )
                rel_type = (
                    rel_type_elem.text.strip()
                    if rel_type_elem is not None and rel_type_elem.text
                    else "RELATED_TO"
                )
                rel_desc = (
                    rel_desc_elem.text.strip()
                    if rel_desc_elem is not None and rel_desc_elem.text
                    else ""
                )
                weight = None
                if rel_weight_elem is not None and rel_weight_elem.text:
                    try:
                        weight = float(rel_weight_elem.text.strip())
                    except ValueError:
                        print(
                            f"Could not parse relationship weight '{rel_weight_elem.text}' as float."
                        )

                if source and target:
                    relationships.append(
                        {
                            "source_entity": source,
                            "target_entity": target,
                            "relation": rel_type,
                            "relationship_description": rel_desc,
                            "relationship_weight": weight,
                        }
                    )

            print(
                f"Successfully parsed {len(entities)} entities and {len(relationships)} relationships using ET.fromstring."
            )

        except ET.ParseError as e:
            err_line, err_col = e.position if hasattr(e, "position") else (-1, -1)
            log_message = (
                f"XML parsing error with ET.fromstring: {e}\n"
                f"Error at line {err_line}, column {err_col} (approximate). Trying regex-based extraction as fallback.\n"
                f"Cleaned XML snippet attempted (first 1000 chars):\n{string_to_parse[:1000]}"
            )
            print(log_message)

            entities = []
            relationships = []

            entity_pattern_attr = r'<entity\s+name\s*=\s*"([^"]*)"\s*>\s*(?:<type>([^<]*)</type>)?\s*(?:<description>([^<]*)</description>)?\s*</entity>'
            entity_pattern_tag = r"<entity>\s*<name>([^<]+)</name>\s*(?:<type>([^<]*)</type>)?\s*(?:<description>([^<]*)</description>)?\s*</entity>"

            for pattern in [entity_pattern_attr, entity_pattern_tag]:
                for match in re.finditer(pattern, string_to_parse):
                    name, entity_type, description = match.groups()
                    if name:
                        entities.append(
                            {
                                "name": name.strip(),
                                "type": (
                                    entity_type.strip()
                                    if entity_type and entity_type.strip()
                                    else "Unknown"
                                ),
                                "description": (
                                    description.strip()
                                    if description and description.strip()
                                    else ""
                                ),
                            }
                        )

            rel_pattern = r"<relationship>\s*(?:<source>([^<]+)</source>)?\s*(?:<target>([^<]+)</target>)?\s*(?:<type>([^<]*)</type>)?\s*(?:<description>([^<]*)</description>)?\s*(?:<weight>([^<]*)</weight>)?\s*</relationship>"
            for match in re.finditer(rel_pattern, string_to_parse):
                source, target, rel_type, description, weight_str = match.groups()
                if source and target:
                    weight = None
                    if weight_str and weight_str.strip():
                        try:
                            weight = float(weight_str.strip())
                        except ValueError:
                            print(
                                f"Regex fallback: Could not parse weight '{weight_str}' for relationship."
                            )

                    relationships.append(
                        {
                            "source_entity": source.strip(),
                            "target_entity": target.strip(),
                            "relation": (
                                rel_type.strip()
                                if rel_type and rel_type.strip()
                                else "RELATED_TO"
                            ),
                            "relationship_description": (
                                description.strip()
                                if description and description.strip()
                                else ""
                            ),
                            "relationship_weight": weight,
                        }
                    )
            if entities or relationships:
                print(
                    f"Regex fallback extracted {len(entities)} entities and {len(relationships)} relationships."
                )
            else:
                print(
                    "Regex fallback also failed to extract any entities or relationships."
                )

        except Exception as final_e:
            print(
                f"An unexpected error occurred during XML parsing (after ET.ParseError or during regex): {final_e}\n"
                f"Original XML content from LLM (first 500 chars):\n{xml_string[:500]}\n"
                f"Cleaned XML attempted for parsing (first 500 chars):\n{string_to_parse[:500]}"
            )

        return entities, relationships

    async def _extract_knowledge_graph(
        self, chunk_text: str, document_summary: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if self.Server_type == "ARMY":
            print("Extracting knowledge graph using NVIDIA API.")
            formatted_prompt = self.graph_extraction_prompt_template.format(
                document_summary=document_summary,
                input=chunk_text,
                entity_types=str([]),
                relation_types=str([]),
            )
            messages = [
                {
                    "role": "system",
                    "content": 'You are an expert assistant that extracts entities and relationships from text and formats them as XML according to the provided schema. Ensure all tags are correctly opened and closed. Use <entity name="..."><type>...</type><description>...</description></entity> and <relationship><source>...</source><target>...</target><type>...</type><description>...</description><weight>...</weight></relationship> format. Wrap multiple entities and relationships in a single <root> or <graph> tag',
                },
                {"role": "user", "content": formatted_prompt},
            ]
            xml_response = await self._call_nvidia_api(
                payload_messages=messages, max_tokens=4000, temperature=0.1
            )
            return self._parse_graph_xml(xml_response) if xml_response else ([], [])

        if not self.aclient_openai or not self.graph_extraction_prompt_template:
            print(
                "OpenAI client or graph extraction prompt not available. Skipping graph extraction."
            )
            return [], []

        formatted_prompt = self.graph_extraction_prompt_template.format(
            document_summary=document_summary,
            input=chunk_text,
            entity_types=str([]),
            relation_types=str([]),
        )
        print(
            f"Formatted prompt for graph extraction (chunk-level, to {OPENAI_SUMMARY_MODEL}): First 200 chars: {formatted_prompt[:200]}..."
        )

        messages = [
            {
                "role": "system",
                "content": 'You are an expert assistant that extracts entities and relationships from text and formats them as XML according to the provided schema. Ensure all tags are correctly opened and closed. Use <entity name="..."><type>...</type><description>...</description></entity> and <relationship><source>...</source><target>...</target><type>...</type><description>...</description><weight>...</weight></relationship> format. Wrap multiple entities and relationships in a single <root> or <graph> tag.',
            },
            {"role": "user", "content": formatted_prompt},
        ]

        xml_response_content = await self._call_openai_api(
            model_name=OPENAI_SUMMARY_MODEL,
            payload_messages=messages,
            max_tokens=4000,
            temperature=0.1,
        )

        if not xml_response_content:
            print("LLM returned empty content for graph extraction.")
            return [], []

        print(
            f"Raw XML response from LLM for chunk-level graph extraction (first 500 chars):\n{xml_response_content[:500]}"
        )
        return self._parse_graph_xml(xml_response_content)

    async def _enrich_chunk_content(
        self,
        chunk_text: str,
        document_summary: str,
        preceding_chunks_texts: List[str],
        succeeding_chunks_texts: List[str],
    ) -> str:
        if self.Server_type == "ARMY":
            print("Enriching chunk content using NVIDIA API.")
            context_prompt = self.enrich_prompt_template.format(
                document_summary=document_summary,
                preceding_chunks="\n---\n".join(preceding_chunks_texts),
                succeeding_chunks="\n---\n".join(succeeding_chunks_texts),
                chunk=chunk_text,
                chunk_size=CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN_ESTIMATE,
            )
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert assistant that refines and enriches text chunks according to specific guidelines",
                },
                {"role": "user", "content": context_prompt},
            ]
            enriched_text = await self._call_nvidia_api(
                payload_messages=messages,
                max_tokens=min(CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS, 4000),
                temperature=0.3,
            )
            return enriched_text if enriched_text else chunk_text

        if not self.aclient_openai or not self.enrich_prompt_template:
            print(
                "OpenAI client or enrichment prompt not available. Skipping enrichment."
            )
            return chunk_text

        preceding_context = "\n---\n".join(preceding_chunks_texts)
        succeeding_context = "\n---\n".join(succeeding_chunks_texts)
        max_output_chars = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN_ESTIMATE

        formatted_prompt = self.enrich_prompt_template.format(
            document_summary=document_summary,
            preceding_chunks=preceding_context,
            succeeding_chunks=succeeding_context,
            chunk=chunk_text,
            chunk_size=max_output_chars,
        )
        print(
            f"Formatted prompt for enrichment (to be sent to {OPENAI_CHAT_MODEL}): ..."
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert assistant that refines and enriches text chunks according to specific guidelines.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        enriched_text_content = await self._call_openai_api(
            model_name=OPENAI_CHAT_MODEL,
            payload_messages=messages,
            max_tokens=min(CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS, 4000),
            temperature=0.3,
        )

        if not enriched_text_content:
            print(
                "LLM returned empty content for chunk enrichment. Using original chunk."
            )
            return chunk_text

        enriched_text = enriched_text_content.strip()
        print(
            f"Chunk enriched. Original length: {len(chunk_text)}, Enriched length: {len(enriched_text)}"
        )
        return enriched_text

    async def _generate_document_summary(self, full_document_text: str) -> str:
        if self.Server_type == "ARMY":
            print("Generating document summary using NVIDIA API.")
            summary_prompt = self.summary_prompt_template.format(
                document=full_document_text
            )
            messages = [{"role": "user", "content": summary_prompt}]

            summary_text = await self._call_nvidia_api(
                payload_messages=messages,
                max_tokens=SUMMARY_MAX_TOKENS,
                temperature=0.3,
            )
            return (
                summary_text.strip()
                if summary_text
                else "Summary generation via NVIDIA failed."
            )

        if not self.aclient_openai or not self.summary_prompt_template:
            print(
                "OpenAI client or summary prompt not available. Skipping document summary generation."
            )
            return "Summary generation skipped due to missing configuration."
        if not full_document_text.strip():
            print("Full document text is empty. Skipping summary generation.")
            return "Document is empty, no summary generated."

        formatted_prompt = self.summary_prompt_template.format(
            document=full_document_text
        )
        print(f"Generating document summary using {OPENAI_SUMMARY_MODEL}...")

        messages = [{"role": "user", "content": formatted_prompt}]

        summary_text_content = await self._call_openai_api(
            model_name=OPENAI_SUMMARY_MODEL,
            payload_messages=messages,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.3,
        )

        if not summary_text_content:
            print("LLM returned empty content for document summary.")
            return "Summary generation resulted in empty content."

        summary_text = summary_text_content.strip()
        print(f"Document summary generated. Length: {len(summary_text)} chars.")
        return summary_text

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.Server_type == "ARMY":
            embedding_service_url = os.getenv(
                "EMBEDDING_API_URL", "http://localhost:8000/embed"
            )

            # Process in smaller batches
            batch_size = 1
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                max_retries = 20

                for attempt in range(max_retries):
                    try:
                        async with httpx.AsyncClient(timeout=60.0) as client:
                            print(
                                f"Generating embeddings for {len(batch_texts)} texts via API: {embedding_service_url} (batch {i//batch_size + 1}, attempt {attempt + 1})"
                            )

                            response = await client.post(
                                embedding_service_url,
                                json={
                                    "texts": batch_texts,
                                    "task": "retrieval.passage",
                                },
                            )
                            response.raise_for_status()

                            result = response.json()
                            batch_embeddings = result.get("embeddings", [])

                            if batch_embeddings and len(batch_embeddings) == len(
                                batch_texts
                            ):
                                all_embeddings.extend(batch_embeddings)
                                print(
                                    f"✅ Successfully received {len(batch_embeddings)} embeddings from service."
                                )
                                break  # Success, exit retry loop
                            else:
                                print(
                                    f"❌ Mismatch in embedding count. Expected {len(batch_texts)}, got {len(batch_embeddings)}"
                                )
                                if attempt == max_retries - 1:
                                    # Last attempt failed, add dummy embeddings
                                    dummy_embeddings = [
                                        [0.0] * 1024 for _ in batch_texts
                                    ]
                                    all_embeddings.extend(dummy_embeddings)

                    except Exception as e:
                        print(
                            f"❌ Error calling embedding service (attempt {attempt + 1}/{max_retries}): {str(e)}"
                        )
                        if attempt == max_retries - 1:
                            # Last attempt failed, add dummy embeddings
                            dummy_embeddings = [[0.0] * 1024 for _ in batch_texts]
                            all_embeddings.extend(dummy_embeddings)
                        else:
                            # Wait before retrying
                            await asyncio.sleep(
                                3 * (attempt + 1)
                            )  # Exponential backoff

                # Add delay between batches to avoid overwhelming the service
                if i + batch_size < len(texts):
                    await asyncio.sleep(1)

            return all_embeddings

        if not self.aclient_openai:
            print("OpenAI client not available. Cannot generate embeddings.")
            return [[] for _ in texts]

        all_embeddings = []
        openai_batch_size = 2048
        try:
            for i in range(0, len(texts), openai_batch_size):
                batch_texts = texts[i : i + openai_batch_size]
                processed_batch_texts = [
                    text if text.strip() else " " for text in batch_texts
                ]

                response = await self.aclient_openai.embeddings.create(
                    input=processed_batch_texts,
                    model=OPENAI_EMBEDDING_MODEL,
                    dimensions=OPENAI_EMBEDDING_DIMENSIONS,
                )
                all_embeddings.extend([item.embedding for item in response.data])
            return all_embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [[] for _ in texts]

    async def _generate_all_raw_chunks_from_doc(
        self, doc_text: str, file_name: str, doc_id: str, page_breaks: List[int] = None
    ) -> List[Dict[str, Any]]:
        all_raw_chunks_with_meta: List[Dict[str, Any]] = []
        if not doc_text or not doc_text.strip():
            print(f"Skipping empty document {file_name} for raw chunk generation.")
            return []

        raw_chunks = self.text_splitter.split_text(doc_text)

        if not page_breaks or len(page_breaks) <= 1:
            for chunk_idx, raw_chunk_text in enumerate(raw_chunks):
                doc_file_name = self.params.get("file_name", file_name)
                print(
                    f"RAW CHUNK (File: {file_name}, Original File Name: {doc_file_name}, Page: 1, Idx: {chunk_idx}, Len {len(raw_chunk_text)}): '''{raw_chunk_text[:100].strip()}...'''"
                )
                all_raw_chunks_with_meta.append(
                    {
                        "text": raw_chunk_text,
                        "page_num": 1,
                        "chunk_idx_on_page": chunk_idx,
                        "file_name": doc_file_name,
                        "doc_id": doc_id,
                    }
                )

        else:
            # Assign page numbers based on character positions
            current_page = 1
            next_page_break_idx = 0
            current_pos = 0

            for chunk_idx, raw_chunk_text in enumerate(raw_chunks):
                chunk_start_pos = current_pos
                chunk_end_pos = current_pos + len(raw_chunk_text)

                # Update current page if we've crossed a page break
                while (
                    next_page_break_idx < len(page_breaks)
                    and chunk_start_pos >= page_breaks[next_page_break_idx]
                ):
                    current_page += 1
                    next_page_break_idx += 1

                doc_file_name = self.params.get("file_name", file_name)
                print(
                    f"RAW CHUNK (File: {file_name}, Original File Name: {doc_file_name}, Page: {current_page}, Idx: {chunk_idx}, Len {len(raw_chunk_text)}): '''{raw_chunk_text[:100].strip()}...'''"
                )

                all_raw_chunks_with_meta.append(
                    {
                        "text": raw_chunk_text,
                        "page_num": current_page,
                        "chunk_idx_on_page": chunk_idx,
                        "file_name": doc_file_name,
                        "doc_id": doc_id,
                    }
                )

                current_pos = chunk_end_pos

        print(
            f"Generated {len(all_raw_chunks_with_meta)} raw chunks from {file_name} across multiple pages."
        )
        return all_raw_chunks_with_meta

    async def _process_individual_chunk_pipeline(
        self,
        raw_chunk_info: Dict[str, Any],
        user_provided_doc_summary: str,
        llm_generated_doc_summary: str,
        all_raw_texts: List[str],
        global_idx: int,
        file_name: str,
        doc_id: str,
        params: Any,
    ) -> Dict[str, Any] | None:
        chunk_text = raw_chunk_info["text"]
        page_num = raw_chunk_info["page_num"]
        chunk_idx_on_page = raw_chunk_info["chunk_idx_on_page"]

        print(
            f"Starting pipeline for chunk: File {file_name}, Page {page_num}, Index {chunk_idx_on_page}"
        )

        file_extension = os.path.splitext(file_name)[1].lower()
        is_tabular_file = file_extension in [".csv", ".xlsx"]
        is_docx_file = file_extension == ".docx"
        # is_pdf_file = file_extension == '.pdf'
        is_ocr_pdf = params.get("is_ocr_pdf", False)

        chunk_entities = []
        chunk_relationships = []
        chunk_hierarchies = []
        enriched_text = chunk_text

        if is_tabular_file:
            print(
                f"Tabular file ({file_extension}) detected. Skipping KG extraction and enrichment."
            )
            # For tabular files, skip all knowledge extraction
            enriched_text = chunk_text
            chunk_entities, chunk_relationships, chunk_hierarchies = [], [], []

        elif is_ocr_pdf:
            print(
                f"OCR PDF detected. Using raw chunks without enrichment for '{file_name}'."
            )
            # For OCR PDFs, only perform KG extraction, skip enrichment
            preceding_indices = range(max(0, global_idx - BACKWARD_CHUNKS), global_idx)
            succeeding_indices = range(
                global_idx + 1, min(len(all_raw_texts), global_idx + 1 + FORWARD_CHUNKS)
            )
            preceding_texts = [all_raw_texts[i] for i in preceding_indices]
            succeeding_texts = [all_raw_texts[i] for i in succeeding_indices]

            contextual_summary = llm_generated_doc_summary
            if (
                not llm_generated_doc_summary
                or llm_generated_doc_summary
                == "Document is empty, no summary generated."
                or llm_generated_doc_summary.startswith(
                    "Error during summary generation"
                )
                or llm_generated_doc_summary
                == "Summary generation skipped due to missing configuration."
            ):
                contextual_summary = user_provided_doc_summary

            # Perform KG and hierarchy extraction for OCR PDFs, no enrichment
            try:
                kg_task = asyncio.create_task(
                    self._extract_knowledge_graph(chunk_text, contextual_summary)
                )
                hierarchy_task = asyncio.create_task(
                    self._extract_hierarchies(chunk_text, contextual_summary)
                )

                kg_result, hierarchy_result = await asyncio.gather(
                    kg_task, hierarchy_task, return_exceptions=True
                )

                if not isinstance(kg_result, Exception) and kg_result:
                    chunk_entities, chunk_relationships = kg_result
                    print(
                        f"KG extracted for OCR PDF chunk (Page {page_num}, Index {chunk_idx_on_page}): {len(chunk_entities)} entities, {len(chunk_relationships)} relationships."
                    )
                else:
                    chunk_entities, chunk_relationships = [], []
                    print(
                        f"No KG extracted for OCR PDF chunk (Page {page_num}, Index {chunk_idx_on_page})."
                    )

                if not isinstance(hierarchy_result, Exception) and hierarchy_result:
                    chunk_hierarchies = hierarchy_result
                    print(
                        f"Hierarchies extracted for OCR PDF chunk (Page {page_num}, Index {chunk_idx_on_page}): {len(chunk_hierarchies)} hierarchies."
                    )
                else:
                    chunk_hierarchies = []
                    print(
                        f"No hierarchies extracted for OCR PDF chunk (Page {page_num}, Index {chunk_idx_on_page})."
                    )

            except Exception as e:
                print(
                    f"KG/Hierarchy extraction failed for OCR PDF chunk (Page {page_num}, Index {chunk_idx_on_page}) for '{file_name}': {e}"
                )
                chunk_entities, chunk_relationships, chunk_hierarchies = [], [], []

            # Use original chunk text without enrichment for OCR PDFs
            enriched_text = chunk_text

        else:
            preceding_indices = range(max(0, global_idx - BACKWARD_CHUNKS), global_idx)
            succeeding_indices = range(
                global_idx + 1, min(len(all_raw_texts), global_idx + 1 + FORWARD_CHUNKS)
            )
            preceding_texts = [all_raw_texts[i] for i in preceding_indices]
            succeeding_texts = [all_raw_texts[i] for i in succeeding_indices]

            contextual_summary = llm_generated_doc_summary
            if (
                not llm_generated_doc_summary
                or llm_generated_doc_summary
                == "Document is empty, no summary generated."
                or llm_generated_doc_summary.startswith(
                    "Error during summary generation"
                )
                or llm_generated_doc_summary
                == "Summary generation skipped due to missing configuration."
            ):
                contextual_summary = user_provided_doc_summary

            kg_task = asyncio.create_task(
                self._extract_knowledge_graph(chunk_text, contextual_summary)
            )

            hierarchy_task = asyncio.create_task(
                self._extract_hierarchies(chunk_text, contextual_summary)
            )

            if is_docx_file:
                # For DOCX files, only perform KG extraction, not enrichment
                print(
                    f"DOCX file detected. Skipping chunk enrichment but performing KG extraction for '{file_name}'."
                )

                # Await both KG and hierarchy tasks
                try:
                    kg_result, hierarchy_result = await asyncio.gather(
                        kg_task, hierarchy_task, return_exceptions=True
                    )

                    if not isinstance(kg_result, Exception) and kg_result:
                        chunk_entities, chunk_relationships = kg_result
                        print(
                            f"KG extracted for DOCX chunk (Page {page_num}, Index {chunk_idx_on_page}): {len(chunk_entities)} entities, {len(chunk_relationships)} relationships."
                        )
                    else:
                        chunk_entities, chunk_relationships = [], []
                        print(
                            f"No KG extracted for DOCX chunk (Page {page_num}, Index {chunk_idx_on_page})."
                        )

                    if not isinstance(hierarchy_result, Exception) and hierarchy_result:
                        chunk_hierarchies = hierarchy_result
                        print(
                            f"Hierarchies extracted for DOCX chunk (Page {page_num}, Index {chunk_idx_on_page}): {len(chunk_hierarchies)} hierarchies."
                        )
                    else:
                        chunk_hierarchies = []
                        print(
                            f"No hierarchies extracted for DOCX chunk (Page {page_num}, Index {chunk_idx_on_page})."
                        )

                except Exception as e:
                    print(
                        f"KG/Hierarchy extraction failed for DOCX chunk (Page {page_num}, Index {chunk_idx_on_page}) for '{file_name}': {e}"
                    )
                    chunk_entities, chunk_relationships, chunk_hierarchies = [], [], []

            else:
                enrich_task = asyncio.create_task(
                    self._enrich_chunk_content(
                        chunk_text,
                        contextual_summary,
                        preceding_texts,
                        succeeding_texts,
                    )
                )

                # Await both tasks for non-DOCX files
                try:
                    results = await asyncio.gather(
                        kg_task, hierarchy_task, enrich_task, return_exceptions=True
                    )

                    if len(results) > 0:
                        kg_result_or_exc = results[0]
                        if not isinstance(kg_result_or_exc, Exception):
                            chunk_entities, chunk_relationships = (
                                kg_result_or_exc if kg_result_or_exc else ([], [])
                            )
                            print(
                                f"KG extracted for chunk: {len(chunk_entities)} entities, {len(chunk_relationships)} relationships."
                            )

                    if len(results) > 1:
                        hierarchy_result_or_exc = results[1]
                        if not isinstance(hierarchy_result_or_exc, Exception):
                            chunk_hierarchies = (
                                hierarchy_result_or_exc
                                if hierarchy_result_or_exc
                                else []
                            )
                            print(
                                f"Hierarchies extracted for chunk: {len(chunk_hierarchies)} hierarchies."
                            )

                    if len(results) > 2:
                        enrich_result_or_exc = results[2]
                        if not isinstance(enrich_result_or_exc, Exception):
                            enriched_text = enrich_result_or_exc

                except Exception as e:
                    print(f"Error processing tasks for '{file_name}': {e}")

            if chunk_entities:
                # Create a list of descriptions to embed
                descriptions_to_embed = []
                entities_with_descriptions = []
                entities_indices = []

                for i, entity in enumerate(chunk_entities):
                    description = entity.get("description", "").strip()
                    if description:
                        descriptions_to_embed.append(description)
                        entities_with_descriptions.append(entity)
                        entities_indices.append(i)

                if descriptions_to_embed:
                    print(
                        f"Generating embeddings for {len(descriptions_to_embed)} entity descriptions."
                    )
                    try:
                        # Generate embeddings in one batch
                        description_embeddings = await self._generate_embeddings(
                            descriptions_to_embed
                        )

                        if description_embeddings and len(
                            description_embeddings
                        ) == len(entities_with_descriptions):
                            # Assign embeddings to entities
                            for i, embedding in enumerate(description_embeddings):
                                if embedding:  # Make sure we have a valid embedding
                                    orig_idx = entities_indices[i]
                                    chunk_entities[orig_idx][
                                        "description_embedding"
                                    ] = embedding

                            print(
                                f"Successfully assigned {len(description_embeddings)} embeddings to entity descriptions."
                            )

                            # Debug - check if embeddings were properly assigned
                            count_with_embeddings = sum(
                                1
                                for e in chunk_entities
                                if "description_embedding" in e
                            )
                            print(
                                f"Entities with embeddings after assignment: {count_with_embeddings}/{len(chunk_entities)}"
                            )
                        else:
                            print(
                                f"Mismatch between entity descriptions ({len(entities_with_descriptions)}) and embeddings ({len(description_embeddings) if description_embeddings else 0})"
                            )
                    except Exception as e:
                        print(
                            f"Failed to generate embeddings for entity descriptions: {e}"
                        )
                        traceback.print_exc()

        doc_file_name = self.params.get("file_name", file_name)
        enriched_text = f"[File: {doc_file_name}]\n\n{enriched_text}"

        embedding_list = await self._generate_embeddings([enriched_text])

        embedding_vector = []
        if embedding_list and embedding_list[0]:
            embedding_vector = embedding_list[0]

        if not embedding_vector:
            print(
                f"Skipping chunk from page {page_num}, index {chunk_idx_on_page} for '{file_name}' due to missing embedding."
            )
            return None

        es_doc_id = f"{doc_id}_p{page_num}_c{chunk_idx_on_page}"
        doc_file_name = self.params.get("file_name", file_name)
        print(f"Original file name: {doc_file_name}")
        metadata_payload = {
            "file_name": doc_file_name,
            "doc_id": doc_id,
            "page_number": page_num,
            "chunk_index_in_page": chunk_idx_on_page,
            "document_summary": llm_generated_doc_summary,
            "entities": chunk_entities,
            "relationships": chunk_relationships,
            "hierarchies": chunk_hierarchies,
        }

        index_name = params.get("index_name")
        print("index name:--", index_name)
        action = {
            "_index": index_name,
            "_id": es_doc_id,
            "_source": {
                "chunk_text": enriched_text,
                "embedding": embedding_vector,
                "metadata": metadata_payload,
            },
        }
        print(
            f"Pipeline complete for chunk (Page {page_num}, Index {chunk_idx_on_page}). ES action prepared."
        )
        return action

    async def process_pdf_structured(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes a PDF file in 'structured' mode: each page is a chunk, no enrichment, no KG.
        """
        print(f"Processing PDF in structured mode: {file_name} (Doc ID: {doc_id})")
        parser = PDFParser(self.aclient_openai, self)
        try:
            with pdfplumber.open(BytesIO(data)) as pdf:
                if not pdf.pages:
                    print("PDF has no pages. Aborting.")
                    return

                all_content_parts = []
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text() or ""
                    tables = page.extract_tables()
                    table_markdown = ""
                    for table in tables:
                        table_markdown += parser._convert_table_to_markdown(table)
                    # Optionally, add image descriptions if needed
                    chunk_text = (
                        page_text.strip() + "\n" + table_markdown.strip()
                    ).strip()
                    if not chunk_text:
                        continue

                    # Generate embedding for the chunk
                    embedding_list = await self._generate_embeddings([chunk_text])
                    embedding_vector = (
                        embedding_list[0]
                        if embedding_list and embedding_list[0]
                        else []
                    )

                    if not embedding_vector:
                        print(f"Skipping page {page_num} due to missing embedding.")
                        continue

                    es_doc_id = f"{doc_id}_p{page_num}_structured"
                    doc_file_name = self.params.get("file_name", file_name)
                    metadata_payload = {
                        "file_name": doc_file_name,
                        "doc_id": doc_id,
                        "page_number": page_num,
                        "chunk_index_in_page": 0,
                        "document_summary": user_provided_document_summary,
                        "entities": [],
                        "relationships": [],
                        "hierarchies": [],
                    }
                    index_name = self.params.get("index_name")
                    action = {
                        "_index": index_name,
                        "_id": es_doc_id,
                        "_source": {
                            "chunk_text": chunk_text,
                            "embedding": embedding_vector,
                            "metadata": metadata_payload,
                        },
                    }
                    yield action
        except Exception as e:
            print(f"Error in structured PDF processing: {e}")
            import traceback

            traceback.print_exc()

    async def process_pdf(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Orchestrates PDF processing, splitting large documents into 100-page batches.
        """
        print(f"Processing PDF: {file_name} (Doc ID: {doc_id})")
        batch_size = 100

        try:
            with fitz.open(stream=data, filetype="pdf") as doc:
                total_pages = len(doc)
                print(
                    f"PDF has {total_pages} pages. Processing in batches of up to {batch_size} pages."
                )

            if total_pages <= batch_size:
                print("Document is small enough to be processed in a single batch.")
                async for action in self._process_pdf_batch(
                    data,
                    file_name,
                    doc_id,
                    user_provided_document_summary,
                    0,
                    total_pages,
                ):
                    yield action
                return

            for batch_start in range(0, total_pages, batch_size):
                batch_end = min(batch_start + batch_size, total_pages)
                print(
                    f"\n--- Processing batch of pages {batch_start + 1} to {batch_end} of {total_pages} ---"
                )

                # Create a new in-memory PDF for the current batch
                with fitz.open(stream=data, filetype="pdf") as original_doc:
                    with fitz.open() as batch_doc:
                        batch_doc.insert_pdf(
                            original_doc, from_page=batch_start, to_page=batch_end - 1
                        )
                        batch_data = batch_doc.write()

                try:
                    # Process the current batch
                    async for action in self._process_pdf_batch(
                        batch_data,
                        file_name,
                        doc_id,
                        user_provided_document_summary,
                        batch_start,
                        batch_end,
                    ):
                        yield action

                    # Add a delay between batches to avoid overwhelming services
                    if batch_end < total_pages:
                        print(
                            f"Completed batch for pages {batch_start + 1}-{batch_end}. Waiting 5 seconds before next batch..."
                        )
                        await asyncio.sleep(5)

                except Exception as e:
                    print(
                        f"❌ Error processing batch for pages {batch_start + 1}-{batch_end}: {e}"
                    )
                    print(
                        "Attempting to continue with the next batch after a 15-second delay..."
                    )
                    await asyncio.sleep(15)
                    continue

        except Exception as e:
            print(f"❌ A critical error occurred during PDF batching setup: {e}")
            print(
                "Falling back to processing the document as a single unit, which may fail for large files."
            )
            async for action in self._process_pdf_batch(
                data, file_name, doc_id, user_provided_document_summary, 0, -1
            ):
                yield action

    async def _process_pdf_batch(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
        batch_start_page: int,
        total_pages_in_batch: int,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing PDF: {file_name} (Doc ID: {doc_id})")

        use_ocr = self.params.get("is_ocr_pdf", False)

        try:
            # pages_with_tables = 0
            # with pdfplumber.open(BytesIO(data)) as pdf:
            #     if not pdf.pages:
            #         print("PDF has no pages. Aborting.")
            #         return

            #     for page in pdf.pages:
            #         if page.extract_tables():
            #             pages_with_tables += 1

            #     if pages_with_tables > 1:
            #         print(f"PDF contains tables on {pages_with_tables} pages. Adjusting chunk size to 2048.")
            #         self.text_splitter = RecursiveCharacterTextSplitter(
            #             chunk_size=2048,
            #             chunk_overlap=1024,
            #             length_function=tiktoken_len,
            #             separators=["\n|", "\n", "|", ". "," ", ""],
            #         )
            print(
                f"PDF processing: Setting consistent chunk size to 2048 with overlap 1024 for '{file_name}'."
            )
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=1024,
                length_function=tiktoken_len,
                separators=["\n|", "\n", "|", ". ", " ", ""],
            )

        except Exception as e:
            print(
                f"Could not pre-check PDF with pdfplumber, defaulting to OCR. Error: {e}"
            )
            use_ocr = True

        # full_document_text = ""
        # if use_ocr:
        #     print("Processing with OCR parser.")
        #     ocr_parser = OCRParser()
        #     ocr_texts = [page_text async for page_text in ocr_parser.ingest(data)]
        #     full_document_text = " ".join(ocr_texts)

        full_document_text = ""
        page_texts = []  # Store text with page numbers

        if use_ocr:
            ocr_engine = self.params.get("ocr_engine", "mistral")
            print(f"Processing with OCR parser, engine: {ocr_engine}.")
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2048,
                chunk_overlap=1024,
                length_function=tiktoken_len,
                separators=["\n|", "\n", "|", ". ", " ", ""],
            )

            try:
                if (
                    ocr_engine == "mistral"
                    and MISTRAL_API_KEY
                    and MISTRAL_SDK_AVAILABLE
                ):
                    print("Using Mistral OCR.")
                    mistral_parser = MistralOCRParser(api_key=MISTRAL_API_KEY)
                    # Modified to capture page numbers
                    async for page_text, page_num in mistral_parser.ingest(data):
                        # Apply cleaning to remove repetitions
                        cleaned_page_text = self._clean_ocr_repetitions(page_text)
                        page_texts.append(
                            (cleaned_page_text, batch_start_page + page_num)
                        )
                        full_document_text += cleaned_page_text + " "

                # Process page_texts to create chunks with correct page numbers
                all_raw_chunks = []
                for page_text, page_num in page_texts:
                    chunks = self.text_splitter.split_text(page_text)
                    for idx, chunk_text in enumerate(chunks):
                        all_raw_chunks.append(
                            {
                                "text": chunk_text,
                                "page_num": page_num,
                                "chunk_idx_on_page": idx,
                                "file_name": file_name,
                                "doc_id": doc_id,
                            }
                        )

            except Exception as ocr_error:
                print(
                    f"An error occurred during OCR with '{ocr_engine}': {ocr_error}. Aborting batch."
                )
                return

        else:
            print("Processing with standard PDF parser.")
            parser = PDFParser(self.aclient_openai, self.Server_type, self)
            content_by_page = {}
            async for content, page_num_in_batch in parser.ingest(data):
                absolute_page_num = batch_start_page + page_num_in_batch
                if absolute_page_num not in content_by_page:
                    content_by_page[absolute_page_num] = []
                content_by_page[absolute_page_num].append(content)

            # Add page and paragraph numbering
            numbered_pages = []
            for page_num in sorted(content_by_page.keys()):
                page_blocks = content_by_page[page_num]
                numbered_paragraphs = []
                for para_idx, block in enumerate(page_blocks, 1):
                    numbered_paragraph = (
                        f"[Page {page_num}, Paragraph {para_idx}] {block}"
                    )
                    numbered_paragraphs.append(numbered_paragraph)

                if numbered_paragraphs:
                    numbered_pages.append("\n\n".join(numbered_paragraphs))

            full_document_text = "\n\n\n".join(numbered_pages)
            print(
                f"Successfully parsed and numbered standard PDF file '{file_name}' with {len(numbered_pages)} pages."
            )

        if not full_document_text.strip():
            print(
                f"No text or tables extracted from '{file_name}'. Aborting processing."
            )
            return

        llm_generated_doc_summary = await self._generate_document_summary(
            full_document_text
        )

        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id
        )

        if not all_raw_chunks_with_meta:
            print(
                f"No raw chunks were generated from '{file_name}'. Aborting further processing."
            )
            return

        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]

        print(
            f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from '{file_name}'."
        )

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            raw_chunk_info_item["page_num"] += batch_start_page
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=all_raw_texts,
                    global_idx=i,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params,
                )
            )
            processing_tasks.append(task)

        num_successfully_processed = 0
        for future in asyncio.as_completed(processing_tasks):
            try:
                es_action = await future
                if es_action:
                    yield es_action
                    num_successfully_processed += 1
            except Exception as e:
                print(f"Error processing a chunk future for '{file_name}': {e}")

        print(
            f"Finished processing for '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks."
        )

    async def process_doc(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing DOC: {file_name} (Doc ID: {doc_id})")

        parser = DOCParser(self.aclient_openai, self.Server_type, self)

        full_document_text = ""
        page_breaks = []
        current_position = 0

        try:
            all_parts = []
            async for part in parser.ingest(data):
                all_parts.append(part)
                # Look for page break indicators
                if "\f" in part or "[PAGE BREAK]" in part:
                    page_breaks.append(current_position)
                current_position += len(part)

            full_document_text = " ".join(all_parts)

            # If no page breaks found, try to estimate based on typical page size
            if (
                not page_breaks and len(full_document_text) > 3000
            ):  # If document is substantial
                avg_chars_per_page = 3000  # Estimate chars per page
                for i in range(1, len(full_document_text) // avg_chars_per_page + 1):
                    page_breaks.append(i * avg_chars_per_page)

            print(
                f"Successfully parsed .docx file '{file_name}' with {len(page_breaks) + 1} detected pages."
            )

        except Exception as e:
            print(f"Failed to parse .docx file '{file_name}': {e}")
            return

        if not full_document_text.strip():
            print(
                f"No text extracted from DOCX file '{file_name}'. Aborting processing."
            )
            return

        llm_generated_doc_summary = await self._generate_document_summary(
            full_document_text
        )

        # Pass page breaks to the chunk generation method
        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id, page_breaks
        )

        if not all_raw_chunks_with_meta:
            print(
                f"No raw chunks were generated from DOC file '{file_name}'. Aborting further processing."
            )
            return

        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]

        print(
            f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from DOC file '{file_name}'."
        )

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=all_raw_texts,
                    global_idx=i,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params,
                )
            )
            processing_tasks.append(task)

        num_successfully_processed = 0
        for future in asyncio.as_completed(processing_tasks):
            try:
                es_action = await future
                if es_action:
                    yield es_action
                    num_successfully_processed += 1
            except Exception as e:
                print(
                    f"Error processing a chunk future for DOC file '{file_name}': {e}"
                )

        print(
            f"Finished processing for DOC file '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks."
        )

    async def process_docx(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing DOCX: {file_name} (Doc ID: {doc_id})")

        parser = DOCXParser(self.aclient_openai, self.Server_type, self)

        try:
            # Collect all text parts from the parser, preserving their structure
            all_parts = [part async for part in parser.ingest(data)]

            # Reconstruct the document with clear paragraph and page breaks
            # Use a unique separator to split later
            page_break_marker = "[[--PAGE_BREAK--]]"
            reconstructed_text = ""
            for part in all_parts:
                if "\f" in part or "[PAGE BREAK]" in part:
                    reconstructed_text += page_break_marker
                else:
                    # Each part from the parser is a paragraph or a table.
                    # End it with a double newline to mark it as a distinct block.
                    reconstructed_text += part.strip() + "\n\n"

            # Split the document into pages
            pages_text = reconstructed_text.split(page_break_marker)

            # Add page and paragraph numbering
            numbered_pages = []
            for i, page_content in enumerate(pages_text):
                page_number = i + 1
                # Split page content into paragraphs/blocks based on the double newline
                paragraphs = [
                    p.strip() for p in page_content.split("\n\n") if p.strip()
                ]

                numbered_paragraphs = []
                for para_idx, paragraph in enumerate(paragraphs, 1):
                    # Prepend the page and paragraph info to each block
                    numbered_paragraph = (
                        f"[Page {page_number}, Paragraph {para_idx}] {paragraph}"
                    )
                    numbered_paragraphs.append(numbered_paragraph)

                if numbered_paragraphs:
                    # Join the numbered paragraphs of a page
                    numbered_pages.append("\n\n".join(numbered_paragraphs))

            # Join all processed pages into the final document text
            full_document_text = "\n\n\n".join(numbered_pages)

            print(
                f"Successfully parsed and numbered .docx file '{file_name}' with {len(numbered_pages)} pages."
            )

        except Exception as e:
            print(f"Failed to parse .docx file '{file_name}': {e}")
            traceback.print_exc()
            return

        if not full_document_text.strip():
            print(
                f"No text extracted from DOCX file '{file_name}'. Aborting processing."
            )
            return

        llm_generated_doc_summary = await self._generate_document_summary(
            full_document_text
        )

        # Generate chunks with the numbered text
        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id
        )

        if not all_raw_chunks_with_meta:
            print(
                f"No raw chunks were generated from DOCX file '{file_name}'. Aborting further processing."
            )
            return

        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]

        print(
            f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from DOCX file '{file_name}'."
        )

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=all_raw_texts,
                    global_idx=i,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params,
                )
            )
            processing_tasks.append(task)

        num_successfully_processed = 0
        for future in asyncio.as_completed(processing_tasks):
            try:
                es_action = await future
                if es_action:
                    yield es_action
                    num_successfully_processed += 1
            except Exception as e:
                print(
                    f"Error processing a chunk future for DOCX file '{file_name}': {e}"
                )

        print(
            f"Finished processing for DOCX file '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks."
        )

    async def process_image(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process an image file and generate a text description using vision models.
        Store the description in Elasticsearch with embedding.
        """
        print(f"Processing image: {file_name} (Doc ID: {doc_id})")

        try:
            # Initialize the image parser with the same openai client and server type
            image_parser = ImageParser(self.aclient_openai, self.Server_type, self)

            # Process the image and get its description
            description = ""
            async for text_part in image_parser.ingest(data, filename=file_name):
                description += text_part

            if (
                not description
                or description == "Could not generate description for this image."
            ):
                print(
                    f"No description generated for image '{file_name}'. Using fallback."
                )
                description = f"Image file: {file_name}"

            # Create a chunk with the image description
            chunk_text = f"[Image: {file_name}]\n\n{description}"

            # Generate embedding for the chunk text
            embedding_list = await self._generate_embeddings([chunk_text])
            embedding_vector = (
                embedding_list[0] if embedding_list and embedding_list[0] else []
            )

            if not embedding_vector:
                print(
                    f"Failed to generate embedding for image '{file_name}'. Skipping."
                )
                return

            # Use the image description as its own summary if no user summary provided
            doc_summary = (
                user_provided_document_summary
                or f"Image description: {description[:100]}..."
            )

            # Create the Elasticsearch document
            es_doc_id = f"{doc_id}_img"
            doc_file_name = self.params.get("file_name", file_name)

            # Prepare metadata payload
            metadata_payload = {
                "file_name": doc_file_name,
                "doc_id": doc_id,
                "page_number": 1,  # Images are treated as single page
                "chunk_index_in_page": 0,
                "document_summary": doc_summary,
                "entities": [],  # No entities for images by default
                "relationships": [],  # No relationships for images by default
                "hierarchies": [],  # No hierarchies for images by default
            }

            index_name = self.params.get("index_name")

            # Create the Elasticsearch action
            action = {
                "_index": index_name,
                "_id": es_doc_id,
                "_source": {
                    "chunk_text": chunk_text,
                    "embedding": embedding_vector,
                    "metadata": metadata_payload,
                },
            }

            print(f"Processing complete for image '{file_name}'. ES action prepared.")
            yield action

        except Exception as e:
            print(f"Error processing image '{file_name}': {e}")
            traceback.print_exc()

    async def process_odt(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing ODT: {file_name} (Doc ID: {doc_id})")

        parser = ODTParser(self.aclient_openai, self.Server_type, self)

        full_document_text = ""
        try:
            all_parts = [p async for p in parser.ingest(data)]
            full_document_text = " ".join(all_parts)
            print(
                f"Successfully parsed .odt file '{file_name}'. Total length: {len(full_document_text)} characters."
            )
        except Exception as e:
            print(f"Failed to parse .odt file '{file_name}': {e}")
            return

        if not full_document_text.strip():
            print(
                f"No content extracted from ODT file '{file_name}'. Aborting processing."
            )
            return

        llm_generated_doc_summary = await self._generate_document_summary(
            full_document_text
        )

        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id
        )

        if not all_raw_chunks_with_meta:
            return

        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=all_raw_texts,
                    global_idx=i,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params,
                )
            )
            processing_tasks.append(task)

        for future in asyncio.as_completed(processing_tasks):
            es_action = await future
            if es_action:
                yield es_action

    async def process_txt(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing TXT: {file_name} (Doc ID: {doc_id})")
        try:
            txt_parser = TextParser()
            full_document_text_parts = [
                text_part async for text_part in txt_parser.ingest(data)
            ]
            full_document_text = " ".join(filter(None, full_document_text_parts))

            if not full_document_text.strip():
                print(f"No text extracted from '{file_name}'. Aborting processing.")
                return

            llm_generated_doc_summary = await self._generate_document_summary(
                full_document_text
            )

            all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
                full_document_text, file_name, doc_id
            )

            if not all_raw_chunks_with_meta:
                print(f"No raw chunks were generated from '{file_name}'. Aborting.")
                return

            all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]
            print(
                f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from '{file_name}'."
            )

            processing_tasks = []
            for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
                task = asyncio.create_task(
                    self._process_individual_chunk_pipeline(
                        raw_chunk_info=raw_chunk_info_item,
                        user_provided_doc_summary=user_provided_document_summary,
                        llm_generated_doc_summary=llm_generated_doc_summary,
                        all_raw_texts=all_raw_texts,
                        global_idx=i,
                        file_name=file_name,
                        doc_id=doc_id,
                        params=self.params,
                    )
                )
                processing_tasks.append(task)

            num_successfully_processed = 0
            for future in asyncio.as_completed(processing_tasks):
                try:
                    es_action = await future
                    if es_action:
                        yield es_action
                        num_successfully_processed += 1
                except Exception as e:
                    print(f"Error processing a chunk future for '{file_name}': {e}")

            print(
                f"Finished processing for '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks."
            )

        except Exception as e:
            print(f"Major failure in process_txt for '{file_name}': {e}")

    async def process_csv_semantic_chunking(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Enhanced CSV processing with semantic chunking that preserves context and structure.
        """
        print(f"Processing CSV with semantic chunking: {file_name} (Doc ID: {doc_id})")

        try:
            csv_parser = CSVParser()
            rows = [row_text async for row_text in csv_parser.ingest(data)]

            if not rows:
                print(f"No data extracted from CSV '{file_name}'. Aborting processing.")
                return

            # Parse CSV structure
            header_row = rows[0] if rows else ""
            data_rows = rows[1:] if len(rows) > 1 else []

            # Create semantic chunks
            chunks = await self._create_semantic_csv_chunks(
                header_row, data_rows, file_name
            )

            if not chunks:
                print(f"No chunks generated from CSV '{file_name}'. Aborting.")
                return

            # Process each chunk through the pipeline
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_text = chunk_data["text"]
                chunk_context = chunk_data.get("context", "")

                # Combine chunk with context for enrichment
                full_chunk_text = (
                    f"{chunk_context}\n\n{chunk_text}" if chunk_context else chunk_text
                )

                # Generate document summary
                if chunk_idx == 0:
                    full_document_text = "\n".join(rows)
                    llm_generated_doc_summary = await self._generate_document_summary(
                        full_document_text
                    )

                # Process through individual chunk pipeline
                raw_chunk_info = {
                    "text": full_chunk_text,
                    "page_num": 1,
                    "chunk_idx_on_page": chunk_idx,
                    "file_name": file_name,
                    "doc_id": doc_id,
                }

                es_action = await self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=[chunk["text"] for chunk in chunks],
                    global_idx=chunk_idx,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params,
                )

                if es_action:
                    yield es_action

        except Exception as e:
            print(f"Error in semantic CSV processing for '{file_name}': {e}")

    async def _create_semantic_csv_chunks(
        self, header_row: str, data_rows: List[str], file_name: str
    ) -> List[Dict[str, Any]]:
        """
        Creates semantically meaningful chunks from CSV data.
        """
        chunks = []

        # Strategy 1: Header + batch of rows
        rows_per_chunk = self._calculate_optimal_rows_per_chunk(header_row, data_rows)

        for i in range(0, len(data_rows), rows_per_chunk):
            batch_rows = data_rows[i : i + rows_per_chunk]

            # Create chunk with header context
            chunk_text = f"CSV Structure:\n{header_row}\n\nData:\n" + "\n".join(
                batch_rows
            )

            # Add metadata context
            context = f"This is part {(i // rows_per_chunk) + 1} of CSV file '{file_name}' containing rows {i+1} to {min(i + rows_per_chunk, len(data_rows))}."

            chunks.append(
                {
                    "text": chunk_text,
                    "context": context,
                    "start_row": i + 1,
                    "end_row": min(i + rows_per_chunk, len(data_rows)),
                    "total_rows": len(batch_rows),
                }
            )

        return chunks

    def _calculate_optimal_rows_per_chunk(
        self, header_row: str, data_rows: List[str]
    ) -> int:
        """
        Calculate optimal number of rows per chunk based on token limits.
        """
        if not data_rows:
            return 1

        # Estimate tokens for header and average row
        header_tokens = tiktoken_len(header_row)
        avg_row_tokens = tiktoken_len(data_rows[0]) if data_rows else 1

        # Reserve space for context and formatting
        available_tokens = (
            CHUNK_SIZE_TOKENS - header_tokens - 100
        )  # 100 for context/formatting

        # Calculate how many rows fit
        rows_per_chunk = max(1, available_tokens // avg_row_tokens)

        # Cap at reasonable limits
        return min(rows_per_chunk, 1)  # Max 50 rows per chunk for readability

    async def process_xlsx_semantic_chunking(
        self,
        data: bytes,
        file_name: str,
        doc_id: str,
        user_provided_document_summary: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes XLSX files using a row-by-row, text-based chunking approach.
        """
        print(
            f"Processing XLSX with text-based chunking: {file_name} (Doc ID: {doc_id})"
        )
        try:
            xlsx_parser = XLSXParser()
            # The parser now yields a list of lists of strings
            rows = [row_list async for row_list in xlsx_parser.ingest(data)]

            if not rows:
                print(
                    f"No data extracted from XLSX '{file_name}'. Aborting processing."
                )
                return

            # Create chunks with the corrected mapping logic
            chunks = await self._create_semantic_xlsx_chunks(rows, file_name)

            if not chunks:
                print(f"No chunks generated from XLSX '{file_name}'. Aborting.")
                return

            # CHANGED: Properly join the list of lists into a single string for the summary
            full_document_text = "\n".join([", ".join(row) for row in rows])
            llm_generated_doc_summary = await self._generate_document_summary(
                full_document_text
            )

            # Process each generated chunk through the standard pipeline
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_text = chunk_data["text"]
                raw_chunk_info = {
                    "text": chunk_text,
                    "page_num": 1,  # XLSX is treated as a single "page"
                    "chunk_idx_on_page": chunk_idx,
                    "file_name": file_name,
                    "doc_id": doc_id,
                }
                es_action = await self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=[c["text"] for c in chunks],
                    global_idx=chunk_idx,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params,
                )
                if es_action:
                    yield es_action
        except Exception as e:
            print(f"Error in semantic XLSX processing for '{file_name}': {e}")
            traceback.print_exc()

    async def _create_semantic_xlsx_chunks(
        self, data_rows: List[List[str]], file_name: str
    ) -> List[Dict[str, Any]]:
        """
        Creates semantically meaningful chunks from XLSX data. Processes a batch of rows
        at a time, handling blank values correctly. If a batch's data exceeds the token
        limit, it is split into multiple, connected chunks.
        """
        chunks = []
        if not data_rows or len(data_rows) < 2:
            print(
                f"XLSX file '{file_name}' has no data rows or is missing a header. Skipping."
            )
            return chunks

        # CHANGED: Headers and rows are now lists, no splitting needed.
        headers = [str(h).strip() for h in data_rows[0]]
        content_rows = data_rows[1:]

        format_description = "This chunk contains data from a row where each value is mapped to its column header in a 'Header : Value' format."

        rows_per_chunk = self._calculate_optimal_xlsx_rows_per_chunk(data_rows)
        print(f"Processing XLSX in batches of {rows_per_chunk} rows.")

        for i in range(0, len(content_rows), rows_per_chunk):
            batch_of_rows = content_rows[i : i + rows_per_chunk]
            start_row = i + 1
            end_row = i + len(batch_of_rows)

            all_batch_items = []
            for row_values in batch_of_rows:
                # Pad/truncate values to match header count, ensuring alignment
                num_headers = len(headers)
                row_values.extend([""] * (num_headers - len(row_values)))
                row_values = row_values[:num_headers]

                if all_batch_items:
                    all_batch_items.append(("---", "New Row ---"))
                all_batch_items.extend(list(zip(headers, row_values)))

            if not all_batch_items:
                continue

            current_chunk_items = []
            is_first_chunk_for_batch = True

            for header, value in all_batch_items:
                if not header:  # Skip empty headers from trailing commas
                    continue

                item_str = (
                    f"--- {value} ---"
                    if header == "---"
                    else f"{header} : {value.strip() if value else 'N/A'}"
                )

                prospective_items = current_chunk_items + [item_str]

                if is_first_chunk_for_batch:
                    context = f"This is part {start_row} of XLSX file '{file_name}' containing rows {start_row} to {end_row}."
                else:
                    context = f"Continuation of data from rows {start_row} to {end_row} in XLSX file '{file_name}'."

                row_data_content = ", ".join(prospective_items)
                prospective_text = (
                    f"{context}\n\n{format_description}\n\nRow Data: {row_data_content}"
                )

                if (
                    tiktoken_len(prospective_text) > CHUNK_SIZE_TOKENS
                    and current_chunk_items
                ):
                    final_context = (
                        f"This is part {start_row} of XLSX file '{file_name}' containing rows {start_row} to {end_row}."
                        if is_first_chunk_for_batch
                        else f"Continuation of data from rows {start_row} to {end_row} in XLSX file '{file_name}'."
                    )
                    final_row_data = ", ".join(current_chunk_items)
                    chunks.append(
                        {
                            "text": f"{final_context}\n\n{format_description}\n\nRow Data: {final_row_data}"
                        }
                    )

                    current_chunk_items = [item_str]
                    is_first_chunk_for_batch = False
                else:
                    current_chunk_items = prospective_items

            if current_chunk_items:
                final_context = (
                    f"This is part {start_row} of XLSX file '{file_name}' containing rows {start_row} to {end_row}."
                    if is_first_chunk_for_batch
                    else f"Continuation of data from rows {start_row} to {end_row} in XLSX file '{file_name}'."
                )
                final_row_data = ", ".join(current_chunk_items)
                chunks.append(
                    {
                        "text": f"{final_context}\n\n{format_description}\n\nRow Data: {final_row_data}"
                    }
                )

        print(
            f"Created {len(chunks)} text-based chunks from {len(content_rows)} rows in '{file_name}'."
        )
        return chunks

    def _calculate_optimal_xlsx_rows_per_chunk(self, data_rows: List[List[str]]) -> int:
        """
        Calculate optimal number of rows per chunk for XLSX based on token limits.
        """
        if not data_rows:
            return 1

        # CORRECTED: Join the list of header strings into a single string before tokenizing.
        header_string = ", ".join(data_rows[0])
        avg_row_tokens = tiktoken_len(header_string) if header_string else 1

        available_tokens = (
            CHUNK_SIZE_TOKENS - 100
        )  # Reserve space for context/formatting
        rows_per_chunk = max(
            1, available_tokens // (avg_row_tokens if avg_row_tokens > 0 else 1)
        )

        return min(rows_per_chunk, 1)  # Max rows per chunk for readability


async def ensure_es_index_exists(client: Any, index_name: str, mappings_body: Dict):
    try:
        if not await client.indices.exists(index=index_name):
            is_army_mode = os.getenv("SERVER_TYPE") == "ARMY"
            updated_mappings = copy.deepcopy(mappings_body)
            if is_army_mode:
                updated_mappings["mappings"]["properties"]["embedding"]["dims"] = 1024
                if (
                    "description_embedding"
                    in updated_mappings["mappings"]["properties"]["metadata"][
                        "properties"
                    ]["entities"]["properties"]
                ):
                    updated_mappings["mappings"]["properties"]["metadata"][
                        "properties"
                    ]["entities"]["properties"]["description_embedding"]["dims"] = 1024
            else:
                updated_mappings["mappings"]["properties"]["embedding"][
                    "dims"
                ] = OPENAI_EMBEDDING_DIMENSIONS
                if (
                    "description_embedding"
                    in updated_mappings["mappings"]["properties"]["metadata"][
                        "properties"
                    ]["entities"]["properties"]
                ):
                    updated_mappings["mappings"]["properties"]["metadata"][
                        "properties"
                    ]["entities"]["properties"]["description_embedding"][
                        "dims"
                    ] = OPENAI_EMBEDDING_DIMENSIONS

            await client.indices.create(index=index_name, body=updated_mappings)
            print(
                f"Elasticsearch index '{index_name}' created with specified mappings."
            )
            return True
        else:
            print("Index already exists in ensure_es_index_exists")

            # Get current mapping to check compatibility
            current_mapping_response = await client.indices.get_mapping(
                index=index_name
            )
            current_mapping = (
                current_mapping_response.get(index_name, {})
                .get("mappings", {})
                .get("properties", {})
            )
            current_metadata = current_mapping.get("metadata", {}).get("properties", {})
            current_entities = current_metadata.get("entities", {})

            hierarchies_exists = "hierarchies" in current_metadata

            # Check if entities field exists and its type
            entities_exists = "entities" in current_metadata
            entities_is_nested = current_entities.get("type") == "nested"

            # Check if description_embedding exists in entities properties
            has_description_embedding = False
            if entities_is_nested and "properties" in current_entities:
                has_description_embedding = (
                    "description_embedding" in current_entities.get("properties", {})
                )

            # If entities field doesn't exist or is not nested, we need to handle this
            if entities_exists and not entities_is_nested:
                print(
                    f"⚠️ Warning: Index '{index_name}' has entities field as non-nested, but we need it to be nested."
                )
                print(
                    "This requires recreating the index or creating a new one with a different name."
                )
                print("Options:")
                print(
                    "1. Delete the existing index and recreate it (will lose all data)"
                )
                print("2. Use a different index name")
                print("3. Continue with limited functionality (no entity embeddings)")

                # For now, we'll continue but warn about limited functionality
                print(
                    "Continuing with existing mapping - entity embeddings will not be available."
                )
                return True

            # If entities field doesn't exist at all, add it as nested
            if not hierarchies_exists:
                print(f"Adding hierarchies field as nested to index '{index_name}'")

                try:
                    update_mapping = {
                        "properties": {
                            "metadata": {
                                "properties": {
                                    "hierarchies": {
                                        "type": "nested",
                                        "properties": {
                                            "name": {"type": "keyword"},
                                            "description": {"type": "text"},
                                            "root_type": {"type": "keyword"},
                                            "levels": {
                                                "type": "nested",
                                                "properties": {
                                                    "id": {"type": "keyword"},
                                                    "name": {"type": "keyword"},
                                                    "nodes": {"type": "nested"},
                                                },
                                            },
                                            "relationships": {"type": "nested"},
                                        },
                                    }
                                }
                            }
                        }
                    }

                    await client.indices.put_mapping(
                        index=index_name, body=update_mapping
                    )
                    print(
                        f"Successfully added nested entities field to index '{index_name}'"
                    )
                    return True

                except Exception as e:
                    print(
                        f"Failed to add entities mapping to index '{index_name}': {e}"
                    )
                    return False

            # If entities is nested but missing description_embedding, add it
            if entities_is_nested and not has_description_embedding:
                print(
                    f"Adding description_embedding field to nested entities in index '{index_name}'"
                )

                dims = (
                    1024
                    if os.getenv("SERVER_TYPE") == "ARMY"
                    else OPENAI_EMBEDDING_DIMENSIONS
                )

                try:
                    update_mapping = {
                        "properties": {
                            "metadata": {
                                "properties": {
                                    "entities": {
                                        "type": "nested",
                                        "properties": {
                                            "description_embedding": {
                                                "type": "dense_vector",
                                                "dims": dims,
                                                "index": True,
                                                "similarity": "cosine",
                                            }
                                        },
                                    }
                                }
                            }
                        }
                    }

                    await client.indices.put_mapping(
                        index=index_name, body=update_mapping
                    )
                    print(
                        f"Successfully added description_embedding to entities in index '{index_name}'"
                    )
                    return True

                except Exception as e:
                    print(
                        f"Failed to update entities mapping in index '{index_name}': {e}"
                    )
                    return False

            # Check for other missing top-level fields
            expected_top_level_props = mappings_body.get("mappings", {}).get(
                "properties", {}
            )
            missing_fields = []

            for field, expected_field_mapping in expected_top_level_props.items():
                if field == "metadata":
                    continue  # Already handled above
                if field not in current_mapping:
                    missing_fields.append(field)

            if missing_fields:
                print(f"Adding missing top-level fields: {missing_fields}")
                update_body = {
                    field: expected_top_level_props[field] for field in missing_fields
                }

                # Adjust embedding dimensions if needed
                if "embedding" in update_body:
                    dims = (
                        1024
                        if os.getenv("SERVER_TYPE") == "ARMY"
                        else OPENAI_EMBEDDING_DIMENSIONS
                    )
                    update_body["embedding"]["dims"] = dims

                try:
                    await client.indices.put_mapping(
                        index=index_name, body={"properties": update_body}
                    )
                    print(f"Successfully added missing fields to index '{index_name}'")
                except Exception as e:
                    print(f"Failed to add missing fields to index '{index_name}': {e}")

            print(f"Index '{index_name}' mapping verification completed.")
            return True

    except Exception as e:
        print(f"❌ Error with Elasticsearch index '{index_name}': {e}")
        traceback.print_exc()
        return False


async def example_run_file_processing(
    file_data: str | bytes,
    original_file_name: str,
    document_id: str,
    user_provided_doc_summary: str,
    es_client: Any,
    aclient_openai: Any,
    params: Any,
    config: Any,
):

    is_army_mode = os.getenv("SERVER_TYPE") == "ARMY"

    if not is_army_mode and not OPENAI_API_KEY:
        print(
            "OPENAI_API_KEY is not set. LLM-dependent operations (summary, KG, enrichment, embeddings) will be skipped or use placeholders."
        )

    index_name = params.get("index_name")
    print("elastic search index_name:--", index_name)

    expected_mappings = copy.deepcopy(CHUNKED_PDF_MAPPINGS)

    if is_army_mode:
        print(
            "ARMY mode is active. Setting embedding dimensions to 1024 for the index."
        )
        dims = 1024
    else:
        print(
            "Standard mode is active. Setting embedding dimensions to 3072 for the index."
        )
        dims = OPENAI_EMBEDDING_DIMENSIONS

    expected_mappings["mappings"]["properties"]["embedding"]["dims"] = dims
    if (
        "description_embedding"
        in expected_mappings["mappings"]["properties"]["metadata"]["properties"][
            "entities"
        ]["properties"]
    ):
        expected_mappings["mappings"]["properties"]["metadata"]["properties"][
            "entities"
        ]["properties"]["description_embedding"]["dims"] = dims

    print(
        f"Mode: {'ARMY' if is_army_mode else 'Standard'}, Embedding Dimensions: {dims}"
    )

    if not await ensure_es_index_exists(es_client, index_name, CHUNKED_PDF_MAPPINGS):
        print(
            f"Failed to ensure Elasticsearch index '{index_name}' exists or is compatible. Aborting."
        )
        return

    file_extension = os.path.splitext(original_file_name)[1].lower()
    processor = ChunkingEmbeddingPDFProcessor(
        params, config, aclient_openai, file_extension
    )
    actions_for_es = []

    file_extension = os.path.splitext(original_file_name)[1].lower()
    print(
        f"\n--- Starting Processing for: {original_file_name} (Doc ID: {file_extension}) ---"
    )

    try:
        doc_iterator = None
        if file_extension == ".pdf":
            doc_iterator = processor.process_pdf(
                file_data, original_file_name, document_id, user_provided_doc_summary
            )
        elif file_extension == ".doc":
            doc_iterator = processor.process_doc(
                file_data, original_file_name, document_id, user_provided_doc_summary
            )
        elif file_extension == ".docx":
            doc_iterator = processor.process_docx(
                file_data, original_file_name, document_id, user_provided_doc_summary
            )
        if file_extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".heic",
            ".tiff",
            ".tif",
        ]:
            doc_iterator = processor.process_image(
                file_data, original_file_name, document_id, user_provided_doc_summary
            )
        elif file_extension == ".pdf":
            doc_iterator = processor.process_pdf(
                file_data, original_file_name, document_id, user_provided_doc_summary
            )
        elif file_extension == ".odt":
            doc_iterator = processor.process_odt(
                file_data, original_file_name, document_id, user_provided_doc_summary
            )
        elif file_extension == ".txt":
            doc_iterator = processor.process_txt(
                file_data, original_file_name, document_id, user_provided_doc_summary
            )
        elif file_extension == ".csv":
            doc_iterator = processor.process_csv_semantic_chunking(
                file_data, original_file_name, document_id, user_provided_doc_summary
            )
        elif file_extension == ".xlsx":
            doc_iterator = processor.process_xlsx_semantic_chunking(
                file_data, original_file_name, document_id, user_provided_doc_summary
            )
        else:
            print(
                f"Unsupported file type: '{file_extension}'. Supported types: .pdf, .doc, .docx, .csv, .txt, .jpg, .jpeg, .png, .gif, .bmp, .webp, .heic, .tiff, .tif"
            )
            return None

        if doc_iterator:
            async for action in doc_iterator:
                if action:
                    actions_for_es.append(action)

        if actions_for_es:
            print(
                f"Collected {len(actions_for_es)} actions for bulk ingestion into '{index_name}'."
            )

            if actions_for_es:
                print(
                    "Sample document to be indexed (first one, embedding vector omitted if long):"
                )
                sample_action_copy = copy.deepcopy(actions_for_es[0])
                if (
                    "_source" in sample_action_copy
                    and "embedding" in sample_action_copy["_source"]
                ):
                    embedding_val = sample_action_copy["_source"]["embedding"]
                    if isinstance(embedding_val, list) and embedding_val:
                        sample_action_copy["_source"][
                            "embedding"
                        ] = f"<embedding_vector_dim_{len(embedding_val)}>"
                    elif not embedding_val:
                        sample_action_copy["_source"][
                            "embedding"
                        ] = "<empty_embedding_vector>"
                    else:
                        sample_action_copy["_source"][
                            "embedding"
                        ] = f"<embedding_vector_unexpected_format: {type(embedding_val).__name__}>"

            errors = []
            try:
                successes, response = await async_bulk(
                    es_client, actions_for_es, raise_on_error=False
                )
                print(f"Elasticsearch bulk ingestion: {successes} successes.")

                failed = [r for r in response if not r[0]]
                if failed:
                    print(
                        f"{len(failed)} document(s) failed to index. Showing first error:"
                    )
                    errors = failed

            except BulkIndexError as e:
                errors = e.errors
                print("BulkIndexError occurred:")
                print(json.dumps(e.errors, indent=2, default=str))

            if errors:
                print(f"Elasticsearch bulk ingestion errors ({len(errors)}):")
                for i, err_info in enumerate(errors):
                    error_item = err_info.get(
                        "index",
                        err_info.get(
                            "create", err_info.get("update", err_info.get("delete", {}))
                        ),
                    )
                    status = error_item.get("status", "N/A")
                    error_details = error_item.get("error", {})
                    error_type = error_details.get("type", "N/A")
                    error_reason = error_details.get("reason", "N/A")
                    doc_id_errored = error_item.get("_id", "N/A")
                    print(
                        f"Error {i+1}: Doc ID '{doc_id_errored}', Status {status}, Type '{error_type}', Reason: {error_reason}"
                    )
        else:
            print(
                f"No chunks generated or processed for ingestion from '{original_file_name}'."
            )

    except Exception as e:
        print(
            f"An error occurred during the example run for '{original_file_name}': {e}"
        )
    print(f"--- Finished PDF Processing for: {original_file_name} ---\n")


def _generate_doc_id_from_content(content_bytes: bytes) -> str:
    """Generates a SHA256 hash for the given byte content."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content_bytes)
    return sha256_hash.hexdigest()


def parse_s3_path(s3_path):
    parsed_url = urlparse(s3_path)
    bucket_name = parsed_url.netloc.split(".")[0]
    file_key = parsed_url.path[1:]
    return bucket_name, file_key


def read_file(file_content, file_key):
    print("here", file_key)
    file_parts = file_key.split("/")
    filename = file_parts[-1]
    random_number = random.randint(1000, 9999)
    folder_path = f"./data/{random_number}"
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, filename)

    try:
        with open(file_path, "wb") as f:
            f.write(file_content)

        if filename.lower().endswith(".pdf"):
            text = ""
            # This part seems incorrect for the class structure, PDFParser needs bytes
            # reader = PDFParser(file_path)
            # for page in reader.pages:
            #     text += page.extract_text() or ""
            document = {"filename": filename, "text": "DEPRECATED_LOGIC"}
        else:
            return {"error": f"Unsupported file format: {filename}"}

    except Exception as e:
        return {"error": f"Error reading file: {str(e)}"}

    finally:
        try:
            shutil.rmtree(folder_path)
        except Exception as cleanup_error:
            print("Cleanup error:", cleanup_error)

    return document, filename


async def handle_request(data: Message) -> FunctionResponse:
    es_client = None
    aclient_openai = None

    try:
        print("Incoming Data:--", data)
        params = data.params
        config = data.config
        es_client = await check_async_elasticsearch_connection()
        if not es_client:
            return FunctionResponse(False, "Could not connect to Elasticsearch.")

        aclient_openai = init_async_openai_client()
        if not aclient_openai:
            return FunctionResponse(False, "Could not connect to Open Ai.")

        doc_path_input = params.get("file_path")
        doc_path = Path(doc_path_input)
        original_file_name = doc_path.name
        print("doc_path:----", doc_path)

        try:
            with open(doc_path, "rb") as f:
                doc_bytes_data = f.read()
                print(f"Successfully read PDF file: {doc_path}")
        except Exception as e:
            print(f"Failed to read PDF file '{doc_path}': {e}")
            return

        generated_doc_id = _generate_doc_id_from_content(doc_bytes_data)
        print(
            f"Generated Document ID (SHA256 of content) for '{original_file_name}': {generated_doc_id}"
        )

        user_provided_summary_input = (
            params.get("description") or f"Content from {original_file_name}"
        )

        await example_run_file_processing(
            file_data=doc_bytes_data,
            original_file_name=original_file_name,
            document_id=generated_doc_id,
            user_provided_doc_summary=user_provided_summary_input,
            es_client=es_client,
            aclient_openai=aclient_openai,
            params=params,
            config=config,
        )

        if es_client:
            await es_client.close()
            print("Elasticsearch client closed.")
        if aclient_openai and hasattr(aclient_openai, "close"):
            try:
                await aclient_openai.close()
                print("OpenAI client closed.")
            except Exception as e:
                print(f"Error closing OpenAI client: {e}")
        elif aclient_openai and hasattr(aclient_openai, "aclose"):
            try:
                await aclient_openai.aclose()
                print("OpenAI client closed.")
            except Exception as e:
                print(f"Error closing OpenAI client: {e}")

        print("Rag unstructured file successfully uploaded")
        return FunctionResponse(message=Messages("success"))
    except Exception as e:
        print(f"❌ Error during indexing: {e}")
        return FunctionResponse(message=Messages(e))


def test():
    params = {
        "index_name": "messbill-rowblaze",
        "file_name": "GbhMAYMessBill.pdf",
        "file_path": "/Users/hari/Downloads/GbhMAYMessBill.pdf",
        "description": "messbill",
        "is_ocr_pdf": False,
    }

    config = {
        "api_key": os.getenv("OPEN_AI_KEY"),
    }
    message = Message(params=params, config=config)
    res = asyncio.run(handle_request(message))
    # print(res)


if __name__ == "__main__":
    test()
