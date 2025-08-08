import os
import asyncio
from io import BytesIO
from typing import AsyncGenerator, Dict, Any, List, Tuple
import yaml 
from pathlib import Path 
import copy 
import xml.etree.ElementTree as ET 
import re
import hashlib 
import json
import shutil
import random
import traceback
import requests
import time
import base64
import fitz
import httpx
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

from typing import Optional
from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk, BulkIndexError
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from openai import AsyncOpenAI
import pdfplumber
from urllib.parse import urlparse
from datetime import datetime, timezone
from sdk.response import FunctionResponse, Messages
from sdk.message import Message

from src.core.base.parsers.csv_parser import CSVParser
from src.core.base.parsers.xlsx_parser import XLSXParser

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

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4.1-nano")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
OPENAI_EMBEDDING_DIMENSIONS = 3072

CHUNK_SIZE_TOKENS = 20000
CHUNK_OVERLAP_TOKENS =0
FORWARD_CHUNKS = 3
BACKWARD_CHUNKS = 3
CHARS_PER_TOKEN_ESTIMATE = 4 
SUMMARY_MAX_TOKENS = 1024
ELASTICSEARCH_URL = os.getenv("RAG_UPLOAD_ELASTIC_URL")                    
ELASTICSEARCH_API_KEY = os.getenv("ELASTICSEARCH_API_KEY")

def init_async_openai_client() -> Optional[AsyncOpenAI]:
  openai_api_key = OPENAI_API_KEY
  print('Open ai key:-', openai_api_key)
  if not openai_api_key:
    print("❌ OPENAI_API_KEY not found in .env. OpenAI client will not be functional.")
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
        print('Connecting to Elsatic client ELASTICSEARCH_URL:-', ELASTICSEARCH_URL)
        es_client = None
        os.getenv("ELASTICSEARCH_API_KEY")
        es_client = AsyncElasticsearch(
            ELASTICSEARCH_URL,
            api_key=ELASTICSEARCH_API_KEY,
            request_timeout=60,
            retry_on_timeout=True
        )
        
        if not await es_client.ping():
            print("❌ Ping to Elasticsearch cluster failed. URL may be incorrect or server is down.")
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
                "similarity": "cosine"
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
                            "description_embedding":{
                                "type": "dense_vector",
                                "dims": 3072,
                                "index": True,
                                "similarity": "cosine"
                            }
                        }
                    },
                    "relationships": {
                        "type": "nested",
                        "properties": {
                            "source_entity": {"type": "keyword"},
                            "target_entity": {"type": "keyword"},
                            "relation": {"type": "keyword"},
                            "relationship_description": {"type": "text"},
                            "relationship_weight": {"type": "float"}
                        }
                    }
                }
            }
        }
    }
}

class PDFParser:
    """A parser for extracting tables and text from PDF files using pdfplumber."""
        
    def __init__(self, aclient_openai: Optional[AsyncOpenAI], processor_ref: Optional[Any] = None):
        self.aclient_openai = aclient_openai
        self.processor_ref = processor_ref
        self.vision_prompt_text = self._load_vision_prompt()
    
    def _load_vision_prompt(self) -> str:
        try:
            prompt_file_path = PROMPTS_DIR / "vision_img.yaml"
            with open(prompt_file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
            
            if prompt_data and "vision_img" in prompt_data and "template" in prompt_data["vision_img"]:
                template_content = prompt_data["vision_img"]["template"]
                print("Successfully loaded vision prompt template.")
                return template_content
            else:
                print(f"Vision prompt template not found or invalid in {prompt_file_path}.")
                return "Describe the image in detail."
        except Exception as e:
            print(f"Error loading vision prompt: {e}")
            return "Describe the image in detail."
    
    async def _get_image_description(self, image_bytes: bytes) -> str:
        
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        media_type = "image/png" #PyMuPDF pixmaps default to png
        
        if not self.aclient_openai:
            print("OpenAI client not available, skipping image description.")
            return ""
                
        if not self.processor_ref:
            print("Processor reference not available for OpenAI call. Skipping image description.")
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
                        "image_url": { "url": f"data:{media_type};base64,{image_data}" },
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


    async def ingest(self, data: bytes) -> AsyncGenerator[str, None]:
        """
        Ingest PDF data and yield a stream of content (text and markdown tables).
        """
        if not isinstance(data, bytes):
            raise TypeError("PDF data must be in bytes format.")

        pdf_stream = BytesIO(data)
        try:
            with pdfplumber.open(pdf_stream) as pdf_plumber:
                with fitz.open(stream=data, filetype="pdf") as pdf_fitz:
                    if len(pdf_plumber.pages) != len(pdf_fitz):
                        print("Warning: Page count mismatch between pdfplumber and PyMuPDF.")
                    
                    for page_num, p_page in enumerate(pdf_plumber.pages, 1):
                        # 1. Extract text from the page
                        page_text = p_page.extract_text()
                        if page_text:
                            yield page_text.strip()
                        
                        # 2. Extract tables from the same page
                        tables = p_page.extract_tables()
                        for table in tables:
                            table_markdown = self._convert_table_to_markdown(table)
                            yield table_markdown
                        
                        # 3. Extract images using PyMuPDF and get descriptions
                        if page_num <= len(pdf_fitz):
                            fitz_page = pdf_fitz.load_page(page_num - 1)
                            image_list = fitz_page.get_images(full=True)
                            if image_list:
                                print(f"Found {len(image_list)} images on page {page_num}.")
                                for img_info in image_list:
                                    xref = img_info[0]
                                    base_image = pdf_fitz.extract_image(xref)
                                    image_bytes = base_image["image"]
                                    
                                    description = await self._get_image_description(image_bytes)
                                    if description:
                                        yield description
                        else:
                            print(f"Skipping image extraction for page {page_num} due to page count mismatch.")
                    
        except Exception as e:
            print(f"Failed to process PDF: {e}")
            raise ValueError(f"Error processing PDF file: {str(e)}") from e

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
                    page_text = await asyncio.to_thread(pytesseract.image_to_string, image)
                    if not page_text or not page_text.strip():
                        print(f"OCR found no text on page {page_num}.")
                    yield page_text
                except pytesseract.TesseractNotFoundError:
                    print("Tesseract executable not found. Please install Tesseract-OCR and ensure it's in your system's PATH.")
                    raise
                except Exception as ocr_err:
                    print(f"Error during OCR on page {page_num}: {ocr_err}")
                    yield ""
        except Exception as e:
            print(f"Failed to convert PDF to images for OCR: {e}")
            raise ValueError(f"Error processing PDF for OCR: {str(e)}") from e
        finally:
            pdf_stream.close()

class ChunkingEmbeddingPDFProcessor:
    def __init__(self, params: Any, config: Any, aclient_openai: Optional[AsyncOpenAI], file_extension: str):
        self.params = params
        self.config = config
        self.aclient_openai = aclient_openai
        self.embedding_model = None
        self.embedding_dims = OPENAI_EMBEDDING_DIMENSIONS
        
        if file_extension in [".docx", ".doc", ".odt", ".txt"]:
            chunk_size = 1024
            chunk_overlap = 512
            print(f"Using document-specific chunking: size={chunk_size}, overlap={chunk_overlap}")
        elif file_extension in [".csv", ".xlsx"]:
            chunk_size = 2000000
            chunk_overlap = 0
            print(f"Using spreadsheet-specific chunking: size={chunk_size}, overlap={chunk_overlap}")
        else:
            chunk_size = CHUNK_SIZE_TOKENS
            chunk_overlap = CHUNK_OVERLAP_TOKENS
            print(f"Using default chunking: size={chunk_size}, overlap={chunk_overlap}")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=chunk_size,
          chunk_overlap=chunk_overlap,
          length_function=tiktoken_len,
          separators=["\n|", "\n", "|", ". "," ", ""],
        )
        self.enrich_prompt_template = self._load_prompt_template("chunk_enrichment")
        self.graph_extraction_prompt_template = self._load_prompt_template("graph_extraction")
        self.summary_prompt_template = self._load_prompt_template("summary") 

    def _load_prompt_template(self, prompt_name: str) -> str:
        try:
            prompt_file_path = PROMPTS_DIR / f"{prompt_name}.yaml"
            print('prompt_file_path:----', prompt_file_path)
            with open(prompt_file_path, 'r') as f:
                prompt_data = yaml.safe_load(f)
            
            if prompt_data and prompt_name in prompt_data and "template" in prompt_data[prompt_name]:
                template_content = prompt_data[prompt_name]["template"]
                print(f"Successfully loaded prompt template for '{prompt_name}'.")
                return template_content
            else:
                print(f"Prompt template for '{prompt_name}' not found or invalid in {prompt_file_path}.")
                raise ValueError(f"Invalid prompt structure for {prompt_name}")
        except FileNotFoundError:
            print(f"Prompt file not found: {prompt_file_path}")
            raise
        except Exception as e:
            print(f"Error loading prompt '{prompt_name}': {e}")
            raise
    
    async def _call_openai_api(
        self,
        model_name: str,
        payload_messages: List[Dict[str, Any]],
        is_vision_call: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.1
    ) -> str:
        """A unified async method to call OpenAI text and vision models with retry logic."""
        if not self.aclient_openai:
            print("OpenAI client not configured. Cannot make API call.")
            return ""

        max_retries = 5
        base_delay_seconds = 3

        for attempt in range(max_retries):
            try:
                start_time = datetime.now(timezone.utc)
                
                response = await self.aclient_openai.chat.completions.create(
                    model=model_name,
                    messages=payload_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                
                content = response.choices[0].message.content

                if content:
                    print(f"OpenAI API call successful. Preview: {content[:100].strip()}...")
                    return content
                else:
                    print(f"OpenAI API returned empty content. Attempt {attempt + 1}/{max_retries}")

            except Exception as e:
                print(f"OpenAI API call failed (Attempt {attempt + 1}/{max_retries}): {e}")
            
            if attempt + 1 < max_retries:
                delay = base_delay_seconds * (2 ** attempt)
                print(f"Waiting for {delay} seconds before retrying...")
                await asyncio.sleep(delay)

        print("Max retries reached for OpenAI API call. Returning empty string.")
        return ""

    def _clean_xml_string(self, xml_string: str) -> str:
        """Cleans the XML string from common LLM artifacts and prepares it for parsing."""
        if not isinstance(xml_string, str):
            print(f"XML input is not a string, type: {type(xml_string)}. Returning empty string.")
            return ""

        cleaned_xml = xml_string.strip()

        if cleaned_xml.startswith("```xml"):
            cleaned_xml = cleaned_xml[len("```xml"):].strip()
        elif cleaned_xml.startswith("```"):
            cleaned_xml = cleaned_xml[len("```"):].strip()
        
        if cleaned_xml.endswith("```"):
            cleaned_xml = cleaned_xml[:-len("```")].strip()

        if cleaned_xml.startswith("<?xml"):
            end_decl = cleaned_xml.find("?>")
            if end_decl != -1:
                cleaned_xml = cleaned_xml[end_decl + 2:].lstrip()
        
        first_angle_bracket = cleaned_xml.find("<")
        last_angle_bracket = cleaned_xml.rfind(">")

        if first_angle_bracket != -1 and last_angle_bracket != -1 and last_angle_bracket > first_angle_bracket:
            cleaned_xml = cleaned_xml[first_angle_bracket : last_angle_bracket + 1]
        elif first_angle_bracket == -1 :
            print(f"No XML tags found in the string after initial cleaning. Original: {xml_string[:200]}")
            return ""


        cleaned_xml = re.sub(r'&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)', '&amp;', cleaned_xml)

        common_prefixes = ["Sure, here is the XML:", "Here's the XML output:", "Okay, here's the XML:"]
        for prefix in common_prefixes:
            if cleaned_xml.lower().startswith(prefix.lower()):
                cleaned_xml = cleaned_xml[len(prefix):].lstrip()
                break
        
        cleaned_xml = re.sub(r'[^\x09\x0A\x0D\x20-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]', '', cleaned_xml)

        return cleaned_xml

    def _parse_graph_xml(self, xml_string: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
            if (cleaned_xml.startswith("<entity") or cleaned_xml.startswith("<relationship")) and \
               (cleaned_xml.endswith("</entity>") or cleaned_xml.endswith("</relationship>")):
                string_to_parse = f"<root_wrapper>{cleaned_xml}</root_wrapper>"
                print("Wrapping multiple top-level entity/relationship tags with <root_wrapper>.")
            elif not (cleaned_xml.count("<") > 1 and cleaned_xml.count(">") > 1 and cleaned_xml.find("</") > 0 and cleaned_xml.startswith("<") and cleaned_xml.endswith(">") and cleaned_xml[1:cleaned_xml.find(">") if cleaned_xml.find(">") > 1 else 0].strip() == cleaned_xml[cleaned_xml.rfind("</")+2:-1].strip() ):
                 string_to_parse = f"<root_wrapper>{cleaned_xml}</root_wrapper>"
                 print("Wrapping content with <root_wrapper> as it doesn't appear to have a single root or matching end tag.")
        
        try:
            root = ET.fromstring(string_to_parse)
            
            for entity_elem in root.findall(".//entity"): 
                name_val = entity_elem.get("name")
                if not name_val:
                    name_elem = entity_elem.find("name")
                    name_val = name_elem.text.strip() if name_elem is not None and name_elem.text else None
                
                ent_type_elem = entity_elem.find("type")
                ent_desc_elem = entity_elem.find("description")
                
                ent_type = ent_type_elem.text.strip() if ent_type_elem is not None and ent_type_elem.text else "Unknown"
                ent_desc = ent_desc_elem.text.strip() if ent_desc_elem is not None and ent_desc_elem.text else ""
                
                if name_val:
                    entities.append({"name": name_val.strip(), "type": ent_type, "description": ent_desc})

            for rel_elem in root.findall(".//relationship"): 
                source_elem = rel_elem.find("source")
                target_elem = rel_elem.find("target")
                rel_type_elem = rel_elem.find("type")
                rel_desc_elem = rel_elem.find("description")
                rel_weight_elem = rel_elem.find("weight")
                
                source = source_elem.text.strip() if source_elem is not None and source_elem.text else None
                target = target_elem.text.strip() if target_elem is not None and target_elem.text else None
                rel_type = rel_type_elem.text.strip() if rel_type_elem is not None and rel_type_elem.text else "RELATED_TO"
                rel_desc = rel_desc_elem.text.strip() if rel_desc_elem is not None and rel_desc_elem.text else ""
                weight = None
                if rel_weight_elem is not None and rel_weight_elem.text:
                    try:
                        weight = float(rel_weight_elem.text.strip())
                    except ValueError:
                        print(f"Could not parse relationship weight '{rel_weight_elem.text}' as float.")
                
                if source and target:
                    relationships.append({
                        "source_entity": source, "target_entity": target, "relation": rel_type,
                        "relationship_description": rel_desc, "relationship_weight": weight
                    })
            
            print(f"Successfully parsed {len(entities)} entities and {len(relationships)} relationships using ET.fromstring.")

        except ET.ParseError as e:
            err_line, err_col = e.position if hasattr(e, 'position') else (-1, -1)
            log_message = (
                f"XML parsing error with ET.fromstring: {e}\n"
                f"Error at line {err_line}, column {err_col} (approximate). Trying regex-based extraction as fallback.\n"
                f"Cleaned XML snippet attempted (first 1000 chars):\n{string_to_parse[:1000]}"
            )
            print(log_message)
            
            entities = [] 
            relationships = [] 

            entity_pattern_attr = r'<entity\s+name\s*=\s*"([^"]*)"\s*>\s*(?:<type>([^<]*)</type>)?\s*(?:<description>([^<]*)</description>)?\s*</entity>'
            entity_pattern_tag = r'<entity>\s*<name>([^<]+)</name>\s*(?:<type>([^<]*)</type>)?\s*(?:<description>([^<]*)</description>)?\s*</entity>'


            for pattern in [entity_pattern_attr, entity_pattern_tag]:
                for match in re.finditer(pattern, string_to_parse): 
                    name, entity_type, description = match.groups()
                    if name:
                        entities.append({
                            "name": name.strip(),
                            "type": entity_type.strip() if entity_type and entity_type.strip() else "Unknown",
                            "description": description.strip() if description and description.strip() else ""
                        })
            
            rel_pattern = r'<relationship>\s*(?:<source>([^<]+)</source>)?\s*(?:<target>([^<]+)</target>)?\s*(?:<type>([^<]*)</type>)?\s*(?:<description>([^<]*)</description>)?\s*(?:<weight>([^<]*)</weight>)?\s*</relationship>'
            for match in re.finditer(rel_pattern, string_to_parse): 
                source, target, rel_type, description, weight_str = match.groups()
                if source and target:
                    weight = None
                    if weight_str and weight_str.strip():
                        try:
                            weight = float(weight_str.strip())
                        except ValueError:
                            print(f"Regex fallback: Could not parse weight '{weight_str}' for relationship.")
                    
                    relationships.append({
                        "source_entity": source.strip(),
                        "target_entity": target.strip(),
                        "relation": rel_type.strip() if rel_type and rel_type.strip() else "RELATED_TO",
                        "relationship_description": description.strip() if description and description.strip() else "",
                        "relationship_weight": weight
                    })
            if entities or relationships:
                 print(f"Regex fallback extracted {len(entities)} entities and {len(relationships)} relationships.")
            else:
                 print("Regex fallback also failed to extract any entities or relationships.")
        
        except Exception as final_e: 
            print(f"An unexpected error occurred during XML parsing (after ET.ParseError or during regex): {final_e}\n"
                        f"Original XML content from LLM (first 500 chars):\n{xml_string[:500]}\n"
                        f"Cleaned XML attempted for parsing (first 500 chars):\n{string_to_parse[:500]}")
        
        return entities, relationships

    async def _extract_knowledge_graph(
        self, chunk_text: str, document_summary: str 
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        
        if not self.aclient_openai or not self.graph_extraction_prompt_template:
            print("OpenAI client or graph extraction prompt not available. Skipping graph extraction.")
            return [], []
        
        formatted_prompt = self.graph_extraction_prompt_template.format(
            document_summary=document_summary, 
            input=chunk_text, 
            entity_types=str([]), 
            relation_types=str([]) 
        )
        print(f"Formatted prompt for graph extraction (chunk-level, to {OPENAI_SUMMARY_MODEL}): First 200 chars: {formatted_prompt[:200]}...")
        
        messages=[
            {"role": "system", "content": "You are an expert assistant that extracts entities and relationships from text and formats them as XML according to the provided schema. Ensure all tags are correctly opened and closed. Use <entity name=\"...\"><type>...</type><description>...</description></entity> and <relationship><source>...</source><target>...</target><type>...</type><description>...</description><weight>...</weight></relationship> format. Wrap multiple entities and relationships in a single <root> or <graph> tag."},
            {"role": "user", "content": formatted_prompt}
        ]

        xml_response_content = await self._call_openai_api(
            model_name=OPENAI_SUMMARY_MODEL,
            payload_messages=messages,
            max_tokens=4000,
            temperature=0.1
        )
        
        if not xml_response_content:
            print("LLM returned empty content for graph extraction.")
            return [], []
        
        print(f"Raw XML response from LLM for chunk-level graph extraction (first 500 chars):\n{xml_response_content[:500]}")
        return self._parse_graph_xml(xml_response_content)

    async def _enrich_chunk_content(
        self, chunk_text: str, document_summary: str, 
        preceding_chunks_texts: List[str], succeeding_chunks_texts: List[str],
    ) -> str:

        if not self.aclient_openai or not self.enrich_prompt_template:
            print("OpenAI client or enrichment prompt not available. Skipping enrichment.")
            return chunk_text 
        
        preceding_context = "\n---\n".join(preceding_chunks_texts)
        succeeding_context = "\n---\n".join(succeeding_chunks_texts)
        max_output_chars = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN_ESTIMATE
        
        formatted_prompt = self.enrich_prompt_template.format(
            document_summary=document_summary, preceding_chunks=preceding_context,
            succeeding_chunks=succeeding_context, chunk=chunk_text, chunk_size=max_output_chars 
        )
        print(f"Formatted prompt for enrichment (to be sent to {OPENAI_CHAT_MODEL}): ...")
        
        messages = [
            {"role": "system", "content": "You are an expert assistant that refines and enriches text chunks according to specific guidelines."},
            {"role": "user", "content": formatted_prompt}
        ]

        enriched_text_content = await self._call_openai_api(
            model_name=OPENAI_CHAT_MODEL,
            payload_messages=messages,
            max_tokens=min(CHUNK_SIZE_TOKENS + CHUNK_OVERLAP_TOKENS, 4000),
            temperature=0.3
        )

        if not enriched_text_content:
            print("LLM returned empty content for chunk enrichment. Using original chunk.")
            return chunk_text
        
        enriched_text = enriched_text_content.strip()
        print(f"Chunk enriched. Original length: {len(chunk_text)}, Enriched length: {len(enriched_text)}")
        return enriched_text

    async def _generate_document_summary(self, full_document_text: str) -> str:         
        
        if not self.aclient_openai or not self.summary_prompt_template:
            print("OpenAI client or summary prompt not available. Skipping document summary generation.")
            return "Summary generation skipped due to missing configuration."
        if not full_document_text.strip():
            print("Full document text is empty. Skipping summary generation.")
            return "Document is empty, no summary generated."

        formatted_prompt = self.summary_prompt_template.format(document=full_document_text)
        print(f"Generating document summary using {OPENAI_SUMMARY_MODEL}...")

        messages = [
            {"role": "user", "content": formatted_prompt}
        ]
        
        summary_text_content = await self._call_openai_api(
            model_name=OPENAI_SUMMARY_MODEL,
            payload_messages=messages,
            max_tokens=SUMMARY_MAX_TOKENS,
            temperature=0.3
        )
        
        if not summary_text_content:
            print("LLM returned empty content for document summary.")
            return "Summary generation resulted in empty content."

        summary_text = summary_text_content.strip()
        print(f"Document summary generated. Length: {len(summary_text)} chars.")
        return summary_text

    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        
        if not self.aclient_openai:
            print("OpenAI client not available. Cannot generate embeddings.")
            return [[] for _ in texts] 
        
        all_embeddings = []
        openai_batch_size = 2048 
        try:
            for i in range(0, len(texts), openai_batch_size):
                batch_texts = texts[i:i + openai_batch_size]
                processed_batch_texts = [text if text.strip() else " " for text in batch_texts]
                
                response = await self.aclient_openai.embeddings.create(
                    input=processed_batch_texts, 
                    model=OPENAI_EMBEDDING_MODEL, 
                    dimensions=OPENAI_EMBEDDING_DIMENSIONS
                )
                all_embeddings.extend([item.embedding for item in response.data])
            return all_embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return [[] for _ in texts] 

    async def _generate_all_raw_chunks_from_doc(
        self,
        doc_text: str,
        file_name: str,
        doc_id: str
    ) -> List[Dict[str, Any]]:
        all_raw_chunks_with_meta: List[Dict[str, Any]] = []
        if not doc_text or not doc_text.strip():
            print(f"Skipping empty document {file_name} for raw chunk generation.")
            return []

        raw_chunks = self.text_splitter.split_text(doc_text)
        print(f'***raw chunks{raw_chunks}')
        for chunk_idx, raw_chunk_text in enumerate(raw_chunks):
            doc_file_name = self.params.get('file_name', file_name)
            print(f"RAW CHUNK (File: {file_name}, Original File Name: {doc_file_name}, Idx: {chunk_idx}, Len {len(raw_chunk_text)}): '''{raw_chunk_text[:100].strip()}...'''")
            all_raw_chunks_with_meta.append({
                "text": raw_chunk_text,
                "page_num": 1,
                "chunk_idx_on_page": chunk_idx,
                "file_name": doc_file_name,
                "doc_id": doc_id
            })
        print(f"Generated {len(all_raw_chunks_with_meta)} raw chunks from {file_name}.")
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
        params: Any
    ) -> Dict[str, Any] | None: 
        chunk_text = raw_chunk_info["text"]
        page_num = raw_chunk_info["page_num"]
        chunk_idx_on_page = raw_chunk_info["chunk_idx_on_page"]
        
        print(f"Starting pipeline for chunk: File {file_name}, Page {page_num}, Index {chunk_idx_on_page}")

        file_extension = os.path.splitext(file_name)[1].lower()
        is_tabular_file = file_extension in ['.csv', '.xlsx']

        if is_tabular_file:
          print(f"Tabular file ({file_extension}) detected. Skipping KG extraction and enrichment.")
          enriched_text = chunk_text
          chunk_entities, chunk_relationships = [], []
        
        else:
            preceding_indices = range(max(0, global_idx - BACKWARD_CHUNKS), global_idx)
            succeeding_indices = range(global_idx + 1, min(len(all_raw_texts), global_idx + 1 + FORWARD_CHUNKS))
            preceding_texts = [all_raw_texts[i] for i in preceding_indices]
            succeeding_texts = [all_raw_texts[i] for i in succeeding_indices]

            contextual_summary = llm_generated_doc_summary
            if not llm_generated_doc_summary or \
            llm_generated_doc_summary == "Document is empty, no summary generated." or \
            llm_generated_doc_summary.startswith("Error during summary generation") or \
            llm_generated_doc_summary == "Summary generation skipped due to missing configuration.":
                contextual_summary = user_provided_doc_summary

            kg_task = asyncio.create_task(
                self._extract_knowledge_graph(chunk_text, contextual_summary)
            )
            enrich_task = asyncio.create_task(
                self._enrich_chunk_content(
                    chunk_text, contextual_summary, preceding_texts, succeeding_texts,
                )
            )

            results = await asyncio.gather(kg_task, enrich_task, return_exceptions=True)
            
            kg_result_or_exc = results[0]
            enrich_result_or_exc = results[1]

            chunk_entities, chunk_relationships = [], []
            if isinstance(kg_result_or_exc, Exception):
                print(f"KG extraction failed for chunk (Page {page_num}, Index {chunk_idx_on_page}) for '{file_name}': {kg_result_or_exc}")
            elif kg_result_or_exc: 
                chunk_entities, chunk_relationships = kg_result_or_exc
                print(f"KG extracted for chunk (Page {page_num}, Index {chunk_idx_on_page}): {len(chunk_entities)} entities, {len(chunk_relationships)} relationships.")
                entity_descriptions_to_embed = [
                    entity['description'] for entity in chunk_entities if entity.get('description', '').strip()
                ]
                entity_indices_with_description = [
                    i for i, entity in enumerate(chunk_entities) if entity.get('description', '').strip()
                ]
                
                if entity_descriptions_to_embed:
                    print(f"Generating embeddings for {len(entity_descriptions_to_embed)} entity descriptions.")
                    try:
                        description_embeddings = await self._generate_embeddings(entity_descriptions_to_embed)
                        
                        if description_embeddings and len(description_embeddings) == len(entity_indices_with_description):
                            for original_index, embedding in zip(entity_indices_with_description, description_embeddings):
                                if embedding:
                                    chunk_entities[original_index]['description_embedding'] = embedding
                            print(f"Successfully assigned {len(description_embeddings)} embeddings to entity descriptions.")
                        else:
                            print(
                                f"Mismatch between number of descriptions ({len(entity_indices_with_description)}) and "
                                f"generated embeddings ({len(description_embeddings) if description_embeddings else 0}). Skipping assignment."
                            )
                    except Exception as e:
                        print(f"Failed to generate or assign embeddings for entity descriptions: {e}")
                
            enriched_text: str
            if isinstance(enrich_result_or_exc, Exception):
                print(f"Enrichment failed for chunk (Page {page_num}, Index {chunk_idx_on_page}) for '{file_name}': {enrich_result_or_exc}. Using original text.")
                enriched_text = chunk_text 
            else:
                enriched_text = enrich_result_or_exc
                print(f"Enrichment successful for chunk (Page {page_num}, Index {chunk_idx_on_page}).")
        
        embedding_list = await self._generate_embeddings([enriched_text]) 
        
        embedding_vector = []
        if embedding_list and embedding_list[0]: 
            embedding_vector = embedding_list[0]
        
        if not embedding_vector: 
            print(f"Skipping chunk from page {page_num}, index {chunk_idx_on_page} for '{file_name}' due to missing embedding.")
            return None

        es_doc_id = f"{doc_id}_p{page_num}_c{chunk_idx_on_page}"
        doc_file_name = self.params.get('file_name', file_name)
        print(f"Original file name: {doc_file_name}")
        metadata_payload = {
            "file_name": doc_file_name, 
            "doc_id": doc_id,
            "page_number": page_num,
            "chunk_index_in_page": chunk_idx_on_page,
            "document_summary": llm_generated_doc_summary,
            "entities": chunk_entities, 
            "relationships": chunk_relationships 
        }
        
        index_name = params.get('index_name')
        print('index name:--', index_name)
        action = {
            "_index": index_name, 
            "_id": es_doc_id,
            "_source": {
                "chunk_text": enriched_text, 
                "embedding": embedding_vector, 
                "metadata": metadata_payload
            }
        }
        print(f"Pipeline complete for chunk (Page {page_num}, Index {chunk_idx_on_page}). ES action prepared.")
        return action

    async def process_pdf(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        print(f"Processing PDF: {file_name} (Doc ID: {doc_id})")
        
        use_ocr = self.params.get('is_ocr_pdf', False)

        try:
            pages_with_tables = 0
            with pdfplumber.open(BytesIO(data)) as pdf:
                if not pdf.pages:
                    print("PDF has no pages. Aborting.")
                    return
                
                for page in pdf.pages:
                    if page.extract_tables():
                        pages_with_tables += 1

                if pages_with_tables > 1:
                    print(f"PDF contains tables on {pages_with_tables} pages. Adjusting chunk size to 2048.")
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2048,
                    chunk_overlap=1024,
                    length_function=tiktoken_len,
                    separators=["\n|", "\n", "|", ". "," ", ""],
                )
        except Exception as e:
            print(f"Could not pre-check PDF with pdfplumber, defaulting to OCR. Error: {e}")
            use_ocr = True

        full_document_text = ""
        if use_ocr:
            print("Processing with OCR parser.")
            ocr_parser = OCRParser()
            ocr_texts = [page_text async for page_text in ocr_parser.ingest(data)]
            full_document_text = " ".join(ocr_texts)
        else:
            print("Processing with standard PDF parser.")
            parser = PDFParser(self.aclient_openai,self)
            all_content_parts = [part async for part in parser.ingest(data)]
            full_document_text = " ".join(all_content_parts)

        if not full_document_text.strip():
            print(f"No text or tables extracted from '{file_name}'. Aborting processing.")
            return

        llm_generated_doc_summary = await self._generate_document_summary(full_document_text)
        
        all_raw_chunks_with_meta = await self._generate_all_raw_chunks_from_doc(
            full_document_text, file_name, doc_id
        )

        if not all_raw_chunks_with_meta:
            print(f"No raw chunks were generated from '{file_name}'. Aborting further processing.")
            return

        all_raw_texts = [chunk["text"] for chunk in all_raw_chunks_with_meta]
        
        print(f"Starting concurrent processing for {len(all_raw_chunks_with_meta)} raw chunks from '{file_name}'.")

        processing_tasks = []
        for i, raw_chunk_info_item in enumerate(all_raw_chunks_with_meta):
            task = asyncio.create_task(
                self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info_item,
                    user_provided_doc_summary=user_provided_document_summary, 
                    llm_generated_doc_summary=llm_generated_doc_summary, 
                    all_raw_texts=all_raw_texts,
                    global_idx=i, file_name=file_name, doc_id=doc_id,
                    params=self.params        
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
        
        print(f"Finished processing for '{file_name}'. Successfully processed and yielded {num_successfully_processed}/{len(all_raw_chunks_with_meta)} chunks.")

    async def process_csv_semantic_chunking(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
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
                full_chunk_text = f"{chunk_context}\n\n{chunk_text}" if chunk_context else chunk_text
            
                # Generate document summary
                if chunk_idx == 0:
                    full_document_text = "\n".join(rows)
                    llm_generated_doc_summary = await self._generate_document_summary(full_document_text)
            
                # Process through individual chunk pipeline
                raw_chunk_info = {
                    "text": full_chunk_text,
                    "page_num": 1,
                    "chunk_idx_on_page": chunk_idx,
                    "file_name": file_name,
                    "doc_id": doc_id
                }
            
                es_action = await self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=[chunk["text"] for chunk in chunks],
                    global_idx=chunk_idx,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params
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
    
        #Strategy 1: Header + batch of rows
        rows_per_chunk = self._calculate_optimal_rows_per_chunk(header_row, data_rows)
    
        for i in range(0, len(data_rows), rows_per_chunk):
            batch_rows = data_rows[i:i + rows_per_chunk]
        
            # Create chunk with header context
            chunk_text = f"CSV Structure:\n{header_row}\n\nData:\n" + "\n".join(batch_rows)
        
            # Add metadata context
            context = f"This is part {(i // rows_per_chunk) + 1} of CSV file '{file_name}' containing rows {i+1} to {min(i + rows_per_chunk, len(data_rows))}."
        
            chunks.append({
                "text": chunk_text,
                "context": context,
                "start_row": i + 1,
                "end_row": min(i + rows_per_chunk, len(data_rows)),
                "total_rows": len(batch_rows)
            })
    
        return chunks
    
    def _calculate_optimal_rows_per_chunk(self, header_row: str, data_rows: List[str]) -> int:
        """
        Calculate optimal number of rows per chunk based on token limits.
        """
        if not data_rows:
            return 1
    
        # Estimate tokens for header and average row
        header_tokens = tiktoken_len(header_row)
        avg_row_tokens = tiktoken_len(data_rows[0]) if data_rows else 1
    
        # Reserve space for context and formatting
        available_tokens = CHUNK_SIZE_TOKENS - header_tokens - 100  # 100 for context/formatting
    
        # Calculate how many rows fit
        rows_per_chunk = max(1, available_tokens // avg_row_tokens)
    
        # Cap at reasonable limits
        return min(rows_per_chunk, 1)  #Max 50 rows per chunk for readability

    async def process_xlsx_semantic_chunking(
        self, data: bytes, file_name: str, doc_id: str, user_provided_document_summary: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processes XLSX files using a row-by-row, text-based chunking approach.
        """
        print(f"Processing XLSX with text-based chunking: {file_name} (Doc ID: {doc_id})")
        try:
            xlsx_parser = XLSXParser()
            # The parser now yields a list of lists of strings
            rows = [row_list async for row_list in xlsx_parser.ingest(data)]

            if not rows:
                print(f"No data extracted from XLSX '{file_name}'. Aborting processing.")
                return

            # Create chunks with the corrected mapping logic
            chunks = await self._create_semantic_xlsx_chunks(rows, file_name)

            if not chunks:
                print(f"No chunks generated from XLSX '{file_name}'. Aborting.")
                return

            # CHANGED: Properly join the list of lists into a single string for the summary
            full_document_text = "\n".join([", ".join(row) for row in rows])
            llm_generated_doc_summary = await self._generate_document_summary(full_document_text)

            # Process each generated chunk through the standard pipeline
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_text = chunk_data["text"]
                raw_chunk_info = {
                    "text": chunk_text,
                    "page_num": 1,  # XLSX is treated as a single "page"
                    "chunk_idx_on_page": chunk_idx,
                    "file_name": file_name,
                    "doc_id": doc_id
                }
                es_action = await self._process_individual_chunk_pipeline(
                    raw_chunk_info=raw_chunk_info,
                    user_provided_doc_summary=user_provided_document_summary,
                    llm_generated_doc_summary=llm_generated_doc_summary,
                    all_raw_texts=[c["text"] for c in chunks],
                    global_idx=chunk_idx,
                    file_name=file_name,
                    doc_id=doc_id,
                    params=self.params
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
            print(f"XLSX file '{file_name}' has no data rows or is missing a header. Skipping.")
            return chunks

        # CHANGED: Headers and rows are now lists, no splitting needed.
        headers = [str(h).strip() for h in data_rows[0]]
        content_rows = data_rows[1:]
            
        format_description = "This chunk contains data from a row where each value is mapped to its column header in a 'Header : Value' format."

        rows_per_chunk = self._calculate_optimal_xlsx_rows_per_chunk(data_rows)
        print(f"Processing XLSX in batches of {rows_per_chunk} rows.")

        for i in range(0, len(content_rows), rows_per_chunk):
            batch_of_rows = content_rows[i:i + rows_per_chunk]
            start_row = i + 1
            end_row = i + len(batch_of_rows)

            all_batch_items = []
            for row_values in batch_of_rows:
                # Pad/truncate values to match header count, ensuring alignment
                num_headers = len(headers)
                row_values.extend([''] * (num_headers - len(row_values)))
                row_values = row_values[:num_headers]
                
                if all_batch_items:
                    all_batch_items.append(("---", "New Row ---"))
                all_batch_items.extend(list(zip(headers, row_values)))

            if not all_batch_items:
                continue

            current_chunk_items = []
            is_first_chunk_for_batch = True

            for header, value in all_batch_items:
                if not header: # Skip empty headers from trailing commas
                    continue

                item_str = f"--- {value} ---" if header == "---" else f"{header} : {value.strip() if value else 'N/A'}"

                prospective_items = current_chunk_items + [item_str]
                
                if is_first_chunk_for_batch:
                    context = f"This is part {start_row} of XLSX file '{file_name}' containing rows {start_row} to {end_row}."
                else:
                    context = f"Continuation of data from rows {start_row} to {end_row} in XLSX file '{file_name}'."
                
                row_data_content = ", ".join(prospective_items)
                prospective_text = f"{context}\n\n{format_description}\n\nRow Data: {row_data_content}"
                
                if tiktoken_len(prospective_text) > CHUNK_SIZE_TOKENS and current_chunk_items:
                    final_context = (f"This is part {start_row} of XLSX file '{file_name}' containing rows {start_row} to {end_row}."
                                     if is_first_chunk_for_batch else f"Continuation of data from rows {start_row} to {end_row} in XLSX file '{file_name}'.")
                    final_row_data = ", ".join(current_chunk_items)
                    chunks.append({"text": f"{final_context}\n\n{format_description}\n\nRow Data: {final_row_data}"})
                    
                    current_chunk_items = [item_str]
                    is_first_chunk_for_batch = False
                else:
                    current_chunk_items = prospective_items

            if current_chunk_items:
                final_context = (f"This is part {start_row} of XLSX file '{file_name}' containing rows {start_row} to {end_row}."
                                 if is_first_chunk_for_batch else f"Continuation of data from rows {start_row} to {end_row} in XLSX file '{file_name}'.")
                final_row_data = ", ".join(current_chunk_items)
                chunks.append({"text": f"{final_context}\n\n{format_description}\n\nRow Data: {final_row_data}"})

        print(f"Created {len(chunks)} text-based chunks from {len(content_rows)} rows in '{file_name}'.")
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
        
        available_tokens = CHUNK_SIZE_TOKENS - 100  # Reserve space for context/formatting
        rows_per_chunk = max(1, available_tokens // (avg_row_tokens if avg_row_tokens > 0 else 1))
        
        return min(rows_per_chunk, 1) # Max rows per chunk for readability

async def ensure_es_index_exists(client: Any, index_name: str, mappings_body: Dict):
    try:
        if not await client.indices.exists(index=index_name):
            updated_mappings = copy.deepcopy(mappings_body)
            updated_mappings["mappings"]["properties"]["embedding"]["dims"] = OPENAI_EMBEDDING_DIMENSIONS
            if "description_embedding" in updated_mappings["mappings"]["properties"]["metadata"]["properties"]["entities"]["properties"]:
                updated_mappings["mappings"]["properties"]["metadata"]["properties"]["entities"]["properties"]["description_embedding"]["dims"] = OPENAI_EMBEDDING_DIMENSIONS

            await client.indices.create(index=index_name, body=updated_mappings)
            print(f"Elasticsearch index '{index_name}' created with specified mappings.")
            return True
        else: 
            print('Index already exits in ensure_es_index_exists')
            current_mapping_response = await client.indices.get_mapping(index=index_name)
            current_top_level_props = current_mapping_response.get(index_name, {}).get('mappings', {}).get('properties', {})
            current_metadata_props = current_top_level_props.get('metadata', {}).get('properties', {})
            
            expected_top_level_props = mappings_body.get('mappings', {}).get('properties', {})
            expected_metadata_props = expected_top_level_props.get('metadata', {}).get('properties', {})
            
            missing_fields = []
            different_fields = []

            for field, expected_field_mapping in expected_top_level_props.items():
                if field == "metadata": continue # Handled separately
                if field not in current_top_level_props:
                    missing_fields.append(field)
                elif current_top_level_props[field].get('type') != expected_field_mapping.get('type'):
                    if field == "embedding" and current_top_level_props[field].get('type') == 'dense_vector' and expected_field_mapping.get('type') == 'dense_vector':
                        if current_top_level_props[field].get('dims') != expected_field_mapping.get('dims'):
                            different_fields.append(f"{field} (dims: {current_top_level_props[field].get('dims')} vs {expected_field_mapping.get('dims')})")
                    else:
                        different_fields.append(f"{field} (type: {current_top_level_props[field].get('type')} vs {expected_field_mapping.get('type')})")
            
            if expected_metadata_props:
                for field, expected_meta_mapping in expected_metadata_props.items():
                    if field not in current_metadata_props:
                        missing_fields.append(f"metadata.{field}")
                    elif current_metadata_props[field].get('type') != expected_meta_mapping.get('type'):
                        different_fields.append(f"metadata.{field} (type: {current_metadata_props[field].get('type')} vs {expected_meta_mapping.get('type')})")


            if missing_fields:
                print(f"Fields {missing_fields} missing in index '{index_name}'. Attempting to update mapping.")
                update_body_props_for_put_mapping = {}
                metadata_updates = {}

                for field_path_to_add in missing_fields:
                    if field_path_to_add.startswith("metadata."):
                        field_name = field_path_to_add.split("metadata.")[1]
                        if field_name in expected_metadata_props:
                            metadata_updates[field_name] = expected_metadata_props[field_name]
                    elif field_path_to_add in expected_top_level_props:
                         update_body_props_for_put_mapping[field_path_to_add] = expected_top_level_props[field_path_to_add]
                
                if metadata_updates:
                    update_body_props_for_put_mapping["metadata"] = {"properties": metadata_updates}

                if update_body_props_for_put_mapping:
                    try:
                        await client.indices.put_mapping(index=index_name, body={"properties": update_body_props_for_put_mapping})
                        print(f"Successfully attempted to add missing fields {missing_fields} to mapping of index '{index_name}'.")
                    except Exception as map_e:
                        print(f"Failed to update mapping for index '{index_name}' to add fields: {map_e}. This might cause issues.")
                else:
                    print(f"Could not prepare update body for missing fields in '{index_name}'.")
            
            if different_fields:
                print(f"Elasticsearch index '{index_name}' exists but mappings for fields {different_fields} differ. This might cause issues.")

            if not missing_fields and not different_fields:
                print(f"Elasticsearch index '{index_name}' already exists and critical fields appear consistent.")
            return True 
    except Exception as e:
        print(f"❌ Error with Elasticsearch index '{index_name}': {e}")
        traceback.print_exc()
        if hasattr(e, 'info'):
            print("🔎 Error details:", json.dumps(e.info, indent=2))
        return False

async def example_run_file_processing(file_data: str | bytes, original_file_name: str, document_id: str, user_provided_doc_summary: str,es_client: Any, aclient_openai: Any, params: Any, config: Any):

    index_name = params.get('index_name')
    print('elastic search index_name:--', index_name)
    
    expected_mappings = copy.deepcopy(CHUNKED_PDF_MAPPINGS)
    dims = OPENAI_EMBEDDING_DIMENSIONS
    
    expected_mappings["mappings"]["properties"]["embedding"]["dims"] = dims
    if "description_embedding" in expected_mappings["mappings"]["properties"]["metadata"]["properties"]["entities"]["properties"]:
        expected_mappings["mappings"]["properties"]["metadata"]["properties"]["entities"]["properties"]["description_embedding"]["dims"] = dims
    
    if not await ensure_es_index_exists(es_client, index_name, CHUNKED_PDF_MAPPINGS):
        print(f"Failed to ensure Elasticsearch index '{index_name}' exists or is compatible. Aborting.")
        return

    file_extension = os.path.splitext(original_file_name)[1].lower()
    processor = ChunkingEmbeddingPDFProcessor(params, config, aclient_openai, file_extension)
    actions_for_es = []
    
    file_extension = os.path.splitext(original_file_name)[1].lower()
    print(f"\n--- Starting Processing for: {original_file_name} (Doc ID: {file_extension}) ---")
    
    try:
        doc_iterator = None
        if file_extension == ".pdf":
            doc_iterator = processor.process_pdf(file_data, original_file_name, document_id,user_provided_doc_summary) 
        elif file_extension == ".csv":
            doc_iterator = processor.process_csv_semantic_chunking(file_data, original_file_name, document_id, user_provided_doc_summary)
        elif file_extension == ".xlsx":
            doc_iterator = processor.process_xlsx_semantic_chunking(file_data, original_file_name, document_id, user_provided_doc_summary)
        else:
            print(f"Unsupported file type: '{file_extension}'. Only .pdf, .csv and .xlsx are supported.")
            return None

        if doc_iterator:
            async for action in doc_iterator:
                if action: 
                    actions_for_es.append(action)

        if actions_for_es:
            print(f"Collected {len(actions_for_es)} actions for bulk ingestion into '{index_name}'.")
            
            if actions_for_es: 
                print("Sample document to be indexed (first one, embedding vector omitted if long):")
                sample_action_copy = copy.deepcopy(actions_for_es[0]) 
                if "_source" in sample_action_copy and "embedding" in sample_action_copy["_source"]:
                    embedding_val = sample_action_copy["_source"]["embedding"]
                    if isinstance(embedding_val, list) and embedding_val:
                        sample_action_copy["_source"]["embedding"] = f"<embedding_vector_dim_{len(embedding_val)}>"
                    elif not embedding_val: 
                         sample_action_copy["_source"]["embedding"] = "<empty_embedding_vector>"
                    else: 
                        sample_action_copy["_source"]["embedding"] = f"<embedding_vector_unexpected_format: {type(embedding_val).__name__}>"

            errors = []
            try:
                successes, response = await async_bulk(es_client, actions_for_es, raise_on_error=False)
                print(f"Elasticsearch bulk ingestion: {successes} successes.")
                
                failed = [r for r in response if not r[0]]
                if failed:
                    print(f"{len(failed)} document(s) failed to index. Showing first error:")
                    errors = failed

            except BulkIndexError as e:
                errors = e.errors
                print("BulkIndexError occurred:")
                print(json.dumps(e.errors, indent=2, default=str))

            if errors:
                print(f"Elasticsearch bulk ingestion errors ({len(errors)}):")
                for i, err_info in enumerate(errors):
                    error_item = err_info.get('index', err_info.get('create', err_info.get('update', err_info.get('delete', {}))))
                    status = error_item.get('status', 'N/A')
                    error_details = error_item.get('error', {})
                    error_type = error_details.get('type', 'N/A')
                    error_reason = error_details.get('reason', 'N/A')
                    doc_id_errored = error_item.get('_id', 'N/A')
                    print(f"Error {i+1}: Doc ID '{doc_id_errored}', Status {status}, Type '{error_type}', Reason: {error_reason}")
        else:
            print(f"No chunks generated or processed for ingestion from '{original_file_name}'.")
            
    except Exception as e:
        print(f"An error occurred during the example run for '{original_file_name}': {e}")
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
  es_client=None
  aclient_openai=None
  
  try:
    print('Incoming Data:--', data)
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
    print('doc_path:----', doc_path)

    try:
        with open(doc_path, "rb") as f:
            doc_bytes_data = f.read()
            print(f"Successfully read PDF file: {doc_path}")
    except Exception as e:
        print(f"Failed to read PDF file '{doc_path}': {e}")
        return

    generated_doc_id = _generate_doc_id_from_content(doc_bytes_data)
    print(f"Generated Document ID (SHA256 of content) for '{original_file_name}': {generated_doc_id}")
    
    user_provided_summary_input = params.get('description') or f"Content from {original_file_name}" 

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
    if aclient_openai and hasattr(aclient_openai, "aclose"): 
      try:
        aclient_openai.aclose() 
        print("OpenAI client closed.")
      except Exception as e:
        print(f"Error closing OpenAI client: {e}")

    print('Rag unstructured file successfully uploaded')
    return FunctionResponse(message=Messages("success"))
  except Exception as e:
    print(f"❌ Error during indexing: {e}")
    return FunctionResponse(message=Messages(e))

def test():
    params = {
        "index_name": "messbill-rowblaze",
        "file_name": "GbhMAYMessBill.pdf",
        "file_path": '/Users/hari/Downloads/GbhMAYMessBill.pdf',
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