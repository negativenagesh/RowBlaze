# src/core/base/parsers/docx_parser.py
import base64
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI

from .base_parser import AsyncParser

try:
    import docx
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.parts.image import ImagePart
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    PYTHON_DOCX_INSTALLED = True
except ImportError:
    PYTHON_DOCX_INSTALLED = False

load_dotenv()

OPENAI_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class DOCXParser(AsyncParser[bytes]):
    """A parser for DOCX data, including text, tables, and image descriptions."""

    def __init__(
        self,
        aclient_openai: Optional[AsyncOpenAI],
        server_type: str,
        processor_ref: Optional[Any] = None,
    ):
        if not PYTHON_DOCX_INSTALLED:
            msg = "DOCX parsing requires 'python-docx'. Please install it (`pip install python-docx`)."
            print(msg)
            raise ImportError(msg)

        self.aclient_openai = aclient_openai
        self.server_type = server_type
        self.processor_ref = processor_ref
        self.vision_prompt_text = self._load_vision_prompt()

    def _load_vision_prompt(self) -> str:
        """Loads the vision prompt from the specified YAML file."""
        try:
            prompt_file_path = Path("./prompts") / "vision_img.yaml"
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

    async def _get_image_description(
        self, image_bytes: bytes, content_type: str
    ) -> str:
        """
        Generates a description for an image using either OpenAI or a referenced processor for ARMY mode.
        Only generates descriptions for images larger than 100x100 pixels.
        """
        # Check image dimensions before processing
        try:
            from io import BytesIO

            from PIL import Image

            img = Image.open(BytesIO(image_bytes))
            width, height = img.size

            # Skip small images (less than 1x1 pixels)
            if width < 1 or height < 1:
                print(
                    f"Skipping small image description (dimensions: {width}x{height})"
                )
                return ""

            print(f"Processing image with dimensions: {width}x{height}")
        except Exception as e:
            print(
                f"Error checking image dimensions: {e}. Will attempt to process anyway."
            )
            # Continue with description generation despite dimension check failure

        image_data = base64.b64encode(image_bytes).decode("utf-8")
        media_type = content_type

        # Route to NVIDIA API via processor_ref for ARMY mode
        if self.server_type == "ARMY":
            if not self.processor_ref:
                print(
                    "Processor reference not available for NVIDIA VLM call. Skipping image description."
                )
                return ""

            print("Using NVIDIA VLM for image description.")
            messages = [
                {
                    "role": "user",
                    "content": self.vision_prompt_text,
                    "image": image_data,
                }
            ]

            description = await self.processor_ref._call_nvidia_api(
                payload_messages=messages, is_vision_call=True, max_tokens=1024
            )
            return (
                f"\n[Image Description]: {description.strip()}\n" if description else ""
            )

        # Default to OpenAI for development mode
        if not self.aclient_openai:
            print("OpenAI client not available, skipping image description.")
            return ""

        try:
            print("Using OpenAI Vision for image description.")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.vision_prompt_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            },
                        },
                    ],
                }
            ]
            response = await self.aclient_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1024,
            )
            description = response.choices[0].message.content
            return (
                f"\n[Image Description]: {description.strip()}\n" if description else ""
            )
        except Exception as e:
            print(f"Error getting image description from OpenAI: {e}")
            return ""

    def _convert_table_to_markdown(self, table: Table) -> str:
        """Converts a docx.table.Table object to a Markdown formatted string."""
        markdown_rows = []
        try:
            # Header row
            header_cells = [
                cell.text.strip().replace("\n", " ").replace("|", r"\|")
                for cell in table.rows[0].cells
            ]
            if not any(header_cells):  # Skip empty tables
                return ""
            markdown_rows.append("| " + " | ".join(header_cells) + " |")

            # Separator line
            markdown_rows.append("| " + " | ".join(["---"] * len(header_cells)) + " |")

            # Body rows
            for row in table.rows[1:]:
                body_cells = [
                    cell.text.strip().replace("\n", " ").replace("|", r"\|")
                    for cell in row.cells
                ]
                # Ensure the number of columns matches the header to avoid malformed Markdown
                if len(body_cells) == len(header_cells):
                    markdown_rows.append("| " + " | ".join(body_cells) + " |")
        except IndexError:
            # Handles cases where the table might be empty or malformed
            return ""

        return "\n".join(markdown_rows)

    async def _process_paragraph_with_precise_ordering(
        self, para: Paragraph, image_parts: dict
    ) -> AsyncGenerator[str, None]:
        """
        Process paragraph content with precise text-image ordering.
        This handles cases where images are embedded mid-sentence.
        """
        # Check if paragraph has any content
        if not para.text.strip() and not any(
            run._r.xpath(".//a:blip/@r:embed") for run in para.runs
        ):
            return

        # Yield paragraph text first (if it exists)
        if para.text.strip():
            yield para.text.strip()

        # Process each run for images (maintaining document order)
        for run_index, run in enumerate(para.runs):
            # Check for images in this specific run
            image_refs = run._r.xpath(".//a:blip/@r:embed")

            for rId in image_refs:
                if rId in image_parts:
                    image_part = image_parts[rId]
                    print(f"Processing image in run {run_index} of paragraph")

                    # Get image dimensions info for logging
                    dimensions_info = ""
                    try:
                        from PIL import Image

                        img = Image.open(BytesIO(image_part.blob))
                        width, height = img.size
                        dimensions_info = f" ({width}x{height})"
                    except Exception as e:
                        print(f"Could not get dimensions for image: {e}")

                    print(
                        f"Found image '{image_part.partname}'{dimensions_info} ({image_part.content_type}) in run {run_index}."
                    )

                    # Process image immediately to maintain order
                    description = await self._get_image_description(
                        image_part.blob, image_part.content_type
                    )
                    if description:
                        yield description

    async def ingest(self, data: bytes, **kwargs) -> AsyncGenerator[str, None]:
        """Ingest DOCX data, yielding paragraphs and tables as strings in their original document order.
        Also saves extracted images to the specified directory."""
        if not isinstance(data, bytes):
            raise TypeError("DOCX data must be in bytes format.")

        # Import PIL here to ensure it's available
        try:
            from PIL import Image
        except ImportError:
            print(
                "Warning: PIL/Pillow not installed. Image dimension filtering will be skipped."
            )
            print("Install with: pip install Pillow")

        docx_stream = BytesIO(data)
        try:
            document = docx.Document(docx_stream)
            print(f"Loaded DOCX file. Iterating through content blocks.")

            # Create a dictionary to map relationship IDs to image parts for easy lookup
            image_parts = {
                rId: part
                for rId, part in document.part.related_parts.items()
                if isinstance(part, ImagePart)
            }

            # Iterate through paragraphs and tables in the order they appear in the document body
            for block in document.element.body:
                if isinstance(block, CT_P):
                    para = Paragraph(block, document)

                    # NEW: Use the precise ordering method instead of the old approach
                    async for (
                        content_item
                    ) in self._process_paragraph_with_precise_ordering(
                        para, image_parts
                    ):
                        yield content_item

                elif isinstance(block, CT_Tbl):
                    table = Table(block, document)
                    markdown_table = self._convert_table_to_markdown(table)
                    if markdown_table:
                        yield markdown_table

        except Exception as e:
            print(f"Failed to read DOCX stream: {e}", exc_info=True)
            raise ValueError(f"Error processing DOCX file: {str(e)}") from e
        finally:
            docx_stream.close()
