import os
import base64
from io import BytesIO
from typing import AsyncGenerator, Optional, Any
import yaml
import logging
from pathlib import Path

try:
    from PIL import Image

    PIL_INSTALLED = True
except ImportError:
    PIL_INSTALLED = False

try:
    import filetype

    FILETYPE_INSTALLED = True
except ImportError:
    FILETYPE_INSTALLED = False

try:
    import pillow_heif

    HEIF_INSTALLED = True
    if PIL_INSTALLED:
        pillow_heif.register_heif_opener()
except ImportError:
    HEIF_INSTALLED = False

from src.core.base.parsers.base_parser import AsyncParser


class ImageParser(AsyncParser):
    """A parser for extracting content and descriptions from image files using vision models."""

    # Mapping of file extensions to MIME types
    MIME_TYPE_MAPPING = {
        "bmp": "image/bmp",
        "gif": "image/gif",
        "heic": "image/heic",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "png": "image/png",
        "tiff": "image/tiff",
        "tif": "image/tiff",
        "webp": "image/webp",
    }

    def __init__(
        self,
        aclient_openai: Optional[Any] = None,
        server_type: str = None,
        processor_ref: Optional[Any] = None,
    ):
        self.aclient_openai = aclient_openai
        self.server_type = server_type or os.getenv("SERVER_TYPE")
        self.processor_ref = processor_ref
        self.vision_prompt_text = self._load_vision_prompt()

    def _load_vision_prompt(self) -> str:
        """Load the vision prompt template from YAML file."""
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
                return "Describe the image in detail, including all visible elements, text, and context."
        except Exception as e:
            print(f"Error loading vision prompt: {e}")
            return "Describe the image in detail, including all visible elements, text, and context."

    def _is_heic(self, data: bytes) -> bool:
        """Detect HEIC format using magic numbers and patterns."""
        if not data or len(data) < 12:
            return False

        heic_patterns = [
            b"ftyp",
            b"heic",
            b"heix",
            b"hevc",
            b"HEIC",
            b"mif1",
            b"msf1",
        ]

        try:
            header = data[:32]  # Get first 32 bytes
            return any(pattern in header for pattern in heic_patterns)
        except Exception as e:
            print(f"Error checking for HEIC format: {str(e)}")
            return False

    def _is_jpeg(self, data: bytes) -> bool:
        """Detect JPEG format using magic numbers."""
        return len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8

    def _is_png(self, data: bytes) -> bool:
        """Detect PNG format using magic numbers."""
        png_signature = b"\x89PNG\r\n\x1a\n"
        return data.startswith(png_signature)

    def _is_tiff(self, data: bytes) -> bool:
        """Detect TIFF format using magic numbers."""
        return data.startswith(b"II*\x00") or data.startswith(  # Little-endian
            b"MM\x00*"
        )  # Big-endian

    def _get_image_media_type(self, data: bytes, filename: Optional[str] = None) -> str:
        """Determine the correct media type based on image data and/or filename."""
        try:
            # First, try format-specific detection functions
            if self._is_heic(data):
                return "image/heic"
            if self._is_jpeg(data):
                return "image/jpeg"
            if self._is_png(data):
                return "image/png"
            if self._is_tiff(data):
                return "image/tiff"

            # Try using filetype if available
            if FILETYPE_INSTALLED:
                if img_type := filetype.guess(data):
                    mime = img_type.mime
                    if mime.startswith("image/"):
                        return mime

            # If we have a filename, try to get the type from the extension
            if filename:
                extension = filename.split(".")[-1].lower()
                if extension in self.MIME_TYPE_MAPPING:
                    return self.MIME_TYPE_MAPPING[extension]

            # Default to generic image type
            return "image/jpeg"  # Most common fallback

        except Exception as e:
            print(f"Error determining image media type: {str(e)}")
            return "image/jpeg"  # Default fallback

    async def _convert_heic_to_jpeg(self, data: bytes) -> bytes:
        """Convert HEIC image to JPEG format."""
        if not PIL_INSTALLED or not HEIF_INSTALLED:
            print("PIL or pillow_heif not installed. Cannot convert HEIC to JPEG.")
            return data

        try:
            # Create BytesIO object for input
            input_buffer = BytesIO(data)

            # Load HEIC image
            heif_file = pillow_heif.read_heif(input_buffer)

            # Get the primary image
            heif_image = heif_file[0]  # Get first image in the container

            # Convert to PIL Image
            pil_image = heif_image.to_pillow()

            # Convert to RGB if needed
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Save as JPEG
            output_buffer = BytesIO()
            pil_image.save(output_buffer, format="JPEG", quality=95)
            return output_buffer.getvalue()

        except Exception as e:
            print(f"Error converting HEIC to JPEG: {str(e)}")
            return data  # Return original data on error

    async def _convert_tiff_to_jpeg(self, data: bytes) -> bytes:
        """Convert TIFF image to JPEG format."""
        if not PIL_INSTALLED:
            print("PIL not installed. Cannot convert TIFF to JPEG.")
            return data

        try:
            # Open TIFF image
            with BytesIO(data) as input_buffer:
                tiff_image = Image.open(input_buffer)

                # Convert to RGB if needed
                if tiff_image.mode not in ("RGB", "L"):
                    tiff_image = tiff_image.convert("RGB")

                # Save as JPEG
                output_buffer = BytesIO()
                tiff_image.save(output_buffer, format="JPEG", quality=95)
                return output_buffer.getvalue()
        except Exception as e:
            print(f"Error converting TIFF to JPEG: {str(e)}")
            return data  # Return original data on error

    async def _get_image_description(
        self, image_bytes: bytes, filename: Optional[str] = None
    ) -> str:
        """Get description of image using vision model."""
        media_type = self._get_image_media_type(image_bytes, filename)

        # Check if conversion is needed
        if media_type == "image/heic":
            image_bytes = await self._convert_heic_to_jpeg(image_bytes)
            media_type = "image/jpeg"
        elif media_type == "image/tiff":
            image_bytes = await self._convert_tiff_to_jpeg(image_bytes)
            media_type = "image/jpeg"

        # Encode image as base64
        image_data = base64.b64encode(image_bytes).decode("utf-8")

        if self.server_type == "ARMY":
            if not self.processor_ref:
                print(
                    "Processor reference not available for NVIDIA VLM call. Skipping image description."
                )
                return ""

            print("Using NVIDIA VLM for image description.")
            # Prepare the payload for the NVIDIA API caller
            messages = [
                {
                    "role": "user",
                    "content": self.vision_prompt_text,
                    "image": image_data,
                }
            ]

            try:
                description = await self.processor_ref._call_nvidia_api(
                    payload_messages=messages,
                    is_vision_call=True,
                    max_tokens=1024,
                    temperature=1.0,
                )
                return description.strip() if description else ""
            except Exception as e:
                print(f"Error calling NVIDIA API for image description: {e}")
                return ""

        elif self.aclient_openai:
            print("Using OpenAI VLM for image description.")

            if not self.processor_ref:
                print(
                    "Processor reference not available for OpenAI API call. Skipping image description."
                )
                return ""

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

            try:
                # Use the OpenAI API caller from the processor reference
                description = await self.processor_ref._call_openai_api(
                    model_name=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                    payload_messages=messages,
                    is_vision_call=True,
                    max_tokens=1024,
                    temperature=1.0,
                )
                return description.strip() if description else ""
            except Exception as e:
                print(f"Error calling OpenAI API for image description: {e}")
                return ""

        else:
            print("No vision model available. Skipping image description.")
            return ""

    async def ingest(self, data: bytes, **kwargs) -> AsyncGenerator[str, None]:
        """
        Process an image and yield its description.

        Args:
            data: The binary image data
            **kwargs: Additional arguments including filename

        Yields:
            String description of the image content
        """
        if not isinstance(data, bytes):
            raise TypeError("Image data must be in bytes format.")

        filename = kwargs.get("filename", None)

        try:
            # Get description from vision model
            description = await self._get_image_description(data, filename)

            if description:
                yield description
            else:
                yield "Could not generate description for this image."

        except Exception as e:
            print(f"Failed to process image: {e}")
            yield f"Error processing image: {str(e)}"
