from typing import Any, Dict, List

from pydantic import BaseModel


class Permissions(BaseModel):
    key: str
    value: str
    allowed: bool


class Message(BaseModel):
    params: Dict[str, Any] = {}
    config: Dict[str, Any] = {}
    permissions: List[Permissions] = []
