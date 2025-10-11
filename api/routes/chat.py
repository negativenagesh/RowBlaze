import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from elasticsearch import AsyncElasticsearch
from pydantic import BaseModel, Field
from datetime import datetime

from api.dependencies import get_elasticsearch_client

logger = logging.getLogger(__name__)
router = APIRouter()

CHAT_HISTORY_INDEX = "rowblaze_chat_history"

class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None

class ChatSession(BaseModel):
    session_id: str
    title: str
    last_updated: datetime
    message_count: int

class ChatHistory(BaseModel):
    session_id: str
    title: Optional[str] = None
    messages: List[Message]
    timestamp: Optional[datetime] = None

class ChatResponse(BaseModel):
    success: bool
    messages: List[Message] = []

class ChatListResponse(BaseModel):
    success: bool
    sessions: List[ChatSession] = []

@router.post("/chat/save", response_model=ChatResponse)
async def save_chat_history(
    chat_history: ChatHistory,
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client)
):
    """
    Save chat history for a session to Elasticsearch.
    """
    try:
        if not await es_client.indices.exists(index=CHAT_HISTORY_INDEX):
            await es_client.indices.create(
                index=CHAT_HISTORY_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "session_id": {"type": "keyword"},
                            "title": {"type": "text"},
                            "timestamp": {"type": "date"},
                            "messages": {
                                "type": "nested",
                                "properties": {
                                    "role": {"type": "keyword"},
                                    "content": {"type": "text"},
                                    "timestamp": {"type": "date"}
                                }
                            }
                        }
                    }
                }
            )
            logger.info(f"Created chat history index: {CHAT_HISTORY_INDEX}")
        
        if not chat_history.timestamp:
            chat_history.timestamp = datetime.now()
        
        for message in chat_history.messages:
            if not message.timestamp:
                message.timestamp = datetime.now()
        
        if not chat_history.title and chat_history.messages:
            user_messages = [msg for msg in chat_history.messages if msg.role == "user"]
            if user_messages:
                first_msg = user_messages[0].content
                chat_history.title = (first_msg[:40] + "...") if len(first_msg) > 40 else first_msg
            else:
                chat_history.title = f"Chat {chat_history.timestamp.strftime('%Y-%m-%d %H:%M')}"
        
        document = chat_history.dict()
        await es_client.index(
            index=CHAT_HISTORY_INDEX,
            id=chat_history.session_id,
            body=document,
            refresh=True
        )
        
        return ChatResponse(success=True, messages=chat_history.messages)
    
    except Exception as e:
        logger.error(f"Error saving chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving chat history: {str(e)}"
        )

@router.get("/chat/{session_id}", response_model=ChatResponse)
async def get_chat_history(
    session_id: str,
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client)
):
    """
    Retrieve chat history for a session from Elasticsearch.
    """
    try:
        # Check if index exists
        if not await es_client.indices.exists(index=CHAT_HISTORY_INDEX):
            return ChatResponse(success=True, messages=[])
        
        # Check if document exists
        exists = await es_client.exists(index=CHAT_HISTORY_INDEX, id=session_id)
        if not exists:
            return ChatResponse(success=True, messages=[])
        
        # Get document
        response = await es_client.get(index=CHAT_HISTORY_INDEX, id=session_id)
        
        # Extract messages
        if "_source" in response:
            source = response["_source"]
            messages = source.get("messages", [])
            return ChatResponse(success=True, messages=messages)
        
        return ChatResponse(success=True, messages=[])
    
    except Exception as e:
        logger.error(f"Error retrieving chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving chat history: {str(e)}"
        )

@router.get("/chat/list/sessions", response_model=ChatListResponse)
async def list_chat_sessions(
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client)
):
    """
    List all chat sessions from Elasticsearch.
    """
    try:
        # Check if index exists
        if not await es_client.indices.exists(index=CHAT_HISTORY_INDEX):
            return ChatListResponse(success=True, sessions=[])
        
        # Query for all sessions, sorted by timestamp
        response = await es_client.search(
            index=CHAT_HISTORY_INDEX,
            body={
                "sort": [{"timestamp": {"order": "desc"}}],
                "size": 50,  # Limit to most recent 50 sessions
                "_source": ["session_id", "title", "timestamp", "messages"]
            }
        )
        
        # Extract sessions
        sessions = []
        if "hits" in response and "hits" in response["hits"]:
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                sessions.append(
                    ChatSession(
                        session_id=source["session_id"],
                        title=source.get("title", "Untitled Chat"),
                        last_updated=source.get("timestamp", datetime.now()),
                        message_count=len(source.get("messages", []))
                    )
                )
        
        return ChatListResponse(success=True, sessions=sessions)
    
    except Exception as e:
        logger.error(f"Error listing chat sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing chat sessions: {str(e)}"
        )

@router.delete("/chat/{session_id}", response_model=dict)
async def delete_chat_history(
    session_id: str,
    es_client: AsyncElasticsearch = Depends(get_elasticsearch_client)
):
    """
    Delete a chat session from Elasticsearch.
    """
    try:
        # Check if index exists
        if not await es_client.indices.exists(index=CHAT_HISTORY_INDEX):
            return {"success": False, "message": "Chat history index does not exist"}
        
        # Check if document exists
        exists = await es_client.exists(index=CHAT_HISTORY_INDEX, id=session_id)
        if not exists:
            return {"success": False, "message": "Chat session not found"}
        
        # Delete document
        await es_client.delete(
            index=CHAT_HISTORY_INDEX,
            id=session_id,
            refresh=True
        )
        
        return {"success": True, "message": "Chat session deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting chat session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting chat session: {str(e)}"
        )