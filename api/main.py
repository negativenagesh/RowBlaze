import os
import logging
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime

from api.routes import ingestion, retrieval, chat

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API versioning
API_VERSION = "v1"
app = FastAPI(
    title="RowBlaze API",
    description="API for RowBlaze - Advanced RAG for Structured and Unstructured Data",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Create versioned router
v1_router = APIRouter(prefix=f"/api/{API_VERSION}")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health endpoint
@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": API_VERSION
    }

# Include routers
app.include_router(ingestion.router, prefix="/api", tags=["ingestion"])
app.include_router(retrieval.router, prefix="/api", tags=["retrieval"])
app.include_router(chat.router, prefix="/api", tags=["chat"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting RowBlaze API server...")
    # Any initialization code can go here

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down RowBlaze API server...")
    # Cleanup code can go here

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)