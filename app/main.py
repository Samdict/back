from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import numpy as np

from .database import engine, Base
from . import endpoints, models
from .voice_utils import voice_processor

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Voice Biometrics API",
    description="A text-dependent voice verification system",
    version="1.0.0"
)

# Preload voice processor model
@app.on_event("startup")
async def startup_event():
    print("Preloading voice processor model...")
    # Initialize the encoder without processing dummy audio
    if hasattr(voice_processor, 'encoder') and voice_processor.encoder:
        print("Model loaded successfully")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(endpoints.router, prefix="/api/v1", tags=["Voice Biometrics"])

@app.get("/")
async def root():
    return {"message": "Voice Biometrics API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}