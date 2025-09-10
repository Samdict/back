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
    # Warm up the model
    print("Preloading voice processor model...")
    start_time = time.time()
    # Create a dummy audio to initialize the model
    dummy_audio = np.zeros(16000, dtype=np.float32)
    if hasattr(voice_processor.encoder, 'embed_utterance'):
        _ = voice_processor.encoder.embed_utterance(dummy_audio)
    print(f"Model loaded in {time.time() - start_time:.2f} seconds")

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