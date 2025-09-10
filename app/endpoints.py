from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, status
from sqlalchemy.orm import Session
import os
import uuid
from datetime import datetime
import numpy as np
import aiofiles 
import re

from . import models, schemas, voice_utils
from .database import get_db

router = APIRouter()

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@router.post("/users", response_model=schemas.UserResponse)
async def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    """Create a new user with name and registration number"""
    # Clean and validate input
    user.name = user.name.strip()
    user.reg_no = user.reg_no.strip()
    
    if not user.name or not user.reg_no:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Name and registration number are required"
        )
    
    # Create user_id from name and reg_no
    user_id = f"{user.name}_{user.reg_no}"
    
    # Check if user already exists
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this name and registration number already exists"
        )
    
    # Create new user
    db_user = models.User(
        user_id=user_id,
        name=user.name,
        reg_no=user.reg_no
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return db_user

@router.post("/enrollments", response_model=schemas.EnrollmentResponse)
async def create_enrollment(
    user_id: str,
    phrase: str,
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Create a new enrollment for a user"""
    # Clean inputs
    user_id = user_id.strip()
    phrase = phrase.strip()
    
    # Check if user exists
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Validate audio file
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an audio file"
        )
    
    # Initialize file_path variable to handle cleanup in case of error
    file_path = None
    
    try:
        # Save uploaded file temporarily
        file_path = f"uploads/{uuid.uuid4()}_{audio_file.filename}"
        async with aiofiles.open(file_path, "wb") as f:
            content = await audio_file.read()
            await f.write(content)
        
        # Process audio to get embedding
        embedding = await voice_utils.voice_processor.process_audio_file(file_path)
        
        # Convert numpy array to bytes for storage
        embedding_bytes = embedding.tobytes()
        
        # Create enrollment record
        db_enrollment = models.Enrollment(
            user_id=user_id,
            phrase=phrase,
            embedding=embedding_bytes
        )
        db.add(db_enrollment)
        db.commit()
        db.refresh(db_enrollment)
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return db_enrollment
        
    except Exception as e:
        # Clean up temporary file if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing enrollment: {str(e)}"
        )

@router.post("/verify", response_model=schemas.VerificationResponse)
async def verify_user(
    user_id: str,
    phrase: str,
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Verify a user's identity using voice biometrics"""
    # Clean inputs
    user_id = user_id.strip()
    phrase = phrase.strip()
    
    # Check if user exists
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get user's enrollments for the specified phrase
    enrollments = db.query(models.Enrollment).filter(
        models.Enrollment.user_id == user_id,
        models.Enrollment.phrase == phrase
    ).all()
    
    print(f"Found {len(enrollments)} enrollments for user {user_id} and phrase '{phrase}'")
    
    if not enrollments:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No enrollments found for this user and phrase"
        )
    
    # Validate audio file
    if not audio_file.content_type or not audio_file.content_type.startswith("audio/"):
        if not audio_file.filename or not any(audio_file.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.m4a', '.ogg', '.webm']):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an audio file"
            )
    
    # Initialize file_path variable to handle cleanup in case of error
    file_path = None
    
    try:
        # Save uploaded file temporarily
        file_path = f"uploads/{uuid.uuid4()}_{audio_file.filename}"
        async with aiofiles.open(file_path, "wb") as f:
            content = await audio_file.read()
            await f.write(content)
        
        # Process verification audio
        verification_embedding = await voice_utils.voice_processor.process_audio_file(file_path)
        
        # Compare with stored enrollments
        best_similarity = 0
        for enrollment in enrollments:
            # Convert bytes back to numpy array
            stored_embedding = np.frombuffer(enrollment.embedding, dtype=np.float32)
            
            # Compare embeddings
            verified, similarity = voice_utils.voice_processor.compare_embeddings(
                verification_embedding, stored_embedding
            )

            print(f"Compared with enrollment {enrollment.id}: similarity={similarity}, embedding_dim={len(stored_embedding)}")
            
            if similarity > best_similarity:
                best_similarity = similarity
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Determine verification result
        # Use a higher threshold for Resemblyzer (0.85), lower for fallback (0.7)
        threshold = 0.80 if voice_utils.voice_processor.detect_embedding_type(verification_embedding) == "resemblyzer" else 0.7
        verified = best_similarity >= threshold
        
        print(f"Verification result: similarity={best_similarity}, threshold={threshold}, verified={verified}")
        
        return {
            "verified": verified,
            "confidence": best_similarity,
            "message": "Verification successful" if verified else "Verification failed"
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        print(f"Error in verification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during verification: {str(e)}"
        )

@router.get("/users/{user_id}/enrollments", response_model=schemas.EnrollmentListResponse)
async def get_user_enrollments(user_id: str, db: Session = Depends(get_db)):
    """Get all enrollments for a user"""
    # Clean input
    user_id = user_id.strip()
    
    # Check if user exists
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get user's enrollments
    enrollments = db.query(models.Enrollment).filter(
        models.Enrollment.user_id == user_id
    ).all()
    
    # Convert to response model without embedding
    enrollment_responses = []
    for enrollment in enrollments:
        enrollment_responses.append(schemas.EnrollmentResponseNoEmbedding(
            id=enrollment.id,
            user_id=enrollment.user_id,
            phrase=enrollment.phrase,
            created_at=enrollment.created_at
        ))
    
    return {
        "enrollments": enrollment_responses,
        "total": len(enrollment_responses)
    }

@router.delete("/users/{user_id}")
async def delete_user(user_id: str, db: Session = Depends(get_db)):
    """Delete a user and all their enrollments"""
    # Clean input
    user_id = user_id.strip()
    
    # Check if user exists
    db_user = db.query(models.User).filter(models.User.user_id == user_id).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Delete user's enrollments first (due to foreign key constraint)
    db.query(models.Enrollment).filter(models.Enrollment.user_id == user_id).delete()
    
    # Delete user
    db.delete(db_user)
    db.commit()
    
    return {"message": "User deleted successfully"}