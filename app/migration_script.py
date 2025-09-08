# migration_script.py
import numpy as np
from sqlalchemy.orm import Session
from app.database import SessionLocal, engine
from app import models
from app.voice_utils import voice_processor
import os

def migrate_enrollments():
    db = SessionLocal()
    try:
        # Get all enrollments
        enrollments = db.query(models.Enrollment).all()
        
        print(f"Found {len(enrollments)} enrollments to migrate")
        
        for enrollment in enrollments:
            # Check if this is a legacy enrollment
            stored_embedding = np.frombuffer(enrollment.embedding, dtype=np.float32)
            
            if len(stored_embedding) == 20:  # Legacy MFCC embedding
                print(f"Migrating enrollment {enrollment.id} for user {enrollment.user_id}")
                
                # Find the original audio file (you'll need to adjust this based on your file storage)
                # For now, we'll skip this and just mark for re-enrollment
                print(f"Please re-enroll user {enrollment.user_id} for phrase '{enrollment.phrase}'")
                
        print("Migration complete. Please ask users to re-enroll for better accuracy.")
        
    finally:
        db.close()

if __name__ == "__main__":
    migrate_enrollments()