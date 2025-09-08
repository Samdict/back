from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class UserCreate(BaseModel):
    user_id: str

class UserResponse(BaseModel):
    user_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class EnrollmentCreate(BaseModel):
    user_id: str
    phrase: str

class EnrollmentResponse(BaseModel):
    enrollment_id: int
    user_id: str
    phrase: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class VerificationRequest(BaseModel):
    user_id: str
    phrase: str

class VerificationResponse(BaseModel):
    verified: bool
    confidence: float
    message: str