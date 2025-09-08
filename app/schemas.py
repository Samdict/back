from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List

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
    id: int = Field(..., description="The enrollment ID")
    user_id: str
    phrase: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# Add this new schema for responses that don't include the embedding
class EnrollmentResponseNoEmbedding(BaseModel):
    id: int = Field(..., description="The enrollment ID")
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

class EnrollmentListResponse(BaseModel):
    enrollments: List[EnrollmentResponseNoEmbedding]
    total: int