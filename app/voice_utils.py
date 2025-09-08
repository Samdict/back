import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import librosa
import io
import aiofiles
import os
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

class VoiceProcessor:
    def __init__(self):
        self.encoder = VoiceEncoder()
        self.sample_rate = 16000  # Resemblyzer expects 16kHz audio
        
    async def process_audio_file(self, file_path: str):
        """Process audio file and return embedding"""
        try:
            # Load and preprocess audio
            wav = preprocess_wav(Path(file_path))
            
            # Create embedding
            embedding = self.encoder.embed_utterance(wav)
            return embedding
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")
    
    async def process_audio_bytes(self, audio_bytes: bytes):
        """Process audio from bytes and return embedding"""
        try:
            # Create a temporary file-like object
            with io.BytesIO(audio_bytes) as audio_file:
                # Load audio using librosa (resemblyzer's preprocess_wav expects a file path)
                audio, _ = librosa.load(audio_file, sr=self.sample_rate)
                
                # Create embedding
                embedding = self.encoder.embed_utterance(audio)
                return embedding
        except Exception as e:
            raise Exception(f"Error processing audio bytes: {str(e)}")
    
    def compare_embeddings(self, embedding1, embedding2, threshold=0.85):
        """Compare two embeddings using cosine similarity"""
        # Reshape for cosine similarity calculation
        embedding1 = embedding1.reshape(1, -1)
        embedding2 = embedding2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # Determine if verification passes based on threshold
        verified = similarity >= threshold
        
        return verified, similarity

# Global instance
voice_processor = VoiceProcessor()