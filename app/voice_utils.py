import numpy as np
import librosa
import io
import aiofiles
import os
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Try to import resemblyzer, fallback to MFCC if not available
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    print("Resemblyzer not available, falling back to MFCC features")

class VoiceProcessor:
    def __init__(self):
        self.sample_rate = 16000
        if RESEMBLYZER_AVAILABLE:
            self.encoder = VoiceEncoder()
        else:
            self.encoder = None
    
    async def process_audio_file(self, file_path: str):
        """Process audio file and return embedding"""
        try:
            if RESEMBLYZER_AVAILABLE:
                # Use resemblyzer if available
                wav = preprocess_wav(file_path)
                embedding = self.encoder.embed_utterance(wav)
            else:
                # Fallback to MFCC features
                audio, _ = librosa.load(file_path, sr=self.sample_rate)
                embedding = self.extract_mfcc(audio)
            
            return embedding
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")
    
    async def process_audio_bytes(self, audio_bytes: bytes):
        """Process audio from bytes and return embedding"""
        try:
            # Load audio from bytes
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate)
            
            if RESEMBLYZER_AVAILABLE:
                # Use resemblyzer if available
                embedding = self.encoder.embed_utterance(audio)
            else:
                # Fallback to MFCC features
                embedding = self.extract_mfcc(audio)
            
            return embedding
        except Exception as e:
            raise Exception(f"Error processing audio bytes: {str(e)}")
    
    def extract_mfcc(self, audio, n_mfcc=20):
        """Extract MFCC features as a fallback when resemblyzer is not available"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        # Average across time to get a fixed-length vector
        return np.mean(mfccs, axis=1)
    
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