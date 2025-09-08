import numpy as np
import librosa
import io
import aiofiles
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class VoiceProcessor:
    def __init__(self):
        self.sample_rate = 16000
        self.scaler = StandardScaler()
        
    async def process_audio_file(self, file_path: str):
        """Process audio file and return enhanced speaker embedding"""
        try:
            # Load audio using librosa
            audio, _ = librosa.load(file_path, sr=self.sample_rate)
            return await self.extract_enhanced_features(audio)
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")
    
    async def process_audio_bytes(self, audio_bytes: bytes):
        """Process audio from bytes and return embedding"""
        try:
            # Load audio from bytes
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate)
            return await self.extract_enhanced_features(audio)
        except Exception as e:
            raise Exception(f"Error processing audio bytes: {str(e)}")
    
    async def extract_enhanced_features(self, audio):
        """Extract multiple audio features for better speaker discrimination"""
        # Extract multiple features for better speaker discrimination
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
        
        # Calculate statistics for each feature type
        mfcc_stats = np.concatenate((
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.median(mfccs, axis=1)
        ))
        
        chroma_stats = np.concatenate((
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ))
        
        contrast_stats = np.concatenate((
            np.mean(spectral_contrast, axis=1),
            np.std(spectral_contrast, axis=1)
        ))
        
        tonnetz_stats = np.concatenate((
            np.mean(tonnetz, axis=1),
            np.std(tonnetz, axis=1)
        ))
        
        # Combine all features into a single embedding
        embedding = np.concatenate([mfcc_stats, chroma_stats, contrast_stats, tonnetz_stats])
        
        return embedding
    
    def extract_legacy_features(self, audio):
        """Extract legacy MFCC features for backward compatibility"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=20)
        # Average across time to get a fixed-length vector
        return np.mean(mfccs, axis=1)
    
    def detect_embedding_type(self, embedding):
        """Detect if an embedding is from the legacy or enhanced system"""
        return "legacy" if len(embedding) == 20 else "enhanced"
    
    def compare_embeddings(self, embedding1, embedding2, threshold=0.7):
        """Compare two embeddings using cosine similarity with compatibility handling"""
        # Detect embedding types
        type1 = self.detect_embedding_type(embedding1)
        type2 = self.detect_embedding_type(embedding2)
        
        # If embeddings are from different systems, return low similarity
        if type1 != type2:
            return False, 0.0
        
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