import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool

import librosa
import io
import aiofiles
import hashlib
from functools import lru_cache
import os
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError as e:
    RESEMBLYZER_AVAILABLE = False
    print(f"Resemblyzer not available, falling back to MFCC features: {str(e)}")

class VoiceProcessor:
    def __init__(self):
        self.sample_rate = 16000
        if RESEMBLYZER_AVAILABLE:
            self.encoder = VoiceEncoder()
        else:
            self.encoder = None
        self._cache = {}
    
    def _get_audio_hash(self, file_path: str):
        """Generate hash for audio file for caching"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return str(os.path.getmtime(file_path))
    
    async def process_audio_file(self, file_path: str):
        """Process audio file and return embedding with caching"""
        try:
            # Generate cache key
            cache_key = self._get_audio_hash(file_path)
            
            if cache_key in self._cache:
                return self._cache[cache_key]
                
            if RESEMBLYZER_AVAILABLE:
                # Use resemblyzer if available
                wav = preprocess_wav(file_path)
                embedding = self.encoder.embed_utterance(wav)
            else:
                # Fallback to enhanced MFCC features
                audio, _ = librosa.load(file_path, sr=self.sample_rate)
                embedding = await self.extract_enhanced_features(audio)
            
            # Cache the result
            self._cache[cache_key] = embedding.astype(np.float32)
            
            return embedding.astype(np.float32)
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
                # Fallback to enhanced MFCC features
                embedding = await self.extract_enhanced_features(audio)
            
            return embedding.astype(np.float32)
        except Exception as e:
            raise Exception(f"Error processing audio bytes: {str(e)}")
    
    async def extract_enhanced_features(self, audio):
        """Extract multiple audio features as fallback when resemblyzer is not available"""
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
        
        return embedding.astype(np.float32)
    
    def detect_embedding_type(self, embedding):
        """Detect if an embedding is from resemblyzer or the fallback system"""
        if RESEMBLYZER_AVAILABLE:
            # Resemblyzer embeddings are 256-dimensional
            return "resemblyzer" if len(embedding) == 256 else "fallback"
        else:
            # Fallback embeddings are 170-dimensional
            return "fallback" if len(embedding) == 170 else "unknown"
    
    def compare_embeddings(self, embedding1, embedding2, threshold=0.75):
        """Compare two embeddings using cosine similarity with compatibility handling"""
        # Detect embedding types
        type1 = self.detect_embedding_type(embedding1)
        type2 = self.detect_embedding_type(embedding2)
        
        # If embeddings are from different systems, return low similarity
        if type1 != type2:
            print(f"Warning: Comparing different embedding types: {type1} vs {type2}")
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