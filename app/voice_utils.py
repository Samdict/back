# voice_utils.py
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
import noisereduce as nr
from scipy import signal
import librosa
import io
import aiofiles
import hashlib
from functools import lru_cache
import os
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Add these imports for audio format conversion
from pydub import AudioSegment
import tempfile

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
        # noise reduction parameters
        self.noise_reduction_params = {
            'stationary': True,
            'prop_decrease': 0.75,
            'n_fft': 512,
            'win_length': 512
        }
    
    def _get_audio_hash(self, audio_bytes: bytes):
        """Generate hash for audio bytes for caching"""
        return hashlib.md5(audio_bytes).hexdigest()
    
    async def convert_audio_format(self, audio_bytes: bytes):
        """Convert audio to WAV format if it's not already"""
        try:
            # Try to load with librosa first
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate)
            return audio_bytes  # Return original if it works
        except:
            # If librosa can't read it, convert to WAV
            try:
                # Create a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                # Convert to WAV using pydub
                audio = AudioSegment.from_file(tmp_path)
                wav_io = io.BytesIO()
                audio.export(wav_io, format="wav")
                wav_bytes = wav_io.getvalue()
                
                # Clean up
                os.unlink(tmp_path)
                
                return wav_bytes
            except Exception as e:
                raise Exception(f"Audio conversion failed: {str(e)}")
    
    async def reduce_noise(self, audio):
        """Apply noise reduction to audio"""
        try:
            # Apply noise reduction to all audio files
            reduced_noise = nr.reduce_noise(
                y=audio, 
                sr=self.sample_rate,
                stationary=self.noise_reduction_params['stationary'],
                prop_decrease=self.noise_reduction_params['prop_decrease'],
                n_fft=self.noise_reduction_params['n_fft'],
                win_length=self.noise_reduction_params['win_length']
            )
            return reduced_noise
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio  # Return original audio if noise reduction fails
    
    async def process_audio_bytes(self, audio_bytes: bytes):
        """Process audio from bytes and return embedding"""
        try:
            # Generate cache key
            cache_key = self._get_audio_hash(audio_bytes)
            
            if cache_key in self._cache:
                return self._cache[cache_key]
                
            # Convert audio to a format librosa can read if needed
            converted_bytes = await self.convert_audio_format(audio_bytes)
            
            # Load audio from bytes
            audio, _ = librosa.load(io.BytesIO(converted_bytes), sr=self.sample_rate)
            
            # Apply noise reduction to all audio files
            audio = await self.reduce_noise(audio)
            
            if RESEMBLYZER_AVAILABLE:
                # Use resemblyzer if available
                embedding = self.encoder.embed_utterance(audio)
            else:
                # Use enhanced features for better accuracy
                embedding = await self.extract_enhanced_features(audio)
            
            # Cache the result
            self._cache[cache_key] = embedding.astype(np.float32)
            
            return embedding.astype(np.float32)
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
            np.std(tonnetz, axis=1)  # Fixed: changed std to np.std
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
            # Enhanced fallback embeddings are 170-dimensional
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