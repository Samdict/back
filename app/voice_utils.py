# voice_utils.py - Optimized version
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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
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
        self._cache = {}
        self._embedding_cache = {}  # Separate cache for embeddings
        self._audio_cache = {}  # Cache for preprocessed audio
        
        # Thread pool for CPU-intensive tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=2)
        
        # Encoder initialization with lazy loading
        self._encoder = None
        self._encoder_lock = threading.Lock()
        
        # Optimized noise reduction parameters
        self.noise_reduction_params = {
            'stationary': True,
            'prop_decrease': 0.75,  # Reduced for faster processing
            'n_fft': 512,  # Reduced from 512 for speed
            'win_length': 512  # Reduced from 512 for speed
        }
        
        # Cache size limits to prevent memory issues
        self.max_cache_size = 100
    
    @property
    def encoder(self):
        """Lazy load encoder only when needed"""
        if not RESEMBLYZER_AVAILABLE:
            return None
            
        if self._encoder is None:
            with self._encoder_lock:
                if self._encoder is None:  # Double-check locking
                    print("Loading Resemblyzer encoder...")
                    self._encoder = VoiceEncoder()
                    print("Resemblyzer encoder loaded successfully")
        return self._encoder
    
    def _get_audio_hash(self, audio_bytes: bytes):
        """Generate hash for audio bytes for caching"""
        # Use first and last 1KB + file size for faster hashing
        if len(audio_bytes) > 2048:
            hash_content = audio_bytes[:1024] + audio_bytes[-1024:] + str(len(audio_bytes)).encode()
        else:
            hash_content = audio_bytes
        return hashlib.md5(hash_content).hexdigest()
    
    def _manage_cache_size(self, cache_dict):
        """Keep cache size under control"""
        if len(cache_dict) > self.max_cache_size:
            # Remove oldest 20% of entries
            items_to_remove = len(cache_dict) // 5
            keys_to_remove = list(cache_dict.keys())[:items_to_remove]
            for key in keys_to_remove:
                cache_dict.pop(key, None)
    
    async def convert_audio_format(self, audio_bytes: bytes):
        """Convert audio to WAV format if it's not already - optimized"""
        cache_key = f"convert_{self._get_audio_hash(audio_bytes)}"
        
        if cache_key in self._audio_cache:
            return self._audio_cache[cache_key]
        
        def _convert_sync():
            try:
                # Try to load with librosa first (fastest path)
                audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate)
                return audio_bytes  # Return original if it works
            except:
                # If librosa can't read it, convert to WAV using temporary file approach
                try:
                    # Create a temporary file with proper extension detection
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.audio') as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    
                    try:
                        # Try different common audio formats
                        for fmt in ['webm', 'mp3', 'wav', 'ogg', 'm4a', 'mp4']:
                            try:
                                audio = AudioSegment.from_file(tmp_path, format=fmt)
                                wav_io = io.BytesIO()
                                audio.export(wav_io, format="wav")
                                converted_bytes = wav_io.getvalue()
                                return converted_bytes
                            except:
                                continue
                        
                        # If all formats fail, try without format specification
                        audio = AudioSegment.from_file(tmp_path)
                        wav_io = io.BytesIO()
                        audio.export(wav_io, format="wav")
                        return wav_io.getvalue()
                        
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                            
                except Exception as e:
                    raise Exception(f"Audio conversion failed: {str(e)}")
        
        # Run conversion in thread pool
        loop = asyncio.get_event_loop()
        converted_bytes = await loop.run_in_executor(self._thread_pool, _convert_sync)
        
        # Cache the result
        self._audio_cache[cache_key] = converted_bytes
        self._manage_cache_size(self._audio_cache)
        
        return converted_bytes
    
    async def reduce_noise_optimized(self, audio):
        """Apply optimized noise reduction"""
        def _reduce_noise_sync():
            try:
                # Skip noise reduction for very short audio (< 1 second) for speed
                if len(audio) < self.sample_rate:
                    return audio
                    
                # Use faster noise reduction with reduced parameters
                reduced_noise = nr.reduce_noise(
                    y=audio,
                    sr=self.sample_rate,
                    stationary=True,
                    prop_decrease=0.75,  # Less aggressive for speed
                    n_fft=512,  # Smaller FFT for speed
                    win_length=512
                )
                return reduced_noise
            except Exception as e:
                print(f"Noise reduction failed: {e}")
                return audio  # Return original audio if noise reduction fails
        
        # Run noise reduction in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, _reduce_noise_sync)

    async def process_audio_bytes(self, audio_bytes: bytes):
        """Process audio from bytes and return embedding - heavily optimized version"""
        try:
            # Generate cache key
            cache_key = self._get_audio_hash(audio_bytes)
            
            # Check embedding cache first
            if cache_key in self._embedding_cache:
                print("Cache hit - returning cached embedding")
                return self._embedding_cache[cache_key]
            
            def _process_audio_sync():
                """Synchronous audio processing for thread pool"""
                # Convert audio to a format librosa can read if needed
                # Load audio from bytes with optimized parameters
                audio_file = io.BytesIO(audio_bytes)
                
                # Load with reduced precision for speed (mono, specific sample rate)
                audio, _ = librosa.load(
                    audio_file, 
                    sr=self.sample_rate,
                    mono=True,
                    dtype=np.float32  # Use float32 instead of float64
                )
                
                # Skip noise reduction for very short audio
                if len(audio) >= self.sample_rate:
                    # Simplified noise reduction
                    try:
                        audio = nr.reduce_noise(
                            y=audio,
                            sr=self.sample_rate,
                            stationary=True,
                            prop_decrease=0.3,
                            n_fft=256,
                            win_length=256
                        )
                    except:
                        pass  # Continue with original audio if noise reduction fails
                
                # Process with appropriate encoder
                if RESEMBLYZER_AVAILABLE and self.encoder is not None:
                    # Use resemblyzer with preprocessing
                    # Preprocess the wav for resemblyzer
                    processed_audio = preprocess_wav(audio, source_sr=self.sample_rate)
                    embedding = self.encoder.embed_utterance(processed_audio)
                else:
                    # Use enhanced features for better accuracy
                    embedding = self._extract_enhanced_features_sync(audio)
                
                return embedding.astype(np.float32)
            
            # Run audio processing in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(self._thread_pool, _process_audio_sync)
            
            # Cache the result
            self._embedding_cache[cache_key] = embedding
            self._manage_cache_size(self._embedding_cache)
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Error processing audio bytes: {str(e)}")
            
    def _extract_enhanced_features_sync(self, audio):
        """Extract multiple audio features for better speaker discrimination - optimized sync version"""
        # Use reduced parameters for faster computation
        hop_length = 512  # Larger hop length for speed
        n_fft = 1024  # Smaller n_fft for speed
        
        # Extract features with optimized parameters
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=40,  # Reduced from 40
            hop_length=hop_length,
            n_fft=n_fft
        )
        
        chroma = librosa.feature.chroma_stft(
            y=audio, 
            sr=self.sample_rate,
            hop_length=hop_length,
            n_fft=n_fft
        )
        
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, 
            sr=self.sample_rate,
            hop_length=hop_length,
            n_fft=n_fft
        )
        
        # Skip tonnetz for speed (most computationally expensive)
        # tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
        
        # Calculate statistics for each feature type
        mfcc_stats = np.concatenate((
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1)
            # Skip median for speed
        ))
        
        chroma_stats = np.concatenate((
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1)
        ))
        
        contrast_stats = np.concatenate((
            np.mean(spectral_contrast, axis=1),
            np.std(spectral_contrast, axis=1)
        ))
        
        # Combine features (without tonnetz)
        embedding = np.concatenate([mfcc_stats, chroma_stats, contrast_stats])
        
        return embedding.astype(np.float32)   # Use float32 instead of float64
    
    async def extract_enhanced_features(self, audio):
        """Async wrapper for enhanced features extraction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._thread_pool, self._extract_enhanced_features_sync, audio)
    
    def detect_embedding_type(self, embedding):
        """Detect if an embedding is from resemblyzer or the fallback system"""
        if RESEMBLYZER_AVAILABLE:
            # Resemblyzer embeddings are 256-dimensional
            return "resemblyzer" if len(embedding) == 256 else "fallback"
        else:
            # Optimized fallback embeddings are now ~91-dimensional (reduced from 170)
            return "fallback" if len(embedding) in [170, 170] else "unknown"
    
    def compare_embeddings(self, embedding1, embedding2, threshold=0.75):
        """Compare two embeddings using cosine similarity with compatibility handling - optimized"""
        # Detect embedding types
        type1 = self.detect_embedding_type(embedding1)
        type2 = self.detect_embedding_type(embedding2)
        
        # If embeddings are from different systems, return low similarity
        if type1 != type2:
            print(f"Warning: Comparing different embedding types: {type1} vs {type2}")
            return False, 0.0
        
        # Use numpy dot product for faster computation than sklearn
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return False, 0.0
            
        # Calculate cosine similarity using dot product
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure similarity is in valid range
        similarity = np.clip(similarity, -1.0, 1.0)
        
        # Determine if verification passes based on threshold
        verified = similarity >= threshold
        
        return verified, float(similarity)
    
    def clear_cache(self):
        """Clear all caches to free memory"""
        self._cache.clear()
        self._embedding_cache.clear()
        self._audio_cache.clear()
        print("All caches cleared")
    
    def get_cache_stats(self):
        """Get cache statistics for monitoring"""
        return {
            "embedding_cache_size": len(self._embedding_cache),
            "audio_cache_size": len(self._audio_cache),
            "general_cache_size": len(self._cache)
        }

# Global instance
voice_processor = VoiceProcessor()