import os
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import io

# Split audio stream at silence points to prevent playback stuttering issues
# caused by AAC encoder frame padding when streaming audio through Gradio audio components.
class AudioStreamProcessor:
    def __init__(self, sr=22050, min_silence_duration=0.1, threshold_db=-40):
        self.sr = sr
        self.min_silence_duration = min_silence_duration
        self.threshold_db = threshold_db
        self.buffer = np.array([])
  
    
    def process(self, audio_data, last=False):
        """
        Add audio data and process it
        params:
            audio_data: audio data in numpy array
            last: whether this is the last chunk of data
        returns:
            Processed audio data, returns None if no split point is found
        """

        # Add new data to buffer
        self.buffer = np.concatenate([self.buffer, audio_data]) if len(self.buffer) > 0 else audio_data
        
        if last:
            result = self.buffer
            self.buffer = np.array([])
            return self._to_wav_bytes(result)
            
        # Find silence boundary
        split_point = self._find_silence_boundary(self.buffer)
        
        if split_point is not None:
            # Modified: Extend split point to the end of silence
            silence_end = self._find_silence_end(split_point)
            result = self.buffer[:silence_end]
            self.buffer = self.buffer[silence_end:]
            return self._to_wav_bytes(result)
            
        return None
        
    def _find_silence_boundary(self, audio):
        """
        Find the starting point of silence boundary in audio
        """
        # Convert audio to decibels
        db = librosa.amplitude_to_db(np.abs(audio), ref=np.max)
        
        # Find points below threshold
        silence_points = np.where(db < self.threshold_db)[0]
        
        if len(silence_points) == 0:
            return None
            
        # Calculate minimum silence samples
        min_silence_samples = int(self.min_silence_duration * self.sr)
        
        # Search backwards for continuous silence segment starting point
        for i in range(len(silence_points) - min_silence_samples, -1, -1):
            if i < 0:
                break
            if np.all(np.diff(silence_points[i:i+min_silence_samples]) == 1):
                return silence_points[i]
                
        return None
        
    def _find_silence_end(self, start_point):
        """
        Find the end point of silence segment
        """
        db = librosa.amplitude_to_db(np.abs(self.buffer[start_point:]), ref=np.max)
        silence_points = np.where(db >= self.threshold_db)[0]
        
        if len(silence_points) == 0:
            return len(self.buffer)
            
        return start_point + silence_points[0]
      
    def _to_wav_bytes(self, audio_data):
        """
        trans_to_wav_bytes
        """
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, self.sr, format='WAV')
        return wav_buffer.getvalue()
      
    
