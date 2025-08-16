import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import soundfile as sf
import librosa
import numpy as np

# Define the models being used
ASR_MODEL = "iamTangsang/Wav2Vec2_XLS-R-300m_Nepali_ASR"
SUMM_MODEL = "csebuetnlp/mT5_multilingual_XLSum"

def _device_dtype():
    """Determines the appropriate device (CPU/GPU) and data type for model inference."""
    if torch.cuda.is_available():
        return "cuda", torch.float16  # Use CUDA (GPU) with half-precision for speed
    elif torch.backends.mps.is_available():
        return "mps", torch.float16  # Use MPS (Apple Silicon) with half-precision
    else:
        return "cpu", torch.float32  # Fallback to CPU with full-precision

class NepaliASRSummarizer:
    """
    A class to handle Nepali Automatic Speech Recognition (ASR) and summarization.
    It loads the necessary Hugging Face models and provides methods for transcription and summarization.
    """
    def __init__(self):
        self.device, self.torch_dtype = _device_dtype()
        print(f"Initializing models on device: {self.device}, dtype: {self.torch_dtype}")

        # ASR model loading
        # The Wav2Vec2Processor handles audio feature extraction and tokenization
        self.asr_processor = Wav2Vec2Processor.from_pretrained(ASR_MODEL)
        # The Wav2Vec2ForCTC model performs the actual speech-to-text conversion
        self.asr_model = Wav2Vec2ForCTC.from_pretrained(ASR_MODEL).to(self.device)
        self.asr_model.eval() # Set model to evaluation mode

        # Summarizer model loading
        # AutoTokenizer handles text tokenization for the summarizer
        self.summ_tokenizer = AutoTokenizer.from_pretrained(SUMM_MODEL, use_fast=True)
        # AutoModelForSeq2SeqLM is a general sequence-to-sequence model for tasks like summarization
        self.summ_model = AutoModelForSeq2SeqLM.from_pretrained(
            SUMM_MODEL,
            torch_dtype=self.torch_dtype if self.device != "cpu" else torch.float32,
        ).to(self.device)
        self.summ_model.eval() # Set model to evaluation mode

        # Create a Hugging Face pipeline for easier summarization
        self.summarizer = pipeline(
            "summarization",
            model=self.summ_model,
            tokenizer=self.summ_tokenizer,
            device=self.device, # Pass device as integer for GPU, string for others (like "mps", "cpu")
            framework="pt", # PyTorch
        )
        print("Models loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribes an audio file from Nepali speech to text.

        Args:
            audio_path (str): The file path to the audio file.

        Returns:
            str: The transcribed text.
        """
        # Load audio and resample to 16kHz if necessary, which is the expected sample rate for Wav2Vec2
        speech, sr = sf.read(audio_path)
        if sr != 16000:
            # librosa.resample expects float, sf.read might return int, convert explicitly
            speech = librosa.resample(y=speech.astype(float), orig_sr=sr, target_sr=16000)
        
        # Process audio to get input features for the ASR model
        input_values = self.asr_processor(speech, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)

        # Perform inference without gradient calculation to save memory and speed up
        with torch.no_grad():
            logits = self.asr_model(input_values).logits
        
        # Get the most likely token IDs and decode them to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.asr_processor.batch_decode(predicted_ids)
        
        return transcription[0].strip()

    def summarize(self, text: str, min_length: int = 40, max_length: int = 160) -> str:
        """
        Summarizes a given text.

        Args:
            text (str): The input text to be summarized.
            min_length (int): Minimum length of the generated summary.
            max_length (int): Maximum length of the generated summary.

        Returns:
            str: The summarized text.
        """
        if not text.strip():
            return "" # Return empty string if input text is empty
        
        # Use the summarization pipeline
        # truncation=True handles cases where input text is too long for the model
        out = self.summarizer(text, min_length=min_length, max_length=max_length, truncation=True)
        return out[0]["summary_text"].strip()

# Singleton engine instance to ensure the models are loaded only once
_engine_instance = None
def get_engine():
    """
    Returns a singleton instance of the NepaliASRSummarizer.
    This prevents reloading models every time get_engine() is called.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = NepaliASRSummarizer()
    return _engine_instance