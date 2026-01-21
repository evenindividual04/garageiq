"""
Translation Service for GarageIQ
Handles language detection and translation to technical English.
NLLB enabled by default for production use.
"""
from typing import Optional
from dataclasses import dataclass
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result of translation operation."""
    original_text: str
    detected_language: str
    translated_text: Optional[str]
    is_english: bool
    warnings: list[str]


class TranslatorService:
    """
    Handles language detection and translation.
    Uses langdetect for detection and NLLB for translation.
    NLLB is enabled by default for production.
    """

    SUPPORTED_LANGUAGES = {"en", "hi"}
    
    NLLB_LANG_MAP = {
        "hi": "hin_Deva",
        "en": "eng_Latn",
    }

    def __init__(self, use_nllb: bool = True):
        """
        Initialize translator.
        
        Args:
            use_nllb: If True, load NLLB model for translation (default: True)
        """
        self._nllb_model = None
        self._nllb_tokenizer = None
        self._langdetect_available = False
        self._nllb_available = False

        # Try to import langdetect
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 42
            self._langdetect_available = True
            logger.info("langdetect loaded successfully")
        except ImportError:
            logger.warning("langdetect not available, will default to 'en'")

        # Load NLLB model if requested (default: True)
        if use_nllb:
            self._load_nllb_model()

    def _load_nllb_model(self) -> None:
        """Load NLLB translation model."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch
            
            model_name = os.getenv("AMI_NLLB_MODEL", "facebook/nllb-200-distilled-600M")
            logger.info(f"Loading NLLB model: {model_name}")
            
            # Use CPU by default for broader compatibility
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self._nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            if device == "cuda":
                self._nllb_model = self._nllb_model.to(device)
            
            self._nllb_available = True
            logger.info(f"NLLB model loaded successfully on {device}")
            
        except ImportError as e:
            logger.warning(f"transformers/torch not available: {e}. Translation disabled.")
            self._nllb_available = False
        except Exception as e:
            logger.warning(f"Failed to load NLLB model: {e}. Translation disabled.")
            self._nllb_available = False

    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Returns:
            ISO 639-1 language code (e.g., 'en', 'hi')
        """
        if not self._langdetect_available:
            return "en"
        
        try:
            from langdetect import detect
            detected = detect(text)
            
            # Map common misdetections
            # Hinglish often detected as Indonesian (id) or other languages
            if detected in ["id", "tl", "ms", "cy"]:
                # Check for Hindi words/patterns
                hindi_indicators = ["nahi", "hai", "rahi", "kya", "ho", "kar", "gadi", "gaadi", "bahut", "thanda"]
                text_lower = text.lower()
                if any(word in text_lower for word in hindi_indicators):
                    return "hi"
            
            if detected in self.SUPPORTED_LANGUAGES:
                return detected
            
            return "en"
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"

    def translate_to_english(self, text: str, source_lang: str) -> Optional[str]:
        """
        Translate text to English using NLLB.
        """
        if source_lang == "en":
            return text
        
        if not self._nllb_available:
            logger.warning("NLLB not available, returning original text")
            return None
        
        try:
            import torch
            
            src_code = self.NLLB_LANG_MAP.get(source_lang, "hin_Deva")
            tgt_code = "eng_Latn"
            
            self._nllb_tokenizer.src_lang = src_code
            inputs = self._nllb_tokenizer(text, return_tensors="pt", padding=True)
            
            # Move to same device as model
            device = next(self._nllb_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            translated_tokens = self._nllb_model.generate(
                **inputs,
                forced_bos_token_id=self._nllb_tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=256
            )
            
            translated = self._nllb_tokenizer.decode(
                translated_tokens[0], 
                skip_special_tokens=True
            )
            
            logger.info(f"Translated '{text[:50]}...' -> '{translated[:50]}...'")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return None

    def process(self, text: str) -> TranslationResult:
        """
        Main entry point: detect language and translate if needed.
        """
        warnings = []
        
        # Step 1: Detect language
        detected_lang = self.detect_language(text)
        is_english = detected_lang == "en"
        
        logger.info(f"Detected language: {detected_lang} for text: {text[:50]}...")
        
        # Step 2: Translate if not English
        translated_text = None
        if not is_english:
            translated_text = self.translate_to_english(text, detected_lang)
            if translated_text is None:
                warnings.append("TRANSLATION_FAILED")
                translated_text = text  # Fallback to original
        else:
            translated_text = text
        
        return TranslationResult(
            original_text=text,
            detected_language=detected_lang,
            translated_text=translated_text,
            is_english=is_english,
            warnings=warnings
        )


# Singleton factory
_translator_instance: Optional[TranslatorService] = None


def get_translator(use_nllb: bool = True) -> TranslatorService:
    """Get or create translator instance. NLLB enabled by default."""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = TranslatorService(use_nllb=use_nllb)
    return _translator_instance
