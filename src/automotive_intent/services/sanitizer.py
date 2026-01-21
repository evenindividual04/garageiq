"""
Input Sanitization Service
Protects against prompt injection and malicious inputs.
"""
import re
import logging
from typing import Tuple, List
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SanitizationResult(BaseModel):
    """Result of input sanitization."""
    original_text: str
    sanitized_text: str
    is_safe: bool
    warnings: List[str] = []
    risk_score: float = 0.0  # 0-1, higher = more risky


# Prompt injection patterns
INJECTION_PATTERNS = [
    # Direct prompt manipulation
    (r"ignore\s+(previous|above|all)\s+(instructions?|prompts?|rules?)", "prompt_override"),
    (r"forget\s+(everything|all|your)\s+(instructions?|training)", "memory_wipe"),
    (r"you\s+are\s+now\s+(?:a|an)\s+", "role_hijack"),
    (r"pretend\s+(?:you\s+are|to\s+be)", "role_hijack"),
    (r"act\s+as\s+(?:if|a|an)", "role_hijack"),
    
    # System prompt extraction
    (r"(?:show|reveal|tell|print|display)\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?prompt", "prompt_extraction"),
    (r"what\s+(?:are|is)\s+your\s+(?:instructions?|prompt|rules?)", "prompt_extraction"),
    
    # Jailbreak attempts
    (r"DAN\s+mode", "jailbreak"),
    (r"developer\s+mode", "jailbreak"),
    (r"bypass\s+(?:safety|filters?|restrictions?)", "jailbreak"),
    
    # Code injection
    (r"```\s*(?:python|javascript|bash|shell|exec)", "code_injection"),
    (r"(?:import|require|eval|exec)\s*\(", "code_injection"),
    
    # Output manipulation
    (r"(?:always\s+)?respond\s+(?:only\s+)?with", "output_control"),
    (r"output\s+(?:only|just)", "output_control"),
]

# Suspicious characters/encodings
SUSPICIOUS_PATTERNS = [
    (r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "control_chars"),  # Control characters
    (r"\\u[0-9a-fA-F]{4}", "unicode_escape"),  # Unicode escapes
    (r"&#x?[0-9a-fA-F]+;", "html_entities"),  # HTML entities
]

# Maximum input length
MAX_INPUT_LENGTH = 2000


class InputSanitizer:
    """
    Sanitizes user input to prevent prompt injection attacks.
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for performance."""
        self.compiled_injection = [
            (re.compile(pattern, re.IGNORECASE), name)
            for pattern, name in INJECTION_PATTERNS
        ]
        self.compiled_suspicious = [
            (re.compile(pattern), name)
            for pattern, name in SUSPICIOUS_PATTERNS
        ]
    
    def sanitize(self, text: str) -> SanitizationResult:
        """
        Sanitize input text and check for injection attempts.
        
        Returns:
            SanitizationResult with sanitized text and safety status
        """
        warnings = []
        risk_score = 0.0
        
        # Check length
        if len(text) > MAX_INPUT_LENGTH:
            text = text[:MAX_INPUT_LENGTH]
            warnings.append(f"Input truncated to {MAX_INPUT_LENGTH} characters")
            risk_score += 0.1
        
        # Check for injection patterns
        for pattern, pattern_name in self.compiled_injection:
            if pattern.search(text):
                warnings.append(f"Potential prompt injection detected: {pattern_name}")
                risk_score += 0.3
                
                if self.strict_mode:
                    # Remove the matched pattern
                    text = pattern.sub("[REMOVED]", text)
        
        # Check for suspicious encodings
        for pattern, pattern_name in self.compiled_suspicious:
            if pattern.search(text):
                warnings.append(f"Suspicious encoding detected: {pattern_name}")
                risk_score += 0.2
                
                # Remove suspicious characters
                text = pattern.sub("", text)
        
        # Basic cleanup
        sanitized = self._basic_cleanup(text)
        
        # Determine safety
        is_safe = risk_score < 0.5
        
        if warnings:
            logger.warning(f"Sanitization warnings: {warnings}, risk_score: {risk_score}")
        
        return SanitizationResult(
            original_text=text,
            sanitized_text=sanitized,
            is_safe=is_safe,
            warnings=warnings,
            risk_score=min(1.0, risk_score)
        )
    
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleanup."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Limit consecutive special characters
        text = re.sub(r'([^\w\s])\1{3,}', r'\1\1\1', text)
        
        return text
    
    def is_automotive_related(self, text: str) -> Tuple[bool, float]:
        """
        Check if input is likely automotive-related.
        Returns (is_automotive, confidence).
        """
        automotive_keywords = {
            # English
            "car", "vehicle", "engine", "brake", "tire", "tyre", "oil", 
            "transmission", "clutch", "steering", "suspension", "battery",
            "alternator", "starter", "ac", "heater", "exhaust", "muffler",
            "fuel", "petrol", "diesel", "cng", "gear", "rpm",
            # Hindi
            "gadi", "gaadi", "brake", "tyre", "engine", "gear", "clutch",
            "steering", "battery", "ac", "petrol", "diesel", "cng",
            # Two-wheeler
            "bike", "scooter", "motorcycle", "chain", "kick", "self start"
        }
        
        text_lower = text.lower()
        matches = sum(1 for keyword in automotive_keywords if keyword in text_lower)
        
        confidence = min(1.0, matches * 0.2)
        is_automotive = matches >= 1
        
        return is_automotive, confidence


# Singleton
_sanitizer = None


def get_sanitizer(strict_mode: bool = False) -> InputSanitizer:
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = InputSanitizer(strict_mode=strict_mode)
    return _sanitizer
