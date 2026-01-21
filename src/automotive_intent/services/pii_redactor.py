"""
PII Redactor Service
Removes personally identifiable information before embedding and display.
"""
import re
import logging
from typing import Tuple, List, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RedactionResult:
    """Result of PII redaction."""
    redacted_text: str
    pii_found: List[Dict[str, str]]
    redaction_count: int


# Regex patterns for various PII types
PII_PATTERNS = {
    # Phone numbers (Indian and International)
    "phone": [
        r'\+91[\s-]?\d{10}',  # Indian with country code
        r'\b[6-9]\d{9}\b',    # Indian mobile
        r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US style
        r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',  # (123) 456-7890
    ],
    
    # Email addresses
    "email": [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    ],
    
    # Indian Aadhaar (12 digits, often with spaces)
    "aadhaar": [
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    ],
    
    # PAN Card (Indian)
    "pan": [
        r'\b[A-Z]{5}\d{4}[A-Z]\b',
    ],
    
    # Credit Card (basic pattern)
    "credit_card": [
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
    ],
    
    # Names (common patterns - limited without NER)
    "name_prefix": [
        r'\b(Mr|Mrs|Ms|Dr|Shri|Smt)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?',
    ],
    
    # Addresses (partial - street numbers and common suffixes)
    "address": [
        r'\b\d+[A-Za-z]?\s+(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Nagar|Colony|Sector)\b',
        r'\bPlot\s+(?:No\.?\s*)?\d+',
        r'\bFlat\s+(?:No\.?\s*)?[A-Za-z]?\d+',
    ],
    
    # Indian PIN codes
    "pincode": [
        r'\b[1-9]\d{5}\b',
    ],
}

# Redaction placeholders
REDACTION_MASKS = {
    "phone": "[PHONE_REDACTED]",
    "email": "[EMAIL_REDACTED]",
    "aadhaar": "[AADHAAR_REDACTED]",
    "pan": "[PAN_REDACTED]",
    "credit_card": "[CARD_REDACTED]",
    "name_prefix": "[NAME_REDACTED]",
    "address": "[ADDRESS_REDACTED]",
    "pincode": "[PIN_REDACTED]",
}


class PIIRedactor:
    """
    Redacts PII from text using regex patterns.
    
    For production, consider:
    - spaCy NER for name detection
    - Presidio for comprehensive PII detection
    - AWS Comprehend or Google DLP for cloud-based
    """
    
    def __init__(self, patterns: Dict = None, masks: Dict = None):
        self.patterns = patterns or PII_PATTERNS
        self.masks = masks or REDACTION_MASKS
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self.compiled = {}
        for pii_type, pattern_list in self.patterns.items():
            self.compiled[pii_type] = [
                re.compile(p, re.IGNORECASE) for p in pattern_list
            ]
    
    def redact(self, text: str) -> RedactionResult:
        """
        Redact all PII from text.
        
        Args:
            text: Input text potentially containing PII
            
        Returns:
            RedactionResult with redacted text and metadata
        """
        if not text:
            return RedactionResult(redacted_text="", pii_found=[], redaction_count=0)
        
        pii_found = []
        redacted = text
        
        for pii_type, patterns in self.compiled.items():
            mask = self.masks.get(pii_type, "[REDACTED]")
            
            for pattern in patterns:
                matches = pattern.findall(redacted)
                for match in matches:
                    # Record what was found (partially masked for logging)
                    pii_found.append({
                        "type": pii_type,
                        "preview": self._partial_mask(match),
                    })
                
                # Replace all occurrences
                redacted = pattern.sub(mask, redacted)
        
        result = RedactionResult(
            redacted_text=redacted,
            pii_found=pii_found,
            redaction_count=len(pii_found)
        )
        
        if pii_found:
            logger.warning(f"Redacted {len(pii_found)} PII items: {[p['type'] for p in pii_found]}")
        
        return result
    
    def _partial_mask(self, value: str) -> str:
        """Create a partial mask for logging (show first/last chars only)."""
        if len(value) <= 4:
            return "***"
        return value[:2] + "***" + value[-2:]
    
    def mask_vin(self, vin: str) -> str:
        """
        Partial VIN masking for display.
        Keeps WMI (first 3) and serial (last 4) for identification.
        """
        if len(vin) != 17:
            return vin
        return vin[:3] + "**********" + vin[-4:]
    
    def is_safe(self, text: str) -> bool:
        """Check if text contains no detectable PII."""
        result = self.redact(text)
        return result.redaction_count == 0


# Singleton
_redactor: PIIRedactor | None = None

def get_pii_redactor() -> PIIRedactor:
    global _redactor
    if _redactor is None:
        _redactor = PIIRedactor()
    return _redactor
