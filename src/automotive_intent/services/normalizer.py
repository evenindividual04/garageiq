"""
Noisy Input Normalizer
Handles automotive abbreviations, slang, and code-switching.
"""
import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Automotive abbreviation dictionary
ABBREVIATIONS: Dict[str, str] = {
    # Position
    "frt": "front",
    "frnt": "front",
    "rr": "rear",
    "lft": "left",
    "lt": "left",
    "rt": "right",
    "rgt": "right",
    "ctr": "center",
    "mid": "middle",
    
    # Components
    "eng": "engine",
    "trns": "transmission",
    "trans": "transmission",
    "brk": "brake",
    "brks": "brakes",
    "susp": "suspension",
    "steer": "steering",
    "strg": "steering",
    "exh": "exhaust",
    "exhst": "exhaust",
    "ac": "air conditioning",
    "a/c": "air conditioning",
    "ps": "power steering",
    "pwr": "power",
    "alt": "alternator",
    "batt": "battery",
    "rad": "radiator",
    "comp": "compressor",
    "cond": "condenser",
    "evap": "evaporator",
    "cat": "catalytic converter",
    "o2": "oxygen sensor",
    "maf": "mass airflow sensor",
    "tps": "throttle position sensor",
    "iac": "idle air control",
    "egr": "exhaust gas recirculation",
    "pcv": "positive crankcase ventilation",
    "abs": "anti-lock braking system",
    "tcs": "traction control system",
    "esp": "electronic stability program",
    "ecu": "engine control unit",
    "tcm": "transmission control module",
    "bcm": "body control module",
    
    # Actions/States
    "sts": "states",
    "cus": "customer",
    "cust": "customer",
    "rplc": "replace",
    "rpl": "replace",
    "chk": "check",
    "insp": "inspect",
    "adj": "adjust",
    "rep": "repair",
    "diag": "diagnose",
    "dx": "diagnose",
    
    # Sounds
    "clnk": "clunk",
    "clnking": "clunking",
    "squek": "squeak",
    "sqk": "squeak",
    "grnd": "grinding",
    "hum": "humming",
    "whne": "whine",
    "thmp": "thump",
    "thmping": "thumping",
    "rttl": "rattle",
    "rtl": "rattle",
    
    # India-specific
    "gaadi": "vehicle",
    "gadi": "vehicle",
    "dikkat": "problem",
    "kharab": "broken",
    "awaaz": "noise",
    "awaz": "noise",
    "nahi": "not",
    "ho": "happening",
    "raha": "happening",
    "rahi": "happening",
    "lagane": "applying",
    "chalu": "start",
    "band": "stop",
    "thanda": "cold",
    "garam": "hot",
    
    # Vehicle Types
    "scooty": "scooter",
    "activa": "scooter",
    "splendor": "motorcycle",
    "pulsar": "motorcycle",
    "alto": "car",
    "swift": "car",
    "innova": "car",
    "ertiga": "car",
}

# Common typo patterns
TYPO_PATTERNS = [
    (r'\bw/\b', 'with'),
    (r'\bw/o\b', 'without'),
    (r'\b&\b', 'and'),
    (r'\bn\'t\b', ' not'),
    (r'\bdoesnt\b', 'does not'),
    (r'\bcant\b', 'cannot'),
    (r'\bwont\b', 'will not'),
    (r'\baint\b', 'is not'),
]


class InputNormalizer:
    """Normalizes noisy automotive service input."""
    
    def __init__(self, abbreviations: Dict[str, str] = None):
        self.abbreviations = abbreviations or ABBREVIATIONS
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for efficiency."""
        self.typo_patterns = [(re.compile(p, re.IGNORECASE), r) for p, r in TYPO_PATTERNS]
        
        # Build word boundary pattern for abbreviations
        abbr_pattern = r'\b(' + '|'.join(re.escape(k) for k in self.abbreviations.keys()) + r')\b'
        self.abbr_regex = re.compile(abbr_pattern, re.IGNORECASE)
    
    def normalize(self, text: str) -> Tuple[str, Dict]:
        """
        Normalize input text.
        
        Returns:
            Tuple of (normalized_text, metadata)
            metadata contains: original, changes_made, abbreviations_expanded
        """
        original = text
        changes = []
        
        # Step 1: Apply typo patterns
        for pattern, replacement in self.typo_patterns:
            if pattern.search(text):
                text = pattern.sub(replacement, text)
                changes.append(f"typo: {pattern.pattern} -> {replacement}")
        
        # Step 2: Expand abbreviations
        abbreviations_found = []
        
        def replace_abbr(match):
            abbr = match.group(0).lower()
            expansion = self.abbreviations.get(abbr, abbr)
            if abbr != expansion:
                abbreviations_found.append((abbr, expansion))
            return expansion
        
        text = self.abbr_regex.sub(replace_abbr, text)
        
        # Step 3: Normalize whitespace
        text = ' '.join(text.split())
        
        # Metadata
        metadata = {
            "original": original,
            "normalized": text,
            "changes_made": len(changes) + len(abbreviations_found),
            "abbreviations_expanded": abbreviations_found,
            "typos_fixed": changes,
        }
        
        if metadata["changes_made"] > 0:
            logger.info(f"Normalized input: {len(abbreviations_found)} abbreviations, {len(changes)} typos")
        
        return text, metadata


# Singleton
_normalizer = None

def get_normalizer() -> InputNormalizer:
    global _normalizer
    if _normalizer is None:
        _normalizer = InputNormalizer()
    return _normalizer
