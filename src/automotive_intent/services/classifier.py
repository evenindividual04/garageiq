"""
Classifier Service for GarageIQ
Handles LLM-based intent classification with strict ontology constraints.
Includes few-shot examples and follow-up question generation.
"""
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import json
import logging
import hashlib
from functools import lru_cache

from ..core.ontology import SERVICE_ONTOLOGY, validate_ontology_path, get_ontology_formatted
from ..core.schemas import Intent

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of classification operation."""
    primary_intent: Optional[Intent]
    alternate_intents: List[Intent]
    is_ambiguous: bool
    is_out_of_scope: bool
    follow_up_questions: List[str] = field(default_factory=list)
    raw_llm_response: Optional[str] = None
    error: Optional[str] = None


# Generate ontology section dynamically
ONTOLOGY_SECTION = get_ontology_formatted()

# System prompt with few-shot examples (English + Hindi + Regional)
SYSTEM_PROMPT = f"""You are an automotive service intent classifier for Indian market.
Classify customer complaints into predefined categories. Understand English, Hindi, Hinglish.

VALID CATEGORIES:
{ONTOLOGY_SECTION}

RULES:
1. Use ONLY exact system/component/failure_mode values from above
2. Confidence 0.0-1.0 (low for vague inputs)
3. If NOT vehicle-related, set out_of_scope: true
4. Hindi terms: gadi=car, awaaz=noise, garam=hot, thanda=cold, panchar=puncture
5. Two-wheelers: scooty/activa/bike/splendor → TWO_WHEELER system
6. CNG/LPG issues → CNG_LPG system

OUTPUT (JSON only):
{{"candidates": [{{"system": "...", "component": "...", "failure_mode": "...", "confidence": 0.XX}}], "out_of_scope": false}}

KEY EXAMPLES:

Input: "Car won't start, clicking" → {{"candidates": [{{"system": "ELECTRICAL", "component": "STARTER_MOTOR", "failure_mode": "SOLENOID_CLICK", "confidence": 0.92}}], "out_of_scope": false}}

Input: "Brake lagane par awaaz" → {{"candidates": [{{"system": "BRAKES", "component": "PADS_ROTORS", "failure_mode": "SQUEALING", "confidence": 0.88}}], "out_of_scope": false}}

Input: "Gadi garam ho rahi hai" → {{"candidates": [{{"system": "POWERTRAIN", "component": "ENGINE", "failure_mode": "OVERHEATING", "confidence": 0.90}}], "out_of_scope": false}}

Input: "AC thanda nahi kar raha" → {{"candidates": [{{"system": "HVAC", "component": "COMPRESSOR", "failure_mode": "NOT_ENGAGING", "confidence": 0.85}}], "out_of_scope": false}}

Input: "Scooty on nahi ho rahi" → {{"candidates": [{{"system": "TWO_WHEELER", "component": "KICK_SELF_START", "failure_mode": "SELF_START_FAILURE", "confidence": 0.88}}], "out_of_scope": false}}

Input: "CNG par switch nahi ho raha" → {{"candidates": [{{"system": "CNG_LPG", "component": "CNG_KIT", "failure_mode": "NOT_SWITCHING", "confidence": 0.90}}], "out_of_scope": false}}

Input: "Gaadi kharab hai" (vague) → {{"candidates": [{{"system": "POWERTRAIN", "component": "ENGINE", "failure_mode": "ROUGH_IDLE", "confidence": 0.30}}], "out_of_scope": false}}

Input: "What is the weather?" → {{"candidates": [], "out_of_scope": true}}

Now classify:
"""

# Follow-up question templates by system
FOLLOW_UP_TEMPLATES = {
    "POWERTRAIN": [
        "Does the issue occur when the engine is cold or warm?",
        "Do you notice any warning lights on the dashboard?",
        "Is there any unusual smell (burning, fuel)?"
    ],
    "ELECTRICAL": [
        "Does this happen consistently or intermittently?",
        "Have you noticed any dim lights or electrical issues?",
        "How old is the battery?"
    ],
    "HVAC": [
        "Does the air blow but not cold, or no air at all?",
        "Do you hear any unusual noises from the AC?",
        "Has the AC worked properly recently?"
    ],
    "BRAKES": [
        "Does the noise happen only when braking or always?",
        "Do you feel vibration in the steering wheel or pedal?",
        "When were the brake pads last replaced?"
    ],
    "SUSPENSION": [
        "Is the noise from the front or rear of the vehicle?",
        "Does it happen on all road conditions or just bumps?",
        "Have you noticed uneven tire wear?"
    ],
    "STEERING": [
        "Is the steering wheel shaking or hard to turn?",
        "Do you notice fluid under the car?",
        "Does it pull to one side while driving?"
    ],
    "EXHAUST": [
        "Is the noise coming from under the car?",
        "Do you smell exhaust or rotten eggs?",
        "Is the check engine light on?"
    ],
    "TIRES_WHEELS": [
        "Which tire is affected (front/rear, left/right)?",
        "Is the TPMS warning light on?",
        "Did you hit a pothole or curb recently?"
    ],
    "GENERAL": [
        "Can you describe where the noise is coming from?",
        "When did you first notice this issue?",
        "Does the problem occur at specific speeds or conditions?"
    ]
}


class ClassifierService:
    """
    LLM-based classifier with strict ontology enforcement.
    Uses Ollama by default, with response caching.
    """

    # Thresholds per PRD §5.4
    CONFIDENCE_THRESHOLD = 0.70
    AMBIGUITY_DELTA = 0.10

    def __init__(self, model_name: str = "mistral", use_ollama: bool = True):
        self.model_name = model_name
        self.use_ollama = use_ollama
        self._client = None
        self._cache: dict = {}

        self.use_ollama = use_ollama
        self._client = None
        self._cache: dict = {}
        
        # Check config for Groq
        from ..config import config
        self.use_groq = getattr(config, "USE_GROQ", False)
        
        if self.use_groq:
            self._init_groq()
        elif use_ollama:
            self._init_ollama()

    def _init_ollama(self) -> None:
        """Initialize Ollama client."""
        try:
            from langchain_ollama import ChatOllama
            self._client = ChatOllama(
                model=self.model_name,
                temperature=0.1,
                format="json"
            )
            logger.info(f"Ollama client initialized with model: {self.model_name}")
        except ImportError:
            logger.warning("langchain_ollama not available, using mock mode")
            self.use_ollama = False
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            self.use_ollama = False

    def _init_groq(self) -> None:
        """Initialize Groq client."""
        try:
            from langchain_groq import ChatGroq
            from ..config import config
            
            if not config.GROQ_API_KEY:
                logger.warning("GROQ_API_KEY not found, falling back to mock")
                self.use_groq = False
                return
                
            self._client = ChatGroq(
                api_key=config.GROQ_API_KEY,
                model_name=config.GROQ_MODEL,
                temperature=0.1,
                max_retries=2,
            )
            logger.info(f"Groq client initialized with model: {config.GROQ_MODEL}")
        except ImportError:
            logger.warning("langchain_groq not available. Install with `pip install langchain-groq`")
            self.use_groq = False
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")
            self.use_groq = False

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from input text."""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _parse_llm_response(self, response: str) -> Tuple[List[dict], bool]:
        """Parse LLM JSON response."""
        logger.info(f"Raw LLM response: {response}")
        
        try:
            data = json.loads(response)
            out_of_scope = data.get("out_of_scope", False)
            candidates = data.get("candidates", [])
            
            if not candidates and "system" in data:
                candidates = [data]
            
            logger.info(f"Parsed candidates: {candidates}, out_of_scope: {out_of_scope}")
            return candidates, out_of_scope
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response was: {response}")
            return [], False

    def _validate_and_create_intents(self, candidates: List[dict]) -> List[Intent]:
        """Validate candidates against ontology and create Intent objects."""
        valid_intents = []
        for c in candidates:
            try:
                system = c.get("system", "").upper()
                component = c.get("component", "").upper()
                failure_mode = c.get("failure_mode", "").upper()
                confidence = float(c.get("confidence", 0.0))

                if validate_ontology_path(system, component, failure_mode):
                    intent = Intent(
                        system=system,
                        component=component,
                        failure_mode=failure_mode,
                        confidence=confidence
                    )
                    valid_intents.append(intent)
                else:
                    logger.warning(f"LLM proposed invalid path: {system}/{component}/{failure_mode}")
            except Exception as e:
                logger.warning(f"Failed to create Intent from candidate: {e}")
        
        return valid_intents

    def _check_ambiguity(self, intents: List[Intent]) -> bool:
        """Check if classification is ambiguous per PRD §5.4."""
        if not intents:
            return True
        
        primary = intents[0]
        
        if primary.confidence < self.CONFIDENCE_THRESHOLD:
            return True
        
        if len(intents) > 1:
            delta = abs(primary.confidence - intents[1].confidence)
            if delta <= self.AMBIGUITY_DELTA:
                return True
        
        return False

    def _generate_follow_up_questions(self, intents: List[Intent], is_ambiguous: bool) -> List[str]:
        """Generate follow-up questions for ambiguous cases."""
        if not is_ambiguous:
            return []
        
        questions = []
        
        # Get questions from candidate systems
        systems_seen = set()
        for intent in intents[:2]:  # Top 2 candidates
            if intent.system not in systems_seen:
                systems_seen.add(intent.system)
                templates = FOLLOW_UP_TEMPLATES.get(intent.system, FOLLOW_UP_TEMPLATES["GENERAL"])
                questions.extend(templates[:1])  # Take 1 from each system
        
        # Add general question if we have space
        if len(questions) < 3:
            questions.extend(FOLLOW_UP_TEMPLATES["GENERAL"][:3 - len(questions)])
        
        return questions[:3]  # Max 3 questions

    def classify(self, text: str) -> ClassificationResult:
        """Classify a complaint text with caching."""
        cache_key = self._get_cache_key(text)
        
        # Check cache
        if cache_key in self._cache:
            logger.info(f"Cache hit for: {text[:50]}...")
            return self._cache[cache_key]
        
        if (not self.use_ollama and not getattr(self, "use_groq", False)) or self._client is None:
            result = self._mock_classify(text)
            self._cache[cache_key] = result
            return result

        try:
            from langchain_core.messages import HumanMessage
            
            # Use system prompt + input in single message for Ollama
            full_prompt = SYSTEM_PROMPT + text
            messages = [HumanMessage(content=full_prompt)]
            
            response = self._client.invoke(messages)
            raw_response = response.content
            
            candidates, out_of_scope = self._parse_llm_response(raw_response)
            
            if out_of_scope:
                result = ClassificationResult(
                    primary_intent=None,
                    alternate_intents=[],
                    is_ambiguous=False,
                    is_out_of_scope=True,
                    follow_up_questions=[],
                    raw_llm_response=raw_response,
                    error=None
                )
                self._cache[cache_key] = result
                return result
            
            valid_intents = self._validate_and_create_intents(candidates)
            
            if not valid_intents:
                result = ClassificationResult(
                    primary_intent=None,
                    alternate_intents=[],
                    is_ambiguous=True,
                    is_out_of_scope=False,
                    follow_up_questions=FOLLOW_UP_TEMPLATES["GENERAL"][:3],
                    raw_llm_response=raw_response,
                    error="No valid intents after ontology validation"
                )
                self._cache[cache_key] = result
                return result
            
            valid_intents.sort(key=lambda x: x.confidence, reverse=True)
            is_ambiguous = self._check_ambiguity(valid_intents)
            follow_ups = self._generate_follow_up_questions(valid_intents, is_ambiguous)
            
            result = ClassificationResult(
                primary_intent=valid_intents[0],
                alternate_intents=valid_intents[1:2],
                is_ambiguous=is_ambiguous,
                is_out_of_scope=False,
                follow_up_questions=follow_ups,
                raw_llm_response=raw_response,
                error=None
            )
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return ClassificationResult(
                primary_intent=None,
                alternate_intents=[],
                is_ambiguous=False,
                is_out_of_scope=False,
                follow_up_questions=[],
                raw_llm_response=None,
                error=str(e)
            )

    def _mock_classify(self, text: str) -> ClassificationResult:
        """Mock classification for testing without Ollama."""
        logger.warning("MOCK CLASSIFIER USED - Ollama not available or disabled")
        text_lower = text.lower()
        
        # Tire/puncture keywords
        if any(w in text_lower for w in ["tire", "tyre", "puncture", "flat", "pressure", "hawa"]):
            intent = Intent(system="TIRES_WHEELS", component="TIRES", failure_mode="PUNCTURE", confidence=0.85)
        elif "start" in text_lower or "crank" in text_lower or "click" in text_lower:
            intent = Intent(system="ELECTRICAL", component="STARTER_MOTOR", failure_mode="NO_CRANK", confidence=0.85)
        elif any(w in text_lower for w in ["ac", "cool", "thanda", "cold", "aircon"]):
            intent = Intent(system="HVAC", component="COMPRESSOR", failure_mode="CLUTCH_FAILURE", confidence=0.78)
        elif any(w in text_lower for w in ["brake", "squeal", "grind", "stopping"]):
            intent = Intent(system="BRAKES", component="PADS_ROTORS", failure_mode="SQUEALING", confidence=0.82)
        elif any(w in text_lower for w in ["bounce", "suspension", "shock", "bumpy"]):
            intent = Intent(system="SUSPENSION", component="SHOCKS_STRUTS", failure_mode="BOUNCY_RIDE", confidence=0.80)
        elif any(w in text_lower for w in ["steering", "wheel turn", "hard to turn"]):
            intent = Intent(system="STEERING", component="POWER_STEERING", failure_mode="STIFF_STEERING", confidence=0.75)
        elif any(w in text_lower for w in ["exhaust", "muffler", "loud noise", "smoke"]):
            intent = Intent(system="EXHAUST", component="MUFFLER", failure_mode="LOUD_EXHAUST", confidence=0.80)
        elif any(w in text_lower for w in ["overheat", "hot", "temperature", "gauage"]):
            intent = Intent(system="POWERTRAIN", component="ENGINE", failure_mode="OVERHEATING", confidence=0.80)
        elif any(w in text_lower for w in ["battery", "dead", "charge"]):
            intent = Intent(system="ELECTRICAL", component="BATTERY", failure_mode="DEAD_CELL", confidence=0.80)
        elif any(w in text_lower for w in ["weather", "hello", "hi", "thanks", "bye"]):
            return ClassificationResult(
                primary_intent=None, alternate_intents=[], is_ambiguous=False,
                is_out_of_scope=True, follow_up_questions=[], raw_llm_response="MOCK", error=None
            )
        else:
            # Unknown - return ambiguous
            logger.warning(f"Mock classifier: No keyword match for: {text[:50]}")
            return ClassificationResult(
                primary_intent=Intent(system="POWERTRAIN", component="ENGINE", failure_mode="ROUGH_IDLE", confidence=0.55),
                alternate_intents=[], is_ambiguous=True, is_out_of_scope=False,
                follow_up_questions=FOLLOW_UP_TEMPLATES["GENERAL"][:3],
                raw_llm_response="MOCK", error=None
            )
        
        is_ambiguous = self._check_ambiguity([intent])
        follow_ups = self._generate_follow_up_questions([intent], is_ambiguous)
        
        return ClassificationResult(
            primary_intent=intent, alternate_intents=[], is_ambiguous=is_ambiguous,
            is_out_of_scope=False, follow_up_questions=follow_ups,
            raw_llm_response="MOCK", error=None
        )

    def clear_cache(self):
        """Clear the classification cache."""
        self._cache.clear()
        logger.info("Classification cache cleared")

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return {"size": len(self._cache), "keys": list(self._cache.keys())[:10]}


# Factory
_classifier_instance: Optional[ClassifierService] = None

def get_classifier(use_ollama: bool = True) -> ClassifierService:
    """Get or create classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ClassifierService(use_ollama=use_ollama)
    return _classifier_instance
