"""
Intent Classification Pipeline
Orchestrates translation, classification, and triage with follow-up questions.
"""
from typing import Optional, List
import logging
from datetime import datetime, timezone

from .core.schemas import (
    ServiceTicket,
    ClassificationRequest,
    RequestMeta,
    Normalization,
    Intent,
    Triage,
)
from .services.translator import TranslatorService, get_translator
from .services.classifier import ClassifierService, get_classifier, ClassificationResult
from .services.normalizer import get_normalizer

logger = logging.getLogger(__name__)


class TriageEngine:
    """
    Deterministic triage logic.
    Maps intents to severity and vehicle state based on hardcoded rules.
    """

    # Critical failure modes that immobilize the vehicle
    CRITICAL_MODES = {"NO_START", "NO_CRANK", "DEAD_CELL"}
    
    # High severity systems/modes
    HIGH_SEVERITY_SYSTEMS = {"BRAKES"}
    HIGH_SEVERITY_MODES = {"OVERHEATING", "FLUID_LEAK", "SPONGY_PEDAL", "WARNING_LIGHT"}
    
    # Medium severity systems
    MEDIUM_SYSTEMS = {"HVAC", "EXHAUST"}

    @classmethod
    def determine_severity(cls, intent: Intent) -> str:
        """Determine severity based on intent."""
        if intent.failure_mode in cls.CRITICAL_MODES:
            return "CRITICAL"
        if intent.system in cls.HIGH_SEVERITY_SYSTEMS:
            return "HIGH"
        if intent.failure_mode in cls.HIGH_SEVERITY_MODES:
            return "HIGH"
        if intent.system in cls.MEDIUM_SYSTEMS:
            return "MEDIUM"
        if intent.system == "SUSPENSION":
            return "MEDIUM"
        return "MEDIUM"

    @classmethod
    def determine_vehicle_state(cls, intent: Intent) -> str:
        """Determine vehicle state based on intent."""
        if intent.failure_mode in cls.CRITICAL_MODES:
            return "IMMOBILIZED"
        if intent.system == "BRAKES" or intent.failure_mode in cls.HIGH_SEVERITY_MODES:
            return "DRIVABLE_WITH_CAUTION"
        return "NORMAL"

    @classmethod
    def generate_action(cls, intent: Intent) -> str:
        """Generate suggested diagnostic action."""
        actions = {
            ("ELECTRICAL", "STARTER_MOTOR"): "Check battery voltage → inspect starter relay → test starter motor",
            ("ELECTRICAL", "BATTERY"): "Test battery voltage → inspect terminals → check charging system",
            ("ELECTRICAL", "ALTERNATOR"): "Test alternator output → check belt tension → inspect connections",
            ("POWERTRAIN", "ENGINE"): "Check fuel system → inspect ignition → test compression",
            ("POWERTRAIN", "TRANSMISSION"): "Check transmission fluid → inspect linkage → scan DTCs",
            ("HVAC", "COMPRESSOR"): "Check refrigerant level → inspect compressor clutch → verify pressures",
            ("HVAC", "BLOWER_MOTOR"): "Check fuse → test blower resistor → inspect motor",
            ("HVAC", "HEATER_CORE"): "Check coolant level → inspect heater hoses → test thermostat",
            ("BRAKES", "PADS_ROTORS"): "Inspect pad thickness → check rotor condition → verify caliper operation",
            ("BRAKES", "BRAKE_FLUID"): "Check fluid level → inspect for leaks → bleed system",
            ("BRAKES", "ABS"): "Scan ABS codes → check wheel sensors → inspect module",
            ("SUSPENSION", "SHOCKS_STRUTS"): "Visual inspection → bounce test → check for leaks",
            ("SUSPENSION", "BALL_JOINTS"): "Check for play → inspect boots → test under load",
            ("SUSPENSION", "CONTROL_ARMS"): "Inspect bushings → check for play → verify alignment",
            ("STEERING", "POWER_STEERING"): "Check fluid level → inspect pump → look for leaks",
            ("STEERING", "STEERING_RACK"): "Check for play → inspect boots → verify tie rod ends",
            ("STEERING", "TIE_RODS"): "Check for play → inspect boots → verify alignment",
            ("EXHAUST", "CATALYTIC_CONVERTER"): "Scan DTCs → check O2 sensors → inspect for damage",
            ("EXHAUST", "MUFFLER"): "Visual inspection → check hangers → look for rust/holes",
            ("EXHAUST", "EXHAUST_MANIFOLD"): "Visual inspection → check for cracks → verify gaskets",
            ("TIRES_WHEELS", "TIRES"): "Inspect tire → check pressure → locate damage → repair or replace",
            ("TIRES_WHEELS", "WHEELS"): "Inspect rim → check for bends/cracks → verify lug torque",
            ("TIRES_WHEELS", "TPMS"): "Scan TPMS sensors → check battery → verify calibration",
        }
        key = (intent.system, intent.component)
        return actions.get(key, "Perform visual inspection and diagnostic scan")

    @classmethod
    def create_triage(cls, intent: Intent) -> Triage:
        """Create full Triage object for an intent."""
        return Triage(
            severity=cls.determine_severity(intent),
            vehicle_state=cls.determine_vehicle_state(intent),
            suggested_action=cls.generate_action(intent)
        )


class IntentPipeline:
    """
    Main orchestration pipeline.
    Processes raw text through translation → classification → triage → validation.
    Now includes follow-up questions for ambiguous cases.
    """

    def __init__(
        self,
        translator: Optional[TranslatorService] = None,
        classifier: Optional[ClassifierService] = None,
        use_ollama: bool = True,
        use_nllb: bool = True
    ):
        self.translator = translator or get_translator(use_nllb=use_nllb)
        self.classifier = classifier or get_classifier(use_ollama=use_ollama)

    def process(self, request: ClassificationRequest) -> ServiceTicket:
        """Process a classification request through the full pipeline."""
        warnings = []
        
        # Step 0: Normalize noisy input (abbreviations, slang)
        normalizer = get_normalizer()
        normalized_text, norm_metadata = normalizer.normalize(request.text)
        if norm_metadata["changes_made"] > 0:
            warnings.append(f"NORMALIZED: {norm_metadata['changes_made']} corrections applied")
        
        # Step 1: Translation
        translation_result = self.translator.process(normalized_text)
        warnings.extend(translation_result.warnings)
        
        meta = RequestMeta(
            original_text=request.text,
            detected_language=translation_result.detected_language,
            timestamp_utc=datetime.now(timezone.utc).isoformat()
        )
        
        # Step 2: Classification
        text_for_classification = translation_result.translated_text or request.text
        classification_result = self.classifier.classify(text_for_classification)
        
        # Step 3: Handle errors
        if classification_result.error and not classification_result.primary_intent:
            return ServiceTicket(
                classification_status="SYSTEM_ERROR",
                meta=meta,
                warnings=warnings + [f"CLASSIFICATION_ERROR: {classification_result.error}"]
            )
        
        # Step 4: Handle out-of-scope
        if classification_result.is_out_of_scope:
            return ServiceTicket(
                classification_status="OUT_OF_SCOPE",
                meta=meta,
                normalization=Normalization(
                    translated_text=translation_result.translated_text,
                    technical_summary="Query is outside automotive service domain"
                ),
                warnings=warnings
            )
        
        # Step 5: Handle ambiguous
        if classification_result.is_ambiguous:
            intent = classification_result.primary_intent
            normalization = Normalization(
                translated_text=translation_result.translated_text,
                technical_summary=self._generate_summary(intent) if intent else "Unable to determine technical summary"
            )
            
            # Add follow-up questions to warnings for now
            # (Future: add dedicated field to ServiceTicket schema)
            if classification_result.follow_up_questions:
                warnings.append(f"FOLLOW_UP: {' | '.join(classification_result.follow_up_questions)}")
            
            return ServiceTicket(
                classification_status="AMBIGUOUS",
                meta=meta,
                normalization=normalization,
                intent=intent,
                warnings=warnings
            )
        
        # Step 6: Confirmed case
        intent = classification_result.primary_intent
        if intent is None:
            return ServiceTicket(
                classification_status="VALIDATION_FAILED",
                meta=meta,
                warnings=warnings + ["No valid intent produced"]
            )
        
        triage = TriageEngine.create_triage(intent)
        normalization = Normalization(
            translated_text=translation_result.translated_text,
            technical_summary=self._generate_summary(intent)
        )
        
        return ServiceTicket(
            classification_status="CONFIRMED",
            meta=meta,
            normalization=normalization,
            intent=intent,
            triage=triage,
            warnings=warnings
        )

    def _generate_summary(self, intent: Intent) -> str:
        """Generate a technical summary based on the classified intent."""
        summaries = {
            "NO_START": "No-start condition reported",
            "NO_CRANK": "Engine fails to crank",
            "SOLENOID_CLICK": "Audible starter solenoid activation without engine engagement",
            "ROUGH_IDLE": "Engine running rough at idle",
            "OVERHEATING": "Engine overheating condition",
            "OIL_LEAK": "Oil leak detected",
            "MISFIRING": "Engine misfiring detected",
            "STALLING": "Engine stalling condition",
            "HARD_SHIFT": "Harsh gear transitions",
            "SLIPPING": "Transmission slippage during acceleration",
            "DELAYED_ENGAGEMENT": "Delayed response when shifting into gear",
            "LOW_VOLTAGE": "Battery voltage below normal operating range",
            "DEAD_CELL": "Battery cell failure detected",
            "CORROSION": "Battery terminal corrosion present",
            "NOT_CHARGING": "Charging system failure",
            "CLUTCH_FAILURE": "AC compressor clutch malfunction",
            "NOISY_OPERATION": "Abnormal noise from HVAC system",
            "LEAK": "Refrigerant or fluid leak detected",
            "NOT_ENGAGING": "Component not engaging properly",
            "NO_AIRFLOW": "No airflow from HVAC vents",
            "NOISE_VIBRATION": "Abnormal noise or vibration",
            "NO_HEAT": "Heater not producing heat",
            "SQUEALING": "Brake squeal during application",
            "GRINDING": "Metal-on-metal grinding noise",
            "VIBRATION_ON_BRAKE": "Vibration felt during brake application",
            "PULSATING_PEDAL": "Brake pedal pulsating during braking",
            "SPONGY_PEDAL": "Soft or spongy brake pedal",
            "BOUNCY_RIDE": "Excessive bounce or poor ride quality",
            "LEAKING": "Shock/strut leaking fluid",
            "CLUNKING_NOISE": "Clunking noise from suspension",
            "CLUNKING": "Clunking noise detected",
            "WANDERING": "Vehicle wanders or drifts",
            "WHINING_NOISE": "Whining noise from power steering",
            "STIFF_STEERING": "High steering effort required",
            "JERKY_MOVEMENT": "Jerky steering response",
            "LOOSE_STEERING": "Excessive play in steering",
            "ROTTEN_EGG_SMELL": "Sulfur smell indicating catalytic converter issue",
            "REDUCED_POWER": "Reduced engine power",
            "RATTLING": "Rattling noise from exhaust",
            "LOUD_EXHAUST": "Loud exhaust noise",
            "TICKING_NOISE": "Ticking noise from exhaust manifold",
            "EXHAUST_SMELL": "Exhaust fumes detected in cabin",
            # Tires/Wheels
            "PUNCTURE": "Tire puncture detected",
            "FLAT": "Flat tire condition",
            "LOW_PRESSURE": "Low tire pressure",
            "WORN_TREAD": "Tire tread wear detected",
            "BENT_RIM": "Wheel rim damage detected",
        }
        return summaries.get(intent.failure_mode, f"{intent.system} {intent.component} issue reported")


def create_pipeline(use_ollama: bool = True, use_nllb: bool = True) -> IntentPipeline:
    """Create a new pipeline instance."""
    return IntentPipeline(use_ollama=use_ollama, use_nllb=use_nllb)
