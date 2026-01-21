from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone
from .ontology import validate_ontology_path, VALID_SYSTEMS, get_valid_components, get_valid_failure_modes

# --- Enums (defined as Literals/Constants for Pydantic) ---

SeverityType = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
VehicleStateType = Literal["IMMOBILIZED", "DRIVABLE_WITH_CAUTION", "NORMAL"]
ClassificationStatus = Literal["CONFIRMED", "AMBIGUOUS", "OUT_OF_SCOPE", "SYSTEM_ERROR", "VALIDATION_FAILED"]

# --- Sub-Models ---

class RequestMeta(BaseModel):
    original_text: str
    detected_language: str = Field(..., description="ISO 639-1 code if possible, e.g. 'en', 'hi'")
    timestamp_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

class Normalization(BaseModel):
    translated_text: Optional[str] = None
    technical_summary: str

class Intent(BaseModel):
    system: str
    component: str
    failure_mode: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    vmrs_code: Optional[str] = Field(default=None, description="VMRS industry standard code")

    @model_validator(mode='after')
    def validate_ontology_compliance(self) -> 'Intent':
        """
        Critical: Enforces that the intent tuple exists in the hardcoded ontology.
        If the LLM hallucinates a new Failure Mode, this validation fails.
        """
        # Upper case everything to be safe/canonical
        sys = self.system.upper()
        comp = self.component.upper()
        mode = self.failure_mode.upper()

        if not validate_ontology_path(sys, comp, mode):
             # You might raise values error or handle it. 
             # For a strict schema, we raise ValueError which Pydantic catches.
             # In the pipeline, we must handle this gracefully (e.g. mark as Validation Failed).
             raise ValueError(f"Invalid Ontology Path: {sys} -> {comp} -> {mode} is not allowed.")
        
        # Normalize casing in the model
        self.system = sys
        self.component = comp
        self.failure_mode = mode
        
        # Auto-populate VMRS code if not already set
        if self.vmrs_code is None:
            try:
                from ..services.vmrs_codes import get_vmrs_mapper
                mapper = get_vmrs_mapper()
                vmrs = mapper.get_vmrs_code(sys, comp, mode)
                if vmrs:
                    self.vmrs_code = vmrs.code
            except Exception:
                pass  # Non-critical, leave as None
        
        return self

class Triage(BaseModel):
    severity: SeverityType
    vehicle_state: VehicleStateType
    suggested_action: str

# --- Main Models ---

class ClassificationRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw customer complaint")
    request_id: Optional[str] = None

class ServiceTicket(BaseModel):
    ticket_id: str = Field(default_factory=lambda: f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    classification_status: ClassificationStatus
    meta: RequestMeta
    normalization: Optional[Normalization] = None
    intent: Optional[Intent] = None
    triage: Optional[Triage] = None
    warnings: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_status_consistency(self) -> 'ServiceTicket':
        """
        Ensure data presence matches status.
        e.g. if CONFIRMED, we must have an intent.
        """
        if self.classification_status == "CONFIRMED":
            if not self.intent:
                raise ValueError("Status is CONFIRMED but 'intent' is missing.")
            if not self.triage:
                raise ValueError("Status is CONFIRMED but 'triage' is missing.")
        
        # If AMBIGUOUS, we might strictly require follow-up questions in the future
        # but for now, just ensure we don't present a false intent as fact.
        # (Though we might return the 'best guess' in intent with low confidence, 
        # usually AMBIGUOUS implies we treat the intent as tentative)
        
        return self
