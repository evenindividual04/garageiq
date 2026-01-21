"""Unit tests for schemas.py"""
import pytest
from pydantic import ValidationError
from automotive_intent.core.schemas import (
    Intent,
    Triage,
    ServiceTicket,
    ClassificationRequest,
    RequestMeta,
    Normalization,
)


class TestIntentValidation:
    """Test that Intent model enforces ontology constraints."""

    def test_valid_intent(self):
        intent = Intent(
            system="ELECTRICAL",
            component="STARTER_MOTOR",
            failure_mode="SOLENOID_CLICK",
            confidence=0.92
        )
        assert intent.system == "ELECTRICAL"
        assert intent.confidence == 0.92

    def test_valid_intent_lowercase_normalized(self):
        """Lowercase input should be uppercased by validator."""
        intent = Intent(
            system="electrical",
            component="starter_motor",
            failure_mode="solenoid_click",
            confidence=0.85
        )
        assert intent.system == "ELECTRICAL"
        assert intent.component == "STARTER_MOTOR"

    def test_invalid_ontology_path_raises(self):
        """Hallucinated failure mode should fail validation."""
        with pytest.raises(ValidationError) as exc_info:
            Intent(
                system="ELECTRICAL",
                component="BATTERY",
                failure_mode="EXPLODED",  # Not in ontology
                confidence=0.9
            )
        assert "Invalid Ontology Path" in str(exc_info.value)

    def test_invalid_system_raises(self):
        with pytest.raises(ValidationError):
            Intent(
                system="SUSPENSION",
                component="SHOCKS",
                failure_mode="WORN",
                confidence=0.8
            )

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            Intent(
                system="POWERTRAIN",
                component="ENGINE",
                failure_mode="NO_START",
                confidence=1.5  # > 1.0
            )


class TestTriageValidation:
    """Test Triage model."""

    def test_valid_triage(self):
        triage = Triage(
            severity="CRITICAL",
            vehicle_state="IMMOBILIZED",
            suggested_action="Check battery voltage"
        )
        assert triage.severity == "CRITICAL"

    def test_invalid_severity(self):
        with pytest.raises(ValidationError):
            Triage(
                severity="URGENT",  # Not a valid literal
                vehicle_state="NORMAL",
                suggested_action="Inspect"
            )


class TestServiceTicket:
    """Test full ServiceTicket model consistency checks."""

    def test_confirmed_requires_intent_and_triage(self):
        with pytest.raises(ValidationError) as exc_info:
            ServiceTicket(
                classification_status="CONFIRMED",
                meta=RequestMeta(
                    original_text="test",
                    detected_language="en"
                ),
                # Missing intent and triage
            )
        assert "intent" in str(exc_info.value).lower() or "triage" in str(exc_info.value).lower()

    def test_confirmed_with_all_fields(self):
        ticket = ServiceTicket(
            classification_status="CONFIRMED",
            meta=RequestMeta(
                original_text="Gadi start nahi ho rahi",
                detected_language="hi"
            ),
            normalization=Normalization(
                translated_text="Car is not starting",
                technical_summary="No-start condition"
            ),
            intent=Intent(
                system="ELECTRICAL",
                component="STARTER_MOTOR",
                failure_mode="NO_CRANK",
                confidence=0.88
            ),
            triage=Triage(
                severity="CRITICAL",
                vehicle_state="IMMOBILIZED",
                suggested_action="Check battery and starter"
            )
        )
        assert ticket.classification_status == "CONFIRMED"
        assert ticket.intent.confidence == 0.88

    def test_ambiguous_status_allowed_without_triage(self):
        """AMBIGUOUS status doesn't require complete triage."""
        ticket = ServiceTicket(
            classification_status="AMBIGUOUS",
            meta=RequestMeta(
                original_text="Awaz aa rahi hai",
                detected_language="hi"
            )
        )
        assert ticket.classification_status == "AMBIGUOUS"

    def test_out_of_scope_status(self):
        ticket = ServiceTicket(
            classification_status="OUT_OF_SCOPE",
            meta=RequestMeta(
                original_text="What is the weather today?",
                detected_language="en"
            )
        )
        assert ticket.classification_status == "OUT_OF_SCOPE"


class TestClassificationRequest:
    """Test input schema."""

    def test_valid_request(self):
        req = ClassificationRequest(text="Bhai AC thanda nahi hai")
        assert req.text == "Bhai AC thanda nahi hai"
        assert req.request_id is None

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            ClassificationRequest(text="")
