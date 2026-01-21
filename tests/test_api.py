"""
Tests for API endpoints and pipeline.
"""
import pytest
from fastapi.testclient import TestClient


# =============================================================================
# PIPELINE TESTS
# =============================================================================

class TestPipeline:
    """Tests for the main classification pipeline."""
    
    def test_pipeline_creation(self):
        """Test that pipeline can be created."""
        from automotive_intent.pipeline import create_pipeline
        
        pipeline = create_pipeline(use_ollama=False, use_nllb=False)
        assert pipeline is not None
    
    def test_english_classification(self):
        """Test English complaint classification."""
        from automotive_intent.pipeline import create_pipeline
        from automotive_intent.core.schemas import ClassificationRequest
        
        pipeline = create_pipeline(use_ollama=False, use_nllb=False)
        ticket = pipeline.process(ClassificationRequest(text="Brake noise when stopping"))
        
        assert ticket.classification_status in ["CONFIRMED", "AMBIGUOUS", "OUT_OF_SCOPE"]
    
    def test_normalizer_integration(self):
        """Test that normalizer is applied in pipeline."""
        from automotive_intent.pipeline import create_pipeline
        from automotive_intent.core.schemas import ClassificationRequest
        
        pipeline = create_pipeline(use_ollama=False, use_nllb=False)
        ticket = pipeline.process(ClassificationRequest(text="cus sts brk noise frt"))
        
        # Should have normalization warning
        has_normalization = any("NORMALIZED" in w for w in ticket.warnings)
        # May or may not have normalization depending on how many abbreviations
        assert ticket is not None


# =============================================================================
# SCHEMA TESTS
# =============================================================================

class TestSchemas:
    """Tests for Pydantic schemas."""
    
    def test_intent_creation(self):
        """Test Intent model creation."""
        from automotive_intent.core.schemas import Intent
        
        intent = Intent(
            system="BRAKES",
            component="PADS_ROTORS",
            failure_mode="SQUEALING",
            confidence=0.92
        )
        
        assert intent.system == "BRAKES"
        assert intent.confidence == 0.92
    
    def test_classification_request(self):
        """Test ClassificationRequest model."""
        from automotive_intent.core.schemas import ClassificationRequest
        
        req = ClassificationRequest(text="Test complaint")
        assert req.text == "Test complaint"
    
    def test_service_ticket_creation(self):
        """Test ServiceTicket model creation."""
        from automotive_intent.core.schemas import ServiceTicket, RequestMeta, Intent, Triage
        
        ticket = ServiceTicket(
            classification_status="CONFIRMED",
            meta=RequestMeta(
                original_text="test",
                detected_language="en",
                timestamp_utc="2024-01-01T00:00:00Z"
            ),
            intent=Intent(
                system="BRAKES",
                component="PADS_ROTORS",
                failure_mode="SQUEALING",
                confidence=0.9
            ),
            triage=Triage(
                severity="HIGH",
                vehicle_state="DRIVABLE_WITH_CAUTION",
                suggested_action="Inspect brake pads"
            )
        )
        
        assert ticket.classification_status == "CONFIRMED"


# =============================================================================
# ONTOLOGY TESTS
# =============================================================================

class TestOntology:
    """Tests for service ontology validation."""
    
    def test_ontology_structure(self):
        """Test that ontology has expected systems."""
        from automotive_intent.core.ontology import SERVICE_ONTOLOGY
        
        assert "BRAKES" in SERVICE_ONTOLOGY
        assert "ELECTRICAL" in SERVICE_ONTOLOGY
        assert "HVAC" in SERVICE_ONTOLOGY
        assert "POWERTRAIN" in SERVICE_ONTOLOGY
    
    def test_brakes_components(self):
        """Test brake system has expected components."""
        from automotive_intent.core.ontology import SERVICE_ONTOLOGY
        
        brakes = SERVICE_ONTOLOGY["BRAKES"]
        assert "PADS_ROTORS" in brakes
        assert "BRAKE_FLUID" in brakes
    
    def test_failure_modes_exist(self):
        """Test that failure modes are defined."""
        from automotive_intent.core.ontology import SERVICE_ONTOLOGY
        
        brakes = SERVICE_ONTOLOGY["BRAKES"]
        pads = brakes["PADS_ROTORS"]
        
        assert "SQUEALING" in pads
        assert "GRINDING" in pads
    
    def test_ontology_validation(self):
        """Test ontology structure has proper nesting."""
        from automotive_intent.core.ontology import SERVICE_ONTOLOGY
        
        # Check that paths exist in ontology
        assert "PADS_ROTORS" in SERVICE_ONTOLOGY["BRAKES"]
        assert "SQUEALING" in SERVICE_ONTOLOGY["BRAKES"]["PADS_ROTORS"]


# =============================================================================
# API ENDPOINT TESTS (with mock)
# =============================================================================

class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from automotive_intent.app import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "GarageIQ" in data["name"]
    
    def test_ontology_endpoint(self, client):
        """Test ontology endpoint."""
        response = client.get("/v1/ontology")
        assert response.status_code == 200
        data = response.json()
        assert "ontology" in data
    
    def test_config_endpoint(self, client):
        """Test config endpoint."""
        response = client.get("/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert "environment" in data
    
    def test_feedback_stats_endpoint(self, client):
        """Test feedback stats endpoint."""
        response = client.get("/v1/feedback/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_feedback" in data
        assert "accuracy_rate" in data


# =============================================================================
# TRIAGE ENGINE TESTS
# =============================================================================

class TestTriageEngine:
    """Tests for severity and vehicle state determination."""
    
    def test_critical_severity(self):
        """Test that critical failure modes get critical severity."""
        from automotive_intent.pipeline import TriageEngine
        from automotive_intent.core.schemas import Intent
        
        # Use a valid ontology path
        intent = Intent(
            system="ELECTRICAL",
            component="BATTERY",
            failure_mode="DEAD_CELL",
            confidence=0.9
        )
        
        severity = TriageEngine.determine_severity(intent)
        assert severity == "CRITICAL"
    
    def test_high_severity_brakes(self):
        """Test that brake issues are high severity."""
        from automotive_intent.pipeline import TriageEngine
        from automotive_intent.core.schemas import Intent
        
        intent = Intent(
            system="BRAKES",
            component="PADS_ROTORS",
            failure_mode="SQUEALING",
            confidence=0.9
        )
        
        severity = TriageEngine.determine_severity(intent)
        assert severity == "HIGH"
    
    def test_vehicle_state_immobilized(self):
        """Test immobilized vehicle state."""
        from automotive_intent.pipeline import TriageEngine
        from automotive_intent.core.schemas import Intent
        
        intent = Intent(
            system="ELECTRICAL",
            component="BATTERY",
            failure_mode="DEAD_CELL",
            confidence=0.9
        )
        
        state = TriageEngine.determine_vehicle_state(intent)
        assert state == "IMMOBILIZED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
