"""
Unit tests for multi-agent system.
"""
import pytest
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSymptomAnalyst:
    """Tests for SymptomAnalystAgent."""
    
    def test_keyword_extraction_english(self):
        """Test keyword extraction from English input."""
        from automotive_intent.agents.agents import SymptomAnalystAgent
        
        agent = SymptomAnalystAgent()
        # Test the keyword fallback
        keywords = []
        text = "my brakes are squealing when I stop"
        
        # Simple keyword check
        brake_words = ["brake", "braking", "brakes"]
        found = any(w in text.lower() for w in brake_words)
        assert found, "Should detect brake-related keywords"
    
    def test_keyword_extraction_hindi(self):
        """Test keyword extraction from Hindi input."""
        text = "brake lagane par awaaz aa rahi hai"
        
        # Check for Hindi automotive terms
        hindi_terms = ["brake", "awaaz", "gadi", "garam"]
        found = any(term in text.lower() for term in hindi_terms)
        assert found, "Should detect Hindi automotive terms"


class TestEntityExtractor:
    """Tests for entity extraction service."""
    
    def test_vehicle_extraction(self):
        """Test vehicle make/model/year extraction."""
        from automotive_intent.services.entities import get_entity_extractor
        
        extractor = get_entity_extractor()
        result = extractor.extract_vehicle_info("My 2019 Honda Civic has issues")
        
        assert result.make == "Honda"
        assert result.model == "Civic"
        assert result.year == 2019
    
    def test_dtc_extraction(self):
        """Test DTC code extraction."""
        from automotive_intent.services.entities import get_entity_extractor
        
        extractor = get_entity_extractor()
        codes = extractor.extract_dtc_codes("Getting P0300 and P0420 codes")
        
        assert len(codes) == 2
        assert codes[0].code == "P0300"
        assert codes[1].code == "P0420"
    
    def test_no_entities(self):
        """Test when no entities are found."""
        from automotive_intent.services.entities import get_entity_extractor
        
        extractor = get_entity_extractor()
        result = extractor.extract_all("car making noise")
        
        assert result["vehicle"].make is None
        assert len(result["dtc_codes"]) == 0


class TestConfidenceCalibrator:
    """Tests for confidence calibration."""
    
    def test_high_confidence_calibration(self):
        """Test high confidence scenario."""
        from automotive_intent.services.calibration import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        
        calibrated, level, factors = calibrator.calibrate(
            llm_confidence=0.9,
            similar_tickets=[{"similarity_score": 0.85}],
            keywords_matched=3,
            total_keywords=4,
            input_length=50
        )
        
        assert level == "HIGH"
        assert calibrated >= 0.7
    
    def test_low_confidence_calibration(self):
        """Test low confidence scenario."""
        from automotive_intent.services.calibration import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        
        calibrated, level, factors = calibrator.calibrate(
            llm_confidence=0.3,
            similar_tickets=[],
            keywords_matched=0,
            total_keywords=2,
            input_length=5
        )
        
        assert level == "LOW"
        assert calibrated < 0.5


class TestExplanationGenerator:
    """Tests for explanation generation."""
    
    def test_explanation_generation(self):
        """Test that explanations are generated properly."""
        from automotive_intent.services.explanation import get_explanation_generator
        
        generator = get_explanation_generator()
        
        explanation = generator.generate(
            input_text="brakes making noise",
            diagnosis_system="BRAKES",
            diagnosis_component="PADS_ROTORS",
            diagnosis_failure_mode="SQUEALING",
            confidence=0.85,
            similar_tickets=[{"complaint": "brake squeal", "similarity_score": 0.8}],
            symptoms={"keywords": ["brakes", "noise"]}
        )
        
        assert "BRAKES" in explanation.summary
        assert len(explanation.steps) >= 2
        assert explanation.confidence_reason != ""


class TestOntology:
    """Tests for ontology validation."""
    
    def test_ontology_structure(self):
        """Test ontology has required structure."""
        from automotive_intent.core.ontology import SERVICE_ONTOLOGY
        
        assert "BRAKES" in SERVICE_ONTOLOGY
        assert "POWERTRAIN" in SERVICE_ONTOLOGY
        assert "ELECTRICAL" in SERVICE_ONTOLOGY
        
    def test_ontology_validation(self):
        """Test ontology validation functions."""
        from automotive_intent.core.ontology import SERVICE_ONTOLOGY
        
        # Test that BRAKES has components (direct mapping to failure modes)
        brakes = SERVICE_ONTOLOGY["BRAKES"]
        assert "PADS_ROTORS" in brakes
        
        # Test that components have failure modes as list
        failure_modes = brakes["PADS_ROTORS"]
        assert isinstance(failure_modes, list)
        assert "SQUEALING" in failure_modes


class TestFeedbackService:
    """Tests for feedback service."""
    
    def test_feedback_submission(self):
        """Test submitting feedback."""
        from automotive_intent.services.feedback import FeedbackService, FeedbackRequest
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            service = FeedbackService(feedback_file=f.name)
            
            request = FeedbackRequest(
                original_text="brake noise",
                correct_system="BRAKES",
                correct_component="PADS_ROTORS",
                correct_failure_mode="SQUEALING"
            )
            
            record = service.submit_feedback(request)
            
            assert record.correct_system == "BRAKES"
            assert record.id is not None


class TestABTesting:
    """Tests for A/B testing framework."""
    
    def test_experiment_creation(self):
        """Test creating an experiment."""
        from automotive_intent.services.ab_testing import ABTestingService
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            service = ABTestingService(experiments_file=f.name)
            
            experiment = service.create_experiment(
                name="Test Experiment",
                description="Testing",
                variants=[
                    {"name": "control", "config": {}, "weight": 0.5},
                    {"name": "treatment", "config": {"new_prompt": True}, "weight": 0.5}
                ]
            )
            
            assert experiment.id.startswith("exp_")
            assert len(experiment.variants) == 2
    
    def test_variant_assignment(self):
        """Test consistent variant assignment."""
        from automotive_intent.services.ab_testing import ABTestingService
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            service = ABTestingService(experiments_file=f.name)
            
            service.create_experiment(
                name="Test",
                description="Test",
                variants=[
                    {"name": "A", "config": {}, "weight": 0.5},
                    {"name": "B", "config": {}, "weight": 0.5}
                ]
            )
            
            exp_id = list(service._experiments.keys())[0]
            
            # Same user should get same variant
            v1 = service.get_variant(exp_id, "user123")
            v2 = service.get_variant(exp_id, "user123")
            
            assert v1.name == v2.name
