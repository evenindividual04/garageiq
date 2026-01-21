"""
Tests for all service modules.
Covers: normalizer, pii_redactor, vin_decoder, feedback_loop, knowledge_hierarchy
"""
import pytest
import json
from pathlib import Path


# =============================================================================
# NORMALIZER TESTS
# =============================================================================

class TestInputNormalizer:
    """Tests for noisy input normalization."""
    
    def test_basic_abbreviations(self):
        """Test common automotive abbreviations."""
        from automotive_intent.services.normalizer import get_normalizer
        
        normalizer = get_normalizer()
        text, meta = normalizer.normalize("frt lft brk noise")
        
        assert "front" in text.lower()
        assert "left" in text.lower()
        assert "brake" in text.lower()
        assert meta["changes_made"] >= 3
    
    def test_customer_abbreviations(self):
        """Test service advisor abbreviations."""
        from automotive_intent.services.normalizer import get_normalizer
        
        normalizer = get_normalizer()
        text, _ = normalizer.normalize("cus sts veh wont start")
        
        assert "customer" in text.lower()
        assert "states" in text.lower()
    
    def test_india_specific_terms(self):
        """Test Hindi/Hinglish terms."""
        from automotive_intent.services.normalizer import get_normalizer
        
        normalizer = get_normalizer()
        text, _ = normalizer.normalize("gaadi garam ho rahi hai")
        
        assert "vehicle" in text.lower()
        assert "hot" in text.lower()
    
    def test_no_changes_on_clean_input(self):
        """Test that clean input remains unchanged."""
        from automotive_intent.services.normalizer import get_normalizer
        
        normalizer = get_normalizer()
        original = "The engine is overheating and needs repair"
        text, meta = normalizer.normalize(original)
        
        assert meta["changes_made"] == 0
        assert text == original
    
    def test_preserves_technical_terms(self):
        """Test that DTC codes and technical terms are preserved."""
        from automotive_intent.services.normalizer import get_normalizer
        
        normalizer = get_normalizer()
        text, _ = normalizer.normalize("P0420 code showing, catalytic converter issue")
        
        assert "P0420" in text


# =============================================================================
# PII REDACTOR TESTS
# =============================================================================

class TestPIIRedactor:
    """Tests for PII detection and redaction."""
    
    def test_phone_redaction_indian(self):
        """Test Indian phone number redaction."""
        from automotive_intent.services.pii_redactor import get_pii_redactor
        
        redactor = get_pii_redactor()
        result = redactor.redact("Call customer at +91 9876543210")
        
        assert "[PHONE_REDACTED]" in result.redacted_text
        assert "9876543210" not in result.redacted_text
        assert result.redaction_count >= 1
    
    def test_email_redaction(self):
        """Test email address redaction."""
        from automotive_intent.services.pii_redactor import get_pii_redactor
        
        redactor = get_pii_redactor()
        result = redactor.redact("Contact: customer@example.com for updates")
        
        assert "[EMAIL_REDACTED]" in result.redacted_text
        assert "customer@example.com" not in result.redacted_text
    
    def test_aadhaar_redaction(self):
        """Test Aadhaar number redaction."""
        from automotive_intent.services.pii_redactor import get_pii_redactor
        
        redactor = get_pii_redactor()
        result = redactor.redact("Aadhaar: 1234 5678 9012")
        
        assert "[AADHAAR_REDACTED]" in result.redacted_text
    
    def test_multiple_pii_types(self):
        """Test redaction of multiple PII types in one text."""
        from automotive_intent.services.pii_redactor import get_pii_redactor
        
        redactor = get_pii_redactor()
        text = "Mr. Sharma, phone 9876543210, email test@mail.com"
        result = redactor.redact(text)
        
        assert result.redaction_count >= 2
    
    def test_no_pii_is_safe(self):
        """Test that text without PII is marked safe."""
        from automotive_intent.services.pii_redactor import get_pii_redactor
        
        redactor = get_pii_redactor()
        assert redactor.is_safe("Brake pads need replacement")
    
    def test_vin_masking(self):
        """Test partial VIN masking."""
        from automotive_intent.services.pii_redactor import get_pii_redactor
        
        redactor = get_pii_redactor()
        masked = redactor.mask_vin("1HGCM82633A123456")
        
        assert masked.startswith("1HG")
        assert masked.endswith("3456")
        assert "**********" in masked


# =============================================================================
# VIN DECODER TESTS
# =============================================================================

class TestVINDecoder:
    """Tests for VIN and registration number decoding."""
    
    def test_indian_registration(self):
        """Test Indian vehicle registration parsing."""
        from automotive_intent.services.vin_decoder import get_vin_decoder
        
        decoder = get_vin_decoder()
        info = decoder.decode("MH12AB1234")
        
        assert info is not None
        assert info.vin == "MH12AB1234"
    
    def test_standard_vin(self):
        """Test 17-character VIN parsing."""
        from automotive_intent.services.vin_decoder import get_vin_decoder
        
        decoder = get_vin_decoder()
        info = decoder.decode("1HGCM82633A123456")
        
        assert info is not None
        assert info.make == "Honda"
    
    def test_invalid_vin(self):
        """Test that invalid VIN returns None."""
        from automotive_intent.services.vin_decoder import get_vin_decoder
        
        decoder = get_vin_decoder()
        assert decoder.decode("INVALID") is None
        assert decoder.decode("ABC") is None
    
    def test_vehicle_filter_tags(self):
        """Test that filter tags are generated."""
        from automotive_intent.services.vin_decoder import get_vin_decoder
        
        decoder = get_vin_decoder()
        info = decoder.decode("1HGCM82633A123456")
        tags = info.get_filter_tags()
        
        assert isinstance(tags, list)


# =============================================================================
# FEEDBACK LOOP TESTS
# =============================================================================

class TestFeedbackLoop:
    """Tests for closed-loop learning feedback system."""
    
    def test_record_feedback(self, tmp_path):
        """Test recording feedback entry."""
        from automotive_intent.services.feedback_loop import FeedbackStore, FeedbackEntry
        
        store = FeedbackStore(storage_path=tmp_path / "feedback.json")
        
        entry = FeedbackEntry(
            ticket_id="TEST-001",
            was_correct=True,
            predicted_system="BRAKES",
            predicted_component="PADS_ROTORS",
            predicted_failure_mode="SQUEALING",
            predicted_confidence=0.9,
            original_complaint="Brake noise"
        )
        
        success = store.record_feedback(entry)
        assert success
    
    def test_accuracy_stats(self, tmp_path):
        """Test accuracy statistics calculation."""
        from automotive_intent.services.feedback_loop import FeedbackStore, FeedbackEntry
        
        store = FeedbackStore(storage_path=tmp_path / "feedback.json")
        
        # Add mix of correct and incorrect
        for i in range(3):
            store.record_feedback(FeedbackEntry(
                ticket_id=f"CORRECT-{i}",
                was_correct=True,
                predicted_system="BRAKES",
                predicted_component="PADS",
                predicted_failure_mode="NOISE",
                predicted_confidence=0.9,
                original_complaint="test"
            ))
        
        store.record_feedback(FeedbackEntry(
            ticket_id="INCORRECT-1",
            was_correct=False,
            predicted_system="ELECTRICAL",
            predicted_component="BATTERY",
            predicted_failure_mode="DEAD",
            predicted_confidence=0.8,
            original_complaint="test",
            actual_resolution="Replaced alternator"
        ))
        
        stats = store.get_accuracy_stats()
        
        assert stats.total_feedback == 4
        assert stats.correct_count == 3
        assert stats.incorrect_count == 1
        assert stats.accuracy_rate == 0.75


# =============================================================================
# KNOWLEDGE HIERARCHY TESTS
# =============================================================================

class TestKnowledgeHierarchy:
    """Tests for TSB override and document prioritization."""
    
    def test_tsb_supersedes_manual(self):
        """Test that TSB documents rank higher than manuals."""
        from automotive_intent.services.knowledge_hierarchy import get_knowledge_hierarchy
        
        hierarchy = get_knowledge_hierarchy()
        
        docs = [
            {"content": "Manual procedure", "metadata": {"source_type": "manual"}, "score": 0.9},
            {"content": "TSB procedure", "metadata": {"source_type": "tsb"}, "score": 0.8},
        ]
        
        ranked = hierarchy.rerank(docs)
        
        assert ranked[0]["metadata"]["source_type"] == "tsb"
    
    def test_recall_highest_priority(self):
        """Test that safety recalls have highest priority."""
        from automotive_intent.services.knowledge_hierarchy import get_knowledge_hierarchy
        
        hierarchy = get_knowledge_hierarchy()
        
        docs = [
            {"content": "TSB", "metadata": {"source_type": "tsb"}, "score": 0.95},
            {"content": "Recall", "metadata": {"source_type": "recall"}, "score": 0.7},
            {"content": "Manual", "metadata": {"source_type": "manual"}, "score": 0.9},
        ]
        
        ranked = hierarchy.rerank(docs)
        
        assert ranked[0]["metadata"]["source_type"] == "recall"
    
    def test_audit_trail(self):
        """Test that audit trail is generated."""
        from automotive_intent.services.knowledge_hierarchy import get_knowledge_hierarchy
        
        hierarchy = get_knowledge_hierarchy()
        docs = [{"content": "test", "metadata": {"source_type": "manual"}, "score": 0.5}]
        
        ranked = hierarchy.rerank(docs)
        audit = hierarchy.get_audit_trail(ranked)
        
        assert "retrieved_count" in audit
        assert "timestamp" in audit


# =============================================================================
# PARTS GRAPH TESTS
# =============================================================================

class TestPartsGraph:
    """Tests for parts dependency graph."""
    
    def test_graph_loads(self):
        """Test that parts graph JSON loads correctly."""
        graph_path = Path(__file__).parent.parent / "data" / "parts_graph.json"
        
        with open(graph_path) as f:
            graph = json.load(f)
        
        assert "water_pump" in graph
        assert "brake_pads_front" in graph
    
    def test_water_pump_dependencies(self):
        """Test water pump has correct mandatory parts."""
        graph_path = Path(__file__).parent.parent / "data" / "parts_graph.json"
        
        with open(graph_path) as f:
            graph = json.load(f)
        
        wp = graph["water_pump"]
        assert "water_pump_gasket" in wp["mandatory"]
        assert "coolant" in wp["mandatory"]
    
    def test_labor_notes_present(self):
        """Test that labor notes are included."""
        graph_path = Path(__file__).parent.parent / "data" / "parts_graph.json"
        
        with open(graph_path) as f:
            graph = json.load(f)
        
        assert "labor_note" in graph["water_pump"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
