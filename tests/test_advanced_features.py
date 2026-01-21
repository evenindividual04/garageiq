"""
Unit tests for advanced RAG features.
"""
import pytest
import json


class TestNormalizer:
    """Tests for noisy input normalization."""
    
    def test_abbreviation_expansion(self):
        from automotive_intent.services.normalizer import get_normalizer
        
        normalizer = get_normalizer()
        text, meta = normalizer.normalize("Cus sts frt lft noise")
        
        assert "customer" in text.lower()
        assert "front" in text.lower()
        assert "left" in text.lower()
        assert meta["changes_made"] >= 3
    
    def test_india_specific_terms(self):
        from automotive_intent.services.normalizer import get_normalizer
        
        normalizer = get_normalizer()
        text, _ = normalizer.normalize("gaadi garam ho rahi")
        
        assert "vehicle" in text.lower()
        assert "hot" in text.lower()
    
    def test_no_changes_clean_input(self):
        from automotive_intent.services.normalizer import get_normalizer
        
        normalizer = get_normalizer()
        _, meta = normalizer.normalize("The engine is overheating")
        
        assert meta["changes_made"] == 0


class TestVINDecoder:
    """Tests for VIN/Registration decoder."""
    
    def test_indian_registration(self):
        from automotive_intent.services.vin_decoder import get_vin_decoder
        
        decoder = get_vin_decoder()
        info = decoder.decode("MH12AB1234")
        
        assert info is not None
        assert info.vin == "MH12AB1234"
    
    def test_invalid_vin(self):
        from automotive_intent.services.vin_decoder import get_vin_decoder
        
        decoder = get_vin_decoder()
        info = decoder.decode("INVALID")
        
        assert info is None


class TestPartsGraph:
    """Tests for parts dependency graph."""
    
    def test_water_pump_dependencies(self):
        with open("data/parts_graph.json") as f:
            graph = json.load(f)
        
        assert "water_pump" in graph
        assert "water_pump_gasket" in graph["water_pump"]["mandatory"]
        assert "coolant" in graph["water_pump"]["mandatory"]
    
    def test_brake_pads_recommendations(self):
        with open("data/parts_graph.json") as f:
            graph = json.load(f)
        
        assert "brake_pads_front" in graph
        assert "brake_rotors_front" in graph["brake_pads_front"]["recommended"]


class TestKnowledgeHierarchy:
    """Tests for TSB override logic."""
    
    def test_tsb_supersedes_manual(self):
        from automotive_intent.services.knowledge_hierarchy import get_knowledge_hierarchy
        
        hierarchy = get_knowledge_hierarchy()
        
        docs = [
            {"content": "Manual says X", "metadata": {"source_type": "manual"}},
            {"content": "TSB says Y", "metadata": {"source_type": "tsb"}},
        ]
        
        ranked = hierarchy.rerank(docs)
        
        # TSB should come first
        assert ranked[0]["metadata"]["source_type"] == "tsb"
    
    def test_recall_highest_priority(self):
        from automotive_intent.services.knowledge_hierarchy import get_knowledge_hierarchy
        
        hierarchy = get_knowledge_hierarchy()
        
        docs = [
            {"content": "TSB", "metadata": {"source_type": "tsb"}},
            {"content": "Recall", "metadata": {"source_type": "recall"}},
        ]
        
        ranked = hierarchy.rerank(docs)
        
        # Recall should come first
        assert ranked[0]["metadata"]["source_type"] == "recall"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
