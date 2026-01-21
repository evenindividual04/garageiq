"""
Confidence Calibration Service
Implements uncertainty quantification for classification decisions.
Uses ensemble-like approach with multiple evidence sources.
"""
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class CalibrationFactors:
    """Factors that affect confidence calibration."""
    llm_confidence: float
    rag_similarity: float
    keyword_match: float
    ontology_coverage: float
    input_clarity: float


class ConfidenceCalibrator:
    """
    Calibrates confidence scores using multiple signals.
    Implements Bayesian-inspired confidence estimation.
    """
    
    # Weights for different signals
    WEIGHTS = {
        "llm": 0.35,
        "rag": 0.25,
        "keywords": 0.15,
        "ontology": 0.15,
        "clarity": 0.10
    }
    
    # Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    LOW_CONFIDENCE_THRESHOLD = 0.40
    
    def calibrate(
        self,
        llm_confidence: float,
        similar_tickets: List[Dict] = None,
        knowledge_scores: List[float] = None,
        keywords_matched: int = 0,
        total_keywords: int = 1,
        input_length: int = 0
    ) -> Tuple[float, str, Dict]:
        """
        Calibrate confidence using multiple signals.
        
        Returns:
            Tuple of (calibrated_confidence, confidence_level, factors_dict)
        """
        
        # Calculate individual factors
        rag_similarity = self._calculate_rag_factor(similar_tickets)
        keyword_match = keywords_matched / max(total_keywords, 1)
        ontology_coverage = self._calculate_ontology_factor(llm_confidence)
        input_clarity = self._calculate_clarity_factor(input_length)
        
        factors = CalibrationFactors(
            llm_confidence=llm_confidence,
            rag_similarity=rag_similarity,
            keyword_match=keyword_match,
            ontology_coverage=ontology_coverage,
            input_clarity=input_clarity
        )
        
        # Weighted combination
        calibrated = (
            self.WEIGHTS["llm"] * factors.llm_confidence +
            self.WEIGHTS["rag"] * factors.rag_similarity +
            self.WEIGHTS["keywords"] * factors.keyword_match +
            self.WEIGHTS["ontology"] * factors.ontology_coverage +
            self.WEIGHTS["clarity"] * factors.input_clarity
        )
        
        # Apply uncertainty penalty for low RAG matches
        if rag_similarity < 0.5:
            calibrated *= 0.9  # 10% penalty
        
        # Clamp to [0, 1]
        calibrated = max(0.0, min(1.0, calibrated))
        
        # Determine level
        if calibrated >= self.HIGH_CONFIDENCE_THRESHOLD:
            level = "HIGH"
        elif calibrated >= self.LOW_CONFIDENCE_THRESHOLD:
            level = "MODERATE"
        else:
            level = "LOW"
        
        factors_dict = {
            "llm_confidence": round(factors.llm_confidence, 3),
            "rag_similarity": round(factors.rag_similarity, 3),
            "keyword_match": round(factors.keyword_match, 3),
            "ontology_coverage": round(factors.ontology_coverage, 3),
            "input_clarity": round(factors.input_clarity, 3),
            "calibrated": round(calibrated, 3),
            "level": level
        }
        
        logger.debug(f"Calibration: {factors_dict}")
        
        return calibrated, level, factors_dict
    
    def _calculate_rag_factor(self, similar_tickets: List[Dict]) -> float:
        """Calculate RAG contribution to confidence."""
        if not similar_tickets:
            return 0.3  # Low base if no RAG
        
        # Use top-k scores with decay
        scores = [t.get("similarity_score", 0) for t in similar_tickets[:5]]
        if not scores:
            return 0.3
        
        # Weighted average with position decay
        weighted_sum = 0
        weight_total = 0
        for i, score in enumerate(scores):
            weight = 1 / (i + 1)  # Position decay
            weighted_sum += score * weight
            weight_total += weight
        
        return weighted_sum / weight_total if weight_total > 0 else 0.3
    
    def _calculate_ontology_factor(self, llm_confidence: float) -> float:
        """
        Ontology factor - if LLM is confident in a valid path, boost it.
        This rewards the LLM for selecting from the strict ontology.
        """
        # If LLM gave high confidence, assume good ontology match
        if llm_confidence >= 0.8:
            return 0.9
        elif llm_confidence >= 0.6:
            return 0.7
        else:
            return 0.5
    
    def _calculate_clarity_factor(self, input_length: int) -> float:
        """
        Input clarity - longer, more descriptive inputs are clearer.
        Very short inputs are harder to classify accurately.
        """
        if input_length < 10:
            return 0.3  # Very short = unclear
        elif input_length < 30:
            return 0.6  # Short
        elif input_length < 100:
            return 0.85  # Good length
        else:
            return 0.75  # Very long might be rambling
    
    def get_uncertainty_band(self, calibrated: float) -> Tuple[float, float]:
        """
        Calculate uncertainty band (confidence interval).
        Returns (lower_bound, upper_bound).
        """
        # Uncertainty is higher for mid-range confidences
        base_uncertainty = 0.1
        
        # More uncertainty in the middle (epistemic uncertainty)
        distance_from_extreme = min(calibrated, 1 - calibrated)
        extra_uncertainty = distance_from_extreme * 0.1
        
        total_uncertainty = base_uncertainty + extra_uncertainty
        
        lower = max(0.0, calibrated - total_uncertainty)
        upper = min(1.0, calibrated + total_uncertainty)
        
        return (round(lower, 3), round(upper, 3))


# Singleton
_calibrator: Optional[ConfidenceCalibrator] = None


def get_confidence_calibrator() -> ConfidenceCalibrator:
    global _calibrator
    if _calibrator is None:
        _calibrator = ConfidenceCalibrator()
    return _calibrator
