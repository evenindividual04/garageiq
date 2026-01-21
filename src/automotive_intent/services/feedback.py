"""
Feedback Service
Captures human corrections to improve future predictions.
Implements active learning pattern.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid

logger = logging.getLogger(__name__)


class FeedbackRecord(BaseModel):
    """A single piece of human feedback."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Original input
    original_text: str
    detected_language: str = "en"
    
    # What the system predicted
    predicted_system: Optional[str] = None
    predicted_component: Optional[str] = None
    predicted_failure_mode: Optional[str] = None
    predicted_confidence: float = 0.0
    
    # Human correction
    correct_system: str
    correct_component: str
    correct_failure_mode: str
    
    # Was it correct?
    was_correct: bool = False
    
    # Additional context
    session_id: Optional[str] = None
    user_notes: Optional[str] = None


class FeedbackRequest(BaseModel):
    """Request to submit feedback."""
    original_text: str
    session_id: Optional[str] = None
    
    # What we predicted (optional, can be looked up)
    predicted_system: Optional[str] = None
    predicted_component: Optional[str] = None
    predicted_failure_mode: Optional[str] = None
    
    # The correct answer
    correct_system: str
    correct_component: str
    correct_failure_mode: str
    
    # Optional notes
    notes: Optional[str] = None


class FeedbackStats(BaseModel):
    """Statistics about feedback."""
    total_feedback: int
    correct_predictions: int
    incorrect_predictions: int
    accuracy: float
    common_corrections: List[Dict[str, Any]]
    feedback_by_system: Dict[str, int]


class FeedbackService:
    """
    Manages feedback collection and learning.
    Stores feedback as JSON for simplicity (could be database in production).
    """
    
    def __init__(self, feedback_file: str = "./data/feedback.json"):
        self.feedback_file = Path(feedback_file)
        self._feedback: List[FeedbackRecord] = []
        self._load()
    
    def _load(self):
        """Load existing feedback from file."""
        if self.feedback_file.exists():
            try:
                with open(self.feedback_file) as f:
                    data = json.load(f)
                self._feedback = [FeedbackRecord(**r) for r in data]
                logger.info(f"Loaded {len(self._feedback)} feedback records")
            except Exception as e:
                logger.warning(f"Could not load feedback: {e}")
                self._feedback = []
        else:
            self._feedback = []
    
    def _save(self):
        """Persist feedback to file."""
        self.feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.feedback_file, "w") as f:
            json.dump([r.model_dump() for r in self._feedback], f, indent=2)
    
    def submit_feedback(self, request: FeedbackRequest) -> FeedbackRecord:
        """Submit human feedback on a prediction."""
        was_correct = (
            request.predicted_system == request.correct_system and
            request.predicted_component == request.correct_component and
            request.predicted_failure_mode == request.correct_failure_mode
        )
        
        record = FeedbackRecord(
            original_text=request.original_text,
            predicted_system=request.predicted_system,
            predicted_component=request.predicted_component,
            predicted_failure_mode=request.predicted_failure_mode,
            correct_system=request.correct_system,
            correct_component=request.correct_component,
            correct_failure_mode=request.correct_failure_mode,
            was_correct=was_correct,
            session_id=request.session_id,
            user_notes=request.notes
        )
        
        self._feedback.append(record)
        self._save()
        
        logger.info(f"Feedback submitted: {record.id} (correct={was_correct})")
        return record
    
    def get_stats(self) -> FeedbackStats:
        """Get feedback statistics."""
        total = len(self._feedback)
        correct = sum(1 for f in self._feedback if f.was_correct)
        incorrect = total - correct
        
        # Count by system
        by_system: Dict[str, int] = {}
        corrections: Dict[tuple, int] = {}
        
        for f in self._feedback:
            by_system[f.correct_system] = by_system.get(f.correct_system, 0) + 1
            
            if not f.was_correct and f.predicted_system:
                key = (f.predicted_system, f.correct_system)
                corrections[key] = corrections.get(key, 0) + 1
        
        # Top corrections
        top_corrections = sorted(corrections.items(), key=lambda x: -x[1])[:5]
        common_corrections = [
            {"from": k[0], "to": k[1], "count": v}
            for k, v in top_corrections
        ]
        
        return FeedbackStats(
            total_feedback=total,
            correct_predictions=correct,
            incorrect_predictions=incorrect,
            accuracy=correct / total if total > 0 else 0.0,
            common_corrections=common_corrections,
            feedback_by_system=by_system
        )
    
    def get_examples_for_system(self, system: str, limit: int = 5) -> List[FeedbackRecord]:
        """Get feedback examples for a specific system (for few-shot learning)."""
        return [
            f for f in self._feedback 
            if f.correct_system == system and not f.was_correct
        ][:limit]
    
    def get_all(self, limit: int = 100) -> List[FeedbackRecord]:
        """Get recent feedback records."""
        return self._feedback[-limit:]


# Singleton
_feedback_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    """Get or create feedback service instance."""
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service
