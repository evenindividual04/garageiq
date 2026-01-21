"""
Closed-Loop Learning Service
Captures technician feedback to enable continuous improvement.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from collections import defaultdict

logger = logging.getLogger(__name__)


class FeedbackEntry(BaseModel):
    """Individual feedback entry from technician."""
    ticket_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # Original diagnosis
    predicted_system: str
    predicted_component: str
    predicted_failure_mode: str
    predicted_confidence: float
    
    # Feedback
    was_correct: bool
    actual_resolution: Optional[str] = None  # What the tech actually did
    actual_system: Optional[str] = None
    actual_component: Optional[str] = None
    actual_failure_mode: Optional[str] = None
    
    # Context
    original_complaint: str
    technician_notes: Optional[str] = None


class AccuracyStats(BaseModel):
    """Aggregated accuracy statistics."""
    total_feedback: int
    correct_count: int
    incorrect_count: int
    accuracy_rate: float
    
    # Breakdown by system
    by_system: Dict[str, Dict[str, int]] = {}
    
    # Common misdiagnoses
    top_misdiagnoses: List[Dict[str, str]] = []
    
    # Trend (last 7 days)
    recent_accuracy: float = 0.0


class FeedbackStore:
    """
    JSON-backed feedback storage with analytics.
    
    In production, this would be a database. For demo, JSON is sufficient.
    """
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or Path(__file__).parent.parent.parent.parent / "data" / "feedback.json"
        self._ensure_file()
    
    def _ensure_file(self):
        """Create storage file if it doesn't exist."""
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.storage_path.write_text("[]")
    
    def _load(self) -> List[dict]:
        """Load all feedback entries."""
        try:
            return json.loads(self.storage_path.read_text())
        except:
            return []
    
    def _save(self, entries: List[dict]):
        """Save all feedback entries."""
        self.storage_path.write_text(json.dumps(entries, indent=2))
    
    def record_feedback(self, entry: FeedbackEntry) -> bool:
        """
        Record a new feedback entry.
        
        Returns:
            True if successful
        """
        try:
            entries = self._load()
            entries.append(entry.model_dump())
            self._save(entries)
            
            logger.info(f"Recorded feedback for {entry.ticket_id}: {'correct' if entry.was_correct else 'incorrect'}")
            return True
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
            return False
    
    def get_accuracy_stats(self) -> AccuracyStats:
        """Calculate accuracy statistics."""
        entries = self._load()
        
        if not entries:
            return AccuracyStats(
                total_feedback=0,
                correct_count=0,
                incorrect_count=0,
                accuracy_rate=0.0
            )
        
        correct = sum(1 for e in entries if e.get("was_correct"))
        incorrect = len(entries) - correct
        
        # By system breakdown
        by_system = defaultdict(lambda: {"correct": 0, "incorrect": 0})
        for e in entries:
            system = e.get("predicted_system", "UNKNOWN")
            if e.get("was_correct"):
                by_system[system]["correct"] += 1
            else:
                by_system[system]["incorrect"] += 1
        
        # Top misdiagnoses
        misdiagnoses = []
        for e in entries:
            if not e.get("was_correct") and e.get("actual_failure_mode"):
                misdiagnoses.append({
                    "predicted": f"{e.get('predicted_system')}/{e.get('predicted_failure_mode')}",
                    "actual": f"{e.get('actual_system', 'UNK')}/{e.get('actual_failure_mode')}",
                })
        
        # Recent accuracy (last 7 days)
        recent_cutoff = (datetime.now().timestamp() - 7 * 24 * 3600)
        recent_entries = [
            e for e in entries 
            if datetime.fromisoformat(e.get("timestamp", "2000-01-01")).timestamp() > recent_cutoff
        ]
        recent_correct = sum(1 for e in recent_entries if e.get("was_correct"))
        recent_accuracy = recent_correct / len(recent_entries) if recent_entries else 0.0
        
        return AccuracyStats(
            total_feedback=len(entries),
            correct_count=correct,
            incorrect_count=incorrect,
            accuracy_rate=correct / len(entries),
            by_system=dict(by_system),
            top_misdiagnoses=misdiagnoses[:5],
            recent_accuracy=recent_accuracy
        )
    
    def get_corrections_for_retraining(self) -> List[dict]:
        """
        Get incorrect diagnoses for potential retraining or knowledge update.
        
        Returns entries where technician provided actual resolution.
        """
        entries = self._load()
        return [
            e for e in entries 
            if not e.get("was_correct") and e.get("actual_resolution")
        ]


# Singleton
_store: FeedbackStore | None = None

def get_feedback_store() -> FeedbackStore:
    global _store
    if _store is None:
        _store = FeedbackStore()
    return _store
