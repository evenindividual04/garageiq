"""
Observability module for GarageIQ
Provides request metrics, timing, and classification statistics.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)


@dataclass
class RequestMetric:
    """Single request metric."""
    request_id: str
    timestamp: str
    endpoint: str
    method: str
    status_code: int
    duration_ms: float
    classification_status: Optional[str] = None
    intent_system: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class MetricsStore:
    """In-memory metrics store for demo purposes."""
    requests: List[RequestMetric] = field(default_factory=list)
    max_entries: int = 1000
    
    # Aggregated stats
    _status_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _system_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _total_requests: int = 0
    _total_duration_ms: float = 0.0
    
    def record(self, metric: RequestMetric) -> None:
        """Record a request metric."""
        # Add to list
        self.requests.append(metric)
        
        # Trim if too many
        if len(self.requests) > self.max_entries:
            self.requests = self.requests[-self.max_entries:]
        
        # Update aggregates
        self._total_requests += 1
        self._total_duration_ms += metric.duration_ms
        
        if metric.classification_status:
            self._status_counts[metric.classification_status] += 1
        
        if metric.intent_system:
            self._system_counts[metric.intent_system] += 1
    
    def get_summary(self) -> Dict:
        """Get metrics summary."""
        avg_duration = self._total_duration_ms / max(self._total_requests, 1)
        
        return {
            "total_requests": self._total_requests,
            "avg_duration_ms": round(avg_duration, 2),
            "classification_breakdown": dict(self._status_counts),
            "system_breakdown": dict(self._system_counts),
            "recent_requests": len(self.requests)
        }
    
    def get_recent(self, limit: int = 10) -> List[Dict]:
        """Get recent requests."""
        recent = self.requests[-limit:]
        return [
            {
                "request_id": r.request_id,
                "timestamp": r.timestamp,
                "endpoint": r.endpoint,
                "status": r.classification_status,
                "duration_ms": r.duration_ms
            }
            for r in reversed(recent)
        ]
    
    def reset(self):
        """Reset all metrics."""
        self.requests.clear()
        self._status_counts.clear()
        self._system_counts.clear()
        self._total_requests = 0
        self._total_duration_ms = 0.0


# Global metrics instance
metrics = MetricsStore()


class RequestTimer:
    """Context manager for timing requests."""
    
    def __init__(self, request_id: str, endpoint: str, method: str = "POST"):
        self.request_id = request_id
        self.endpoint = endpoint
        self.method = method
        self.start_time: Optional[float] = None
        self.duration_ms: float = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            self.duration_ms = (time.perf_counter() - self.start_time) * 1000
    
    def record(
        self,
        status_code: int,
        classification_status: Optional[str] = None,
        intent_system: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """Record the completed request."""
        metric = RequestMetric(
            request_id=self.request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            endpoint=self.endpoint,
            method=self.method,
            status_code=status_code,
            duration_ms=round(self.duration_ms, 2),
            classification_status=classification_status,
            intent_system=intent_system,
            confidence=confidence
        )
        metrics.record(metric)
        
        # Log the metric
        logger.info(
            f"[{self.request_id}] {self.method} {self.endpoint} "
            f"status={status_code} duration={self.duration_ms:.2f}ms "
            f"classification={classification_status}"
        )


def log_classification_event(
    request_id: str,
    original_text: str,
    detected_language: str,
    classification_status: str,
    intent: Optional[Dict] = None,
    duration_ms: float = 0.0
):
    """Log a structured classification event."""
    event = {
        "event_type": "classification",
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": {
            "text_length": len(original_text),
            "language": detected_language
        },
        "output": {
            "status": classification_status,
            "intent": intent
        },
        "duration_ms": round(duration_ms, 2)
    }
    
    # Log as JSON for easy parsing by log aggregators
    logger.info(f"CLASSIFICATION_EVENT: {json.dumps(event)}")
