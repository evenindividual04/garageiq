"""
Shop Performance Analytics
Provides aggregated metrics for service network performance monitoring.
"""
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsSnapshot:
    """Current analytics state."""
    timestamp: str
    
    # Classification metrics
    total_classifications: int = 0
    confirmed_count: int = 0
    ambiguous_count: int = 0
    out_of_scope_count: int = 0
    
    # Accuracy (from feedback)
    accuracy_rate: float = 0.0
    feedback_count: int = 0
    
    # Performance
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    
    # By system breakdown
    classifications_by_system: Dict[str, int] = field(default_factory=dict)
    
    # Top issues (most common)
    top_issues: List[Dict] = field(default_factory=list)
    
    # Trend (7-day)
    daily_counts: List[Dict] = field(default_factory=list)


class AnalyticsService:
    """
    Aggregates metrics for shop performance dashboard.
    
    Combines data from:
    - Classification logs
    - Feedback store
    - Prometheus metrics
    """
    
    def __init__(self, data_path: Path = None):
        self.data_path = data_path or Path(__file__).parent.parent.parent.parent / "data"
        self._metrics_log: List[dict] = []
    
    def record_classification(
        self,
        status: str,
        system: Optional[str],
        latency_ms: float,
        vmrs_code: Optional[str] = None
    ):
        """Record a classification event."""
        self._metrics_log.append({
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "system": system,
            "latency_ms": latency_ms,
            "vmrs_code": vmrs_code
        })
        
        # Keep last 1000 for in-memory analytics
        if len(self._metrics_log) > 1000:
            self._metrics_log = self._metrics_log[-1000:]
    
    def get_snapshot(self) -> AnalyticsSnapshot:
        """Get current analytics snapshot."""
        now = datetime.now()
        
        # Count by status
        confirmed = sum(1 for m in self._metrics_log if m.get("status") == "CONFIRMED")
        ambiguous = sum(1 for m in self._metrics_log if m.get("status") == "AMBIGUOUS")
        out_of_scope = sum(1 for m in self._metrics_log if m.get("status") == "OUT_OF_SCOPE")
        
        # By system
        by_system = defaultdict(int)
        for m in self._metrics_log:
            if m.get("system"):
                by_system[m["system"]] += 1
        
        # Latency
        latencies = [m["latency_ms"] for m in self._metrics_log if "latency_ms" in m]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if len(latencies) > 10 else avg_latency
        
        # Top issues
        issue_counts = defaultdict(int)
        for m in self._metrics_log:
            if m.get("system"):
                issue_counts[m["system"]] += 1
        top_issues = [
            {"system": k, "count": v}
            for k, v in sorted(issue_counts.items(), key=lambda x: -x[1])[:5]
        ]
        
        # Daily trend (last 7 days)
        daily_counts = []
        for i in range(7):
            day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
            count = sum(1 for m in self._metrics_log if m.get("timestamp", "").startswith(day))
            daily_counts.append({"date": day, "count": count})
        
        # Get feedback accuracy
        try:
            from .feedback_loop import get_feedback_store
            feedback_store = get_feedback_store()
            stats = feedback_store.get_accuracy_stats()
            accuracy = stats.accuracy_rate
            feedback_count = stats.total_feedback
        except:
            accuracy = 0.0
            feedback_count = 0
        
        return AnalyticsSnapshot(
            timestamp=now.isoformat(),
            total_classifications=len(self._metrics_log),
            confirmed_count=confirmed,
            ambiguous_count=ambiguous,
            out_of_scope_count=out_of_scope,
            accuracy_rate=accuracy,
            feedback_count=feedback_count,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            classifications_by_system=dict(by_system),
            top_issues=top_issues,
            daily_counts=daily_counts
        )


# Singleton
_analytics: AnalyticsService | None = None

def get_analytics_service() -> AnalyticsService:
    global _analytics
    if _analytics is None:
        _analytics = AnalyticsService()
    return _analytics
