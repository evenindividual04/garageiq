"""
Metrics Service
Tracks RAG retrieval quality, confidence distributions, and system performance.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict
from pydantic import BaseModel, Field
import statistics

logger = logging.getLogger(__name__)


class RetrievalMetric(BaseModel):
    """Metrics for a single RAG retrieval."""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    query: str
    num_results: int
    top_score: float
    avg_score: float
    retrieval_time_ms: float
    sources: List[str] = []


class ClassificationMetric(BaseModel):
    """Metrics for a single classification."""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    input_text: str
    detected_language: str
    predicted_system: Optional[str]
    predicted_component: Optional[str]
    predicted_failure_mode: Optional[str]
    confidence: float
    is_ambiguous: bool
    processing_time_ms: float
    agent_path: List[str] = []


class ConfidenceBucket(BaseModel):
    """Confidence distribution bucket."""
    range: str
    count: int
    percentage: float


class SystemMetrics(BaseModel):
    """Aggregated system metrics."""
    system: str
    total_classifications: int
    avg_confidence: float
    accuracy: Optional[float] = None


class DashboardData(BaseModel):
    """Full dashboard data."""
    # Summary
    total_requests: int
    requests_today: int
    avg_confidence: float
    avg_processing_time_ms: float
    
    # Confidence distribution
    confidence_distribution: List[ConfidenceBucket]
    
    # By system
    system_breakdown: List[SystemMetrics]
    
    # RAG metrics
    avg_retrieval_score: float
    total_retrievals: int
    
    # Language breakdown
    language_breakdown: Dict[str, int]
    
    # Recent activity
    recent_classifications: List[Dict[str, Any]]


class MetricsService:
    """
    Tracks and aggregates metrics for the diagnostic system.
    In-memory for simplicity (could be Prometheus/InfluxDB in production).
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._retrievals: List[RetrievalMetric] = []
        self._classifications: List[ClassificationMetric] = []
    
    def record_retrieval(
        self, 
        query: str, 
        results: List[Any],
        time_ms: float
    ):
        """Record a RAG retrieval operation."""
        scores = [r.score if hasattr(r, 'score') else 0.5 for r in results]
        sources = [r.metadata.get('source', 'unknown') if hasattr(r, 'metadata') else 'unknown' for r in results]
        
        metric = RetrievalMetric(
            query=query[:100],
            num_results=len(results),
            top_score=max(scores) if scores else 0.0,
            avg_score=statistics.mean(scores) if scores else 0.0,
            retrieval_time_ms=time_ms,
            sources=sources[:3]
        )
        
        self._retrievals.append(metric)
        self._trim_history()
    
    def record_classification(
        self,
        input_text: str,
        language: str,
        system: Optional[str],
        component: Optional[str],
        failure_mode: Optional[str],
        confidence: float,
        is_ambiguous: bool,
        time_ms: float,
        agent_path: List[str] = []
    ):
        """Record a classification operation."""
        metric = ClassificationMetric(
            input_text=input_text[:100],
            detected_language=language,
            predicted_system=system,
            predicted_component=component,
            predicted_failure_mode=failure_mode,
            confidence=confidence,
            is_ambiguous=is_ambiguous,
            processing_time_ms=time_ms,
            agent_path=agent_path
        )
        
        self._classifications.append(metric)
        self._trim_history()
    
    def _trim_history(self):
        """Keep history within limits."""
        if len(self._retrievals) > self.max_history:
            self._retrievals = self._retrievals[-self.max_history:]
        if len(self._classifications) > self.max_history:
            self._classifications = self._classifications[-self.max_history:]
    
    def get_dashboard(self) -> DashboardData:
        """Get full dashboard data."""
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Filter today's data
        today_classifications = [
            c for c in self._classifications
            if datetime.fromisoformat(c.timestamp) >= today_start
        ]
        
        # Confidence distribution
        confidence_buckets = self._calculate_confidence_distribution()
        
        # System breakdown
        system_breakdown = self._calculate_system_breakdown()
        
        # Language breakdown
        language_counts: Dict[str, int] = defaultdict(int)
        for c in self._classifications:
            language_counts[c.detected_language] += 1
        
        # Calculate averages
        confidences = [c.confidence for c in self._classifications]
        times = [c.processing_time_ms for c in self._classifications]
        retrieval_scores = [r.top_score for r in self._retrievals]
        
        return DashboardData(
            total_requests=len(self._classifications),
            requests_today=len(today_classifications),
            avg_confidence=statistics.mean(confidences) if confidences else 0.0,
            avg_processing_time_ms=statistics.mean(times) if times else 0.0,
            confidence_distribution=confidence_buckets,
            system_breakdown=system_breakdown,
            avg_retrieval_score=statistics.mean(retrieval_scores) if retrieval_scores else 0.0,
            total_retrievals=len(self._retrievals),
            language_breakdown=dict(language_counts),
            recent_classifications=[c.model_dump() for c in self._classifications[-10:]]
        )
    
    def _calculate_confidence_distribution(self) -> List[ConfidenceBucket]:
        """Calculate confidence distribution buckets."""
        buckets = {
            "0-20%": 0,
            "20-40%": 0,
            "40-60%": 0,
            "60-80%": 0,
            "80-100%": 0
        }
        
        for c in self._classifications:
            conf = c.confidence * 100
            if conf < 20:
                buckets["0-20%"] += 1
            elif conf < 40:
                buckets["20-40%"] += 1
            elif conf < 60:
                buckets["40-60%"] += 1
            elif conf < 80:
                buckets["60-80%"] += 1
            else:
                buckets["80-100%"] += 1
        
        total = len(self._classifications) or 1
        return [
            ConfidenceBucket(
                range=k,
                count=v,
                percentage=v / total * 100
            )
            for k, v in buckets.items()
        ]
    
    def _calculate_system_breakdown(self) -> List[SystemMetrics]:
        """Calculate metrics by system."""
        by_system: Dict[str, List[float]] = defaultdict(list)
        
        for c in self._classifications:
            if c.predicted_system:
                by_system[c.predicted_system].append(c.confidence)
        
        return [
            SystemMetrics(
                system=system,
                total_classifications=len(confidences),
                avg_confidence=statistics.mean(confidences)
            )
            for system, confidences in by_system.items()
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get quick summary metrics."""
        return {
            "total_classifications": len(self._classifications),
            "total_retrievals": len(self._retrievals),
            "avg_confidence": statistics.mean([c.confidence for c in self._classifications]) if self._classifications else 0
        }


# Singleton
_metrics_service: Optional[MetricsService] = None


def get_metrics_service() -> MetricsService:
    global _metrics_service
    if _metrics_service is None:
        _metrics_service = MetricsService()
    return _metrics_service
