"""
Prometheus Metrics Service
Provides production-grade observability metrics.
"""
import time
import logging
from typing import Dict, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Try to import prometheus_client, fallback to mock if not available
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Metrics will be mocked.")


class MetricsCollector:
    """
    Collects and exposes Prometheus metrics for the application.
    """
    
    def __init__(self, app_name: str = "automotive_intent"):
        self.app_name = app_name
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Initialize all Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            self._mock_metrics()
            return
        
        # Request metrics
        self.request_count = Counter(
            f'{self.app_name}_requests_total',
            'Total number of requests',
            ['endpoint', 'method', 'status']
        )
        
        self.request_latency = Histogram(
            f'{self.app_name}_request_latency_seconds',
            'Request latency in seconds',
            ['endpoint'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Classification metrics
        self.classifications_total = Counter(
            f'{self.app_name}_classifications_total',
            'Total classifications',
            ['status', 'system', 'mode']
        )
        
        self.classification_confidence = Histogram(
            f'{self.app_name}_classification_confidence',
            'Classification confidence distribution',
            ['system'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        self.classification_latency = Histogram(
            f'{self.app_name}_classification_latency_seconds',
            'Classification latency in seconds',
            ['mode'],
            buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
        )
        
        # LLM metrics
        self.llm_calls_total = Counter(
            f'{self.app_name}_llm_calls_total',
            'Total LLM API calls',
            ['model', 'success']
        )
        
        self.llm_latency = Histogram(
            f'{self.app_name}_llm_latency_seconds',
            'LLM call latency in seconds',
            ['model'],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )
        
        self.llm_tokens = Counter(
            f'{self.app_name}_llm_tokens_total',
            'Total LLM tokens used',
            ['model', 'type']  # type: prompt, completion
        )
        
        # RAG metrics
        self.rag_queries_total = Counter(
            f'{self.app_name}_rag_queries_total',
            'Total RAG queries',
            ['collection']
        )
        
        self.rag_similarity_score = Histogram(
            f'{self.app_name}_rag_similarity_score',
            'RAG similarity scores',
            ['collection'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        
        # Security metrics
        self.sanitization_warnings = Counter(
            f'{self.app_name}_sanitization_warnings_total',
            'Input sanitization warnings',
            ['warning_type']
        )
        
        self.injection_attempts = Counter(
            f'{self.app_name}_injection_attempts_total',
            'Potential prompt injection attempts blocked',
            ['pattern_type']
        )
        
        # System metrics
        self.active_sessions = Gauge(
            f'{self.app_name}_active_sessions',
            'Number of active sessions'
        )
        
        self.pipeline_ready = Gauge(
            f'{self.app_name}_pipeline_ready',
            'Whether the ML pipeline is ready (1=ready, 0=not ready)'
        )
        
        # Info metric
        self.app_info = Info(
            f'{self.app_name}_info',
            'Application information'
        )
        self.app_info.info({
            'version': '1.0.0',
            'model': 'mistral:latest',
            'embedding_model': 'all-MiniLM-L6-v2'
        })
    
    def _mock_metrics(self):
        """Create mock metrics when prometheus_client is not available."""
        class MockMetric:
            def labels(self, *args, **kwargs): return self
            def inc(self, *args, **kwargs): pass
            def dec(self, *args, **kwargs): pass
            def observe(self, *args, **kwargs): pass
            def set(self, *args, **kwargs): pass
            def info(self, *args, **kwargs): pass
        
        self.request_count = MockMetric()
        self.request_latency = MockMetric()
        self.classifications_total = MockMetric()
        self.classification_confidence = MockMetric()
        self.classification_latency = MockMetric()
        self.llm_calls_total = MockMetric()
        self.llm_latency = MockMetric()
        self.llm_tokens = MockMetric()
        self.rag_queries_total = MockMetric()
        self.rag_similarity_score = MockMetric()
        self.sanitization_warnings = MockMetric()
        self.injection_attempts = MockMetric()
        self.active_sessions = MockMetric()
        self.pipeline_ready = MockMetric()
        self.app_info = MockMetric()
    
    # Helper methods
    def record_request(self, endpoint: str, method: str, status: int, latency: float):
        """Record an API request."""
        self.request_count.labels(endpoint=endpoint, method=method, status=str(status)).inc()
        self.request_latency.labels(endpoint=endpoint).observe(latency)
    
    def record_classification(self, status: str, system: str, mode: str, confidence: float, latency: float):
        """Record a classification result."""
        self.classifications_total.labels(status=status, system=system, mode=mode).inc()
        self.classification_confidence.labels(system=system).observe(confidence)
        self.classification_latency.labels(mode=mode).observe(latency)
    
    def record_llm_call(self, model: str, success: bool, latency: float, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Record an LLM API call."""
        self.llm_calls_total.labels(model=model, success=str(success)).inc()
        self.llm_latency.labels(model=model).observe(latency)
        if prompt_tokens:
            self.llm_tokens.labels(model=model, type="prompt").inc(prompt_tokens)
        if completion_tokens:
            self.llm_tokens.labels(model=model, type="completion").inc(completion_tokens)
    
    def record_rag_query(self, collection: str, top_score: float):
        """Record a RAG query."""
        self.rag_queries_total.labels(collection=collection).inc()
        self.rag_similarity_score.labels(collection=collection).observe(top_score)
    
    def record_sanitization_warning(self, warning_type: str):
        """Record a sanitization warning."""
        self.sanitization_warnings.labels(warning_type=warning_type).inc()
    
    def record_injection_attempt(self, pattern_type: str):
        """Record a potential injection attempt."""
        self.injection_attempts.labels(pattern_type=pattern_type).inc()
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest()
        return b"# Prometheus client not installed\n"
    
    def get_content_type(self) -> str:
        """Get content type for metrics endpoint."""
        if PROMETHEUS_AVAILABLE:
            return CONTENT_TYPE_LATEST
        return "text/plain"


# Singleton
_metrics: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


# Decorator for timing functions
def timed(metric_name: str = "function"):
    """Decorator to time function execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start
                logger.debug(f"{metric_name} took {elapsed:.3f}s")
        return wrapper
    return decorator
