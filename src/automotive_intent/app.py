"""
FastAPI Application for GarageIQ
Enterprise-grade REST API with structured logging and rate limiting.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import json
import time
import uuid
from collections import defaultdict
from datetime import datetime

from .config import config
from .core.schemas import ClassificationRequest, ServiceTicket
from .pipeline import IntentPipeline, create_pipeline


# Structured JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            log_entry["request_id"] = record.request_id
        if hasattr(record, "response_time_ms"):
            log_entry["response_time_ms"] = record.response_time_ms
        return json.dumps(log_entry)


# Setup logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# Global pipeline instance
_pipeline: IntentPipeline | None = None

# Rate limiting state (simple in-memory, use Redis in production)
_rate_limit_store: dict = defaultdict(list)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - load models on startup."""
    global _pipeline
    
    logger.info("=" * 50)
    logger.info("GarageIQ API Starting...")
    logger.info(f"Environment: {config.ENV}")
    logger.info(f"Ollama Enabled: {config.USE_OLLAMA} (Model: {config.OLLAMA_MODEL})")
    logger.info(f"NLLB Enabled: {config.USE_NLLB}")
    logger.info("=" * 50)
    
    # Initialize pipeline with real models
    _pipeline = create_pipeline(
        use_ollama=config.USE_OLLAMA,
        use_nllb=config.USE_NLLB
    )
    
    logger.info("Pipeline initialized successfully!")
    
    # Warm up models for faster first request
    try:
        from .services.warmup import warmup_all
        warmup_results = warmup_all()
        logger.info(f"Model warmup results: {warmup_results}")
    except Exception as e:
        logger.warning(f"Model warmup failed (non-critical): {e}")
    
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="GarageIQ API",
    description="Multilingual intent normalization for automotive service complaints. "
                "Converts natural language complaints into ontology-validated service tickets.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware for tracing
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{process_time:.3f}s"
    
    logger.info(f"[{request_id}] {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    
    return response


# Rate limiting middleware (100 requests per minute per IP)
RATE_LIMIT = 100
RATE_WINDOW = 60  # seconds

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    
    # Clean old entries
    _rate_limit_store[client_ip] = [t for t in _rate_limit_store[client_ip] if now - t < RATE_WINDOW]
    
    # Check limit
    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    # Record request
    _rate_limit_store[client_ip].append(now)
    
    response = await call_next(request)
    response.headers["X-Rate-Limit-Limit"] = str(RATE_LIMIT)
    response.headers["X-Rate-Limit-Remaining"] = str(RATE_LIMIT - len(_rate_limit_store[client_ip]))
    
    return response


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns service status and model availability.
    """
    return {
        "status": "healthy",
        "pipeline_loaded": _pipeline is not None,
        "config": {
            "ollama_enabled": config.USE_OLLAMA,
            "nllb_enabled": config.USE_NLLB,
            "model": config.OLLAMA_MODEL
        }
    }


@app.get("/")
async def root():
    """Root endpoint - API info."""
    return {
        "name": "GarageIQ API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "classify": "POST /v1/classify"
    }


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Return 204 No Content for favicon requests."""
    from starlette.responses import Response
    return Response(status_code=204)


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    Scrape this endpoint to collect application metrics.
    """
    from starlette.responses import Response
    from .services.prometheus import get_metrics_collector
    
    collector = get_metrics_collector()
    return Response(
        content=collector.get_metrics(),
        media_type=collector.get_content_type()
    )


@app.post("/v1/classify", response_model=ServiceTicket)
async def classify_complaint(request: ClassificationRequest) -> ServiceTicket:
    """
    Classify a customer complaint into a structured service ticket.
    
    **Features:**
    - Accepts multilingual input (English, Hindi, Hinglish)
    - Automatically detects language and translates
    - Returns ontology-validated JSON with confidence scores
    - Explicitly handles ambiguity (won't force incorrect classifications)
    - **Input sanitization** protects against prompt injection
    
    **Response Statuses:**
    - `CONFIRMED`: High-confidence match to ontology
    - `AMBIGUOUS`: Low confidence or multiple candidates
    - `OUT_OF_SCOPE`: Not an automotive service complaint
    - `SYSTEM_ERROR`: Processing failed
    """
    import time
    start_time = time.time()
    
    # Import services
    from .services.sanitizer import get_sanitizer
    from .services.prometheus import get_metrics_collector
    
    sanitizer = get_sanitizer()
    metrics = get_metrics_collector()
    
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized. Check model availability."
        )
    
    # Sanitize input
    sanitization_result = sanitizer.sanitize(request.text)
    
    # Record sanitization warnings
    for warning in sanitization_result.warnings:
        warning_type = warning.split(":")[0] if ":" in warning else "unknown"
        metrics.record_sanitization_warning(warning_type)
    
    # Block if not safe
    if not sanitization_result.is_safe:
        metrics.record_injection_attempt("blocked")
        logger.warning(f"Blocked unsafe input: risk_score={sanitization_result.risk_score}")
        raise HTTPException(
            status_code=400,
            detail="Input contains potentially unsafe content. Please rephrase your query."
        )
    
    # Use sanitized text
    sanitized_request = ClassificationRequest(text=sanitization_result.sanitized_text)
    
    try:
        ticket = _pipeline.process(sanitized_request)
        
        # Record metrics
        latency = time.time() - start_time
        metrics.record_classification(
            status=ticket.classification_status,
            system=ticket.intent.system if ticket.intent else "UNKNOWN",
            mode="fast",
            confidence=ticket.intent.confidence if ticket.intent else 0.0,
            latency=latency
        )
        
        return ticket
    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@app.get("/v1/ontology")
async def get_ontology():
    """
    Return the service ontology for reference.
    Useful for UI dropdowns or validation on client side.
    """
    from .core.ontology import SERVICE_ONTOLOGY
    return {
        "ontology": SERVICE_ONTOLOGY,
        "description": "Valid system -> component -> failure_mode paths"
    }


@app.get("/v1/config")
async def get_config():
    """Return current configuration (non-sensitive)."""
    return {
        "environment": config.ENV,
        "ollama_model": config.OLLAMA_MODEL,
        "nllb_model": config.NLLB_MODEL if config.USE_NLLB else None,
        "thresholds": {
            "confidence": config.CONFIDENCE_THRESHOLD,
            "ambiguity_delta": config.AMBIGUITY_DELTA
        }
    }


@app.get("/v1/metrics")
async def get_metrics():
    """Return classification metrics and statistics."""
    from .observability import metrics
    return {
        "summary": metrics.get_summary(),
        "recent": metrics.get_recent(10)
    }


# ===== Shop Performance Analytics =====

@app.get("/v1/analytics")
async def get_analytics():
    """
    Shop Performance Dashboard Data.
    
    Returns aggregated metrics for service network monitoring:
    - Classification counts by status and system
    - Accuracy trends from feedback
    - Latency percentiles
    - Top issues
    - 7-day volume trend
    """
    from .services.analytics import get_analytics_service
    
    service = get_analytics_service()
    snapshot = service.get_snapshot()
    
    return {
        "timestamp": snapshot.timestamp,
        "classifications": {
            "total": snapshot.total_classifications,
            "confirmed": snapshot.confirmed_count,
            "ambiguous": snapshot.ambiguous_count,
            "out_of_scope": snapshot.out_of_scope_count
        },
        "accuracy": {
            "rate": snapshot.accuracy_rate,
            "feedback_count": snapshot.feedback_count
        },
        "performance": {
            "avg_latency_ms": snapshot.avg_latency_ms,
            "p95_latency_ms": snapshot.p95_latency_ms
        },
        "by_system": snapshot.classifications_by_system,
        "top_issues": snapshot.top_issues,
        "daily_trend": snapshot.daily_counts
    }


@app.get("/v1/vmrs/codes")
async def get_vmrs_codes():
    """
    Return VMRS system codes for reference.
    
    VMRS (Vehicle Maintenance Reporting Standards) is the industry-standard
    coding system used by fleets, OEMs, and service networks.
    """
    from .services.vmrs_codes import get_vmrs_mapper
    
    mapper = get_vmrs_mapper()
    return {
        "systems": mapper.get_all_system_codes(),
        "description": "VMRS codes auto-included in classification responses"
    }


# ===== Feedback Loop Endpoints (Closed-Loop Learning) =====

from pydantic import BaseModel
from typing import Optional

class FeedbackRequest(BaseModel):
    """Request to submit diagnosis feedback."""
    ticket_id: str
    was_correct: bool
    actual_resolution: Optional[str] = None
    actual_system: Optional[str] = None
    actual_component: Optional[str] = None
    actual_failure_mode: Optional[str] = None
    original_complaint: str
    predicted_system: str
    predicted_component: str
    predicted_failure_mode: str
    predicted_confidence: float
    technician_notes: Optional[str] = None


@app.post("/v1/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit technician feedback on a diagnosis.
    
    This enables closed-loop learning:
    - Correct diagnoses boost model confidence
    - Incorrect diagnoses are logged for improvement
    
    **Business Value**: Turns static AI into self-improving system.
    """
    from .services.feedback_loop import get_feedback_store, FeedbackEntry
    
    store = get_feedback_store()
    
    entry = FeedbackEntry(
        ticket_id=request.ticket_id,
        was_correct=request.was_correct,
        actual_resolution=request.actual_resolution,
        actual_system=request.actual_system,
        actual_component=request.actual_component,
        actual_failure_mode=request.actual_failure_mode,
        original_complaint=request.original_complaint,
        predicted_system=request.predicted_system,
        predicted_component=request.predicted_component,
        predicted_failure_mode=request.predicted_failure_mode,
        predicted_confidence=request.predicted_confidence,
        technician_notes=request.technician_notes
    )
    
    success = store.record_feedback(entry)
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to record feedback")
    
    return {
        "status": "recorded",
        "ticket_id": request.ticket_id,
        "was_correct": request.was_correct
    }


@app.get("/v1/feedback/stats")
async def get_feedback_stats():
    """
    Get aggregated accuracy statistics.
    
    Returns:
    - Overall accuracy rate
    - Accuracy by system
    - Top misdiagnoses (for targeted improvement)
    - Recent trend (7-day window)
    """
    from .services.feedback_loop import get_feedback_store
    
    store = get_feedback_store()
    stats = store.get_accuracy_stats()
    
    return stats.model_dump()


# ===== Multi-Agent Chat Endpoints =====

@app.post("/v1/agent/chat")
async def agent_chat(request: Request):
    """
    Multi-agent diagnostic chat endpoint.
    Processes messages through the agent orchestration workflow.
    """
    from .agents.orchestrator import get_orchestrator
    from .agents.state import ChatRequest
    
    body = await request.json()
    chat_request = ChatRequest(
        session_id=body.get("session_id"),
        message=body.get("message", "")
    )
    
    if not chat_request.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")
    
    orchestrator = get_orchestrator()
    response = orchestrator.process_message(chat_request)
    
    return {
        "session_id": response.session_id,
        "message": response.message,
        "agent": response.agent,
        "is_complete": response.is_complete,
        "confidence": response.confidence,
        "diagnosis": response.diagnosis.model_dump() if response.diagnosis else None,
        "similar_tickets": [t.model_dump() for t in response.similar_tickets]
    }


@app.get("/v1/agent/session/{session_id}")
async def get_agent_session(session_id: str):
    """Get the state of an agent session."""
    from .agents.orchestrator import get_orchestrator
    
    orchestrator = get_orchestrator()
    state = orchestrator.get_session(session_id)
    
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": state.session_id,
        "messages": [m.model_dump() for m in state.messages],
        "diagnosis": state.final_diagnosis.model_dump() if state.final_diagnosis else None,
        "confidence": state.overall_confidence,
        "is_complete": state.is_complete
    }


@app.delete("/v1/agent/session/{session_id}")
async def clear_agent_session(session_id: str):
    """Clear an agent session."""
    from .agents.orchestrator import get_orchestrator
    
    orchestrator = get_orchestrator()
    orchestrator.clear_session(session_id)
    
    return {"status": "cleared"}


# ===== Feedback Endpoints =====

@app.post("/v1/feedback")
async def submit_feedback(request: Request):
    """
    Submit human feedback on a classification.
    Used for active learning and model improvement.
    """
    from .services.feedback import get_feedback_service, FeedbackRequest
    
    body = await request.json()
    feedback_request = FeedbackRequest(**body)
    
    service = get_feedback_service()
    record = service.submit_feedback(feedback_request)
    
    return {
        "id": record.id,
        "was_correct": record.was_correct,
        "message": "Feedback recorded. Thank you for helping improve the system!"
    }


@app.get("/v1/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics and accuracy metrics."""
    from .services.feedback import get_feedback_service
    
    service = get_feedback_service()
    stats = service.get_stats()
    
    return stats.model_dump()


@app.get("/v1/feedback")
async def get_feedback_history(limit: int = 50):
    """Get recent feedback records."""
    from .services.feedback import get_feedback_service
    
    service = get_feedback_service()
    records = service.get_all(limit=limit)
    
    return {"records": [r.model_dump() for r in records]}


# ===== Dashboard & Metrics Endpoints =====

@app.get("/v1/dashboard")
async def get_dashboard():
    """
    Get full dashboard data including:
    - Confidence distributions
    - System breakdown
    - RAG metrics
    - Recent activity
    """
    from .services.metrics import get_metrics_service
    
    service = get_metrics_service()
    return service.get_dashboard().model_dump()


# ===== Entity Extraction Endpoints =====

@app.post("/v1/extract")
async def extract_entities(request: Request):
    """
    Extract entities from text:
    - Vehicle make/model/year
    - DTC codes (P0300, etc.)
    """
    from .services.entities import get_entity_extractor
    
    body = await request.json()
    text = body.get("text", "")
    
    extractor = get_entity_extractor()
    result = extractor.extract_all(text)
    
    return {
        "vehicle": result["vehicle"].model_dump() if result["vehicle"] else None,
        "dtc_codes": [c.model_dump() for c in result["dtc_codes"]]
    }


@app.get("/v1/dtc/{code}")
async def lookup_dtc_code(code: str):
    """Look up a specific DTC code."""
    from .services.entities import DTC_DATABASE
    
    code = code.upper()
    if code in DTC_DATABASE:
        desc, system = DTC_DATABASE[code]
        return {
            "code": code,
            "description": desc,
            "system": system,
            "found": True
        }
    
    return {
        "code": code,
        "description": "Code not in database",
        "system": "UNKNOWN",
        "found": False
    }


# ===== Streaming Endpoint (SSE) =====

@app.get("/v1/agent/stream/{session_id}")
async def stream_agent_status(session_id: str):
    """
    Server-Sent Events endpoint for streaming agent status.
    Shows real-time agent thinking process.
    """
    from fastapi.responses import StreamingResponse
    import asyncio
    
    async def event_generator():
        from .agents.orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        
        # Simulate agent steps (in real implementation, would track actual progress)
        steps = [
            {"agent": "symptom_analyst", "status": "Extracting symptoms..."},
            {"agent": "historian", "status": "Searching similar tickets..."},
            {"agent": "knowledge", "status": "Retrieving technical docs..."},
            {"agent": "diagnosis", "status": "Generating diagnosis..."},
            {"agent": "complete", "status": "Done!"}
        ]
        
        for step in steps:
            yield f"data: {json.dumps(step)}\n\n"
            await asyncio.sleep(0.5)
    
    import json
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "automotive_intent.app:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.ENV == "development"
    )
