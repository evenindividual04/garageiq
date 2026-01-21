"""
Agent State and Schemas
Defines the shared state for multi-agent orchestration.
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class Message(BaseModel):
    """A message in the conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    agent: Optional[str] = None  # Which agent generated this


class SymptomExtraction(BaseModel):
    """Extracted symptoms from user input."""
    raw_complaint: str
    detected_language: str
    keywords: List[str]
    location: Optional[str] = None  # front, rear, engine, etc.
    condition: Optional[str] = None  # when starting, at speed, etc.
    noise_type: Optional[str] = None  # squealing, grinding, clicking


class SimilarTicket(BaseModel):
    """A similar historical ticket."""
    ticket_id: str
    complaint: str
    system: str
    component: str
    failure_mode: str
    resolution: str
    similarity_score: float


class KnowledgeContext(BaseModel):
    """Retrieved knowledge context."""
    source: str
    title: str
    content: str
    relevance_score: float


class DiagnosisCandidate(BaseModel):
    """A potential diagnosis."""
    system: str
    component: str
    failure_mode: str
    confidence: float
    reasoning: str
    supporting_evidence: List[str] = []


class AgentState(BaseModel):
    """
    Shared state for the multi-agent system.
    This is passed between agents in the LangGraph workflow.
    """
    # Session info
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Conversation
    messages: List[Message] = []
    
    # Current processing
    current_input: str = ""
    detected_language: str = "en"
    
    # Symptom extraction
    symptoms: Optional[SymptomExtraction] = None
    
    # Retrieved context
    similar_tickets: List[SimilarTicket] = []
    knowledge_context: List[KnowledgeContext] = []
    
    # Diagnosis
    candidates: List[DiagnosisCandidate] = []
    final_diagnosis: Optional[DiagnosisCandidate] = None
    
    # Flow control
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    current_agent: str = "router"
    next_agent: Optional[str] = None
    is_complete: bool = False
    
    # Confidence tracking
    overall_confidence: float = 0.0
    confidence_history: List[Dict[str, float]] = []
    
    # Reflection loop state
    loop_count: int = 0
    refined_query: Optional[str] = None
    
    # Advanced Features Integration
    retrieval_audit: Optional[Dict[str, Any]] = None  # Knowledge hierarchy audit trail
    parts_dependencies: Optional[Dict[str, list]] = None  # Parts graph lookup result
    vehicle_info: Optional[Dict[str, Any]] = None  # VIN decoder result


class ChatRequest(BaseModel):
    """Request for agent chat endpoint."""
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    """Response from agent chat endpoint."""
    session_id: str
    message: str
    agent: str
    is_complete: bool
    diagnosis: Optional[DiagnosisCandidate] = None
    similar_tickets: List[SimilarTicket] = []
    confidence: float
    needs_input: bool = True
