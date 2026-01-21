"""
Individual Agents for Diagnostic Workflow
Each agent has a specific role in the multi-agent system.
"""
import json
import logging
from typing import Optional, List
from langchain_core.messages import HumanMessage, SystemMessage

from .state import (
    AgentState, Message, SymptomExtraction, 
    SimilarTicket, KnowledgeContext, DiagnosisCandidate
)
from ..services.embeddings import get_embedding_service
from ..core.ontology import SERVICE_ONTOLOGY, get_ontology_formatted

logger = logging.getLogger(__name__)


class SymptomAnalystAgent:
    """
    Extracts and clarifies symptoms from user input.
    Detects language and identifies key information.
    """
    
    SYSTEM_PROMPT = """You are a symptom analyst for automotive issues (cars AND two-wheelers).
Extract key information:
1. Vehicle type: car OR two-wheeler (scooty/bike/motorcycle/activa/splendor)
2. Symptoms (noise, behavior, warning lights, not starting)
3. Location (engine, wheels, chain, starter)
4. Condition (starting, driving, braking, idle)

Output as JSON:
{"vehicle_type": "car|two_wheeler", "keywords": ["..."], "location": "...", "condition": "...", "noise_type": "...", "needs_clarification": false, "clarification_question": null}

Understand Hindi/Hinglish:
- gadi=car, awaaz=noise, garam=hot, thanda=cold
- scooty/activa/bike/splendor = TWO-WHEELER
- on nahi ho rahi = won't start
- chain/kick/self start = two-wheeler parts"""

    def __init__(self, llm_client=None):
        self._llm = llm_client
        self._init_llm()
    
    def _init_llm(self):
        if self._llm is None:
            try:
                from ..config import config
                
                if getattr(config, "USE_GROQ", False) and config.GROQ_API_KEY:
                    # Use Groq
                    from langchain_groq import ChatGroq
                    self._llm = ChatGroq(
                        api_key=config.GROQ_API_KEY,
                        model_name=config.GROQ_MODEL,
                        temperature=0.1,
                        max_retries=2
                    )
                    logger.info("Agent using Groq")
                else:
                    # Use Ollama
                    from langchain_ollama import ChatOllama
                    self._llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0.1, format="json")
            except Exception as e:
                logger.warning(f"Could not init LLM: {e}")
    
    def process(self, state: AgentState) -> AgentState:
        """Analyze user input and extract symptoms."""
        user_input = state.current_input
        
        if self._llm is None:
            # Fallback: basic keyword extraction
            return self._fallback_extraction(state)
        
        try:
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"Analyze this complaint: {user_input}")
            ]
            
            response = self._llm.invoke(messages)
            data = json.loads(response.content)
            
            symptoms = SymptomExtraction(
                raw_complaint=user_input,
                detected_language=state.detected_language,
                keywords=data.get("keywords", []),
                location=data.get("location"),
                condition=data.get("condition"),
                noise_type=data.get("noise_type")
            )
            
            state.symptoms = symptoms
            state.needs_clarification = data.get("needs_clarification", False)
            state.clarification_question = data.get("clarification_question")
            
            logger.info(f"Extracted symptoms: {symptoms.keywords}")
            
        except Exception as e:
            logger.error(f"Symptom extraction failed: {e}")
            return self._fallback_extraction(state)
        
        return state
    
    def _fallback_extraction(self, state: AgentState) -> AgentState:
        """Basic keyword extraction without LLM."""
        text = state.current_input.lower()
        
        keywords = []
        if any(w in text for w in ["brake", "braking", "stop"]):
            keywords.append("brake")
        if any(w in text for w in ["noise", "sound", "awaaz"]):
            keywords.append("noise")
        if any(w in text for w in ["start", "crank"]):
            keywords.append("starting")
        if any(w in text for w in ["hot", "garam", "overheat", "temperature"]):
            keywords.append("overheating")
        if any(w in text for w in ["cold", "thanda", "ac", "cooling"]):
            keywords.append("cooling")
        if any(w in text for w in ["tire", "tyre", "puncture", "flat"]):
            keywords.append("tire")
        
        state.symptoms = SymptomExtraction(
            raw_complaint=state.current_input,
            detected_language=state.detected_language,
            keywords=keywords if keywords else ["general_issue"]
        )
        
        return state


class KnowledgeAgent:
    """
    Retrieves relevant technical knowledge from the knowledge base.
    Uses RAG to find diagnostic procedures.
    """
    
    def __init__(self):
        self._embedding_service = None
    
    def _ensure_service(self):
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
            # Index if empty
            if self._embedding_service.get_collection_stats()["knowledge_chunks"] == 0:
                self._embedding_service.index_knowledge_base()
    
    def process(self, state: AgentState) -> AgentState:
        """Retrieve relevant knowledge for the complaint."""
        try:
            self._ensure_service()
            
            # Import hierarchy for re-ranking
            from ..services.knowledge_hierarchy import get_knowledge_hierarchy
            hierarchy = get_knowledge_hierarchy()
            
            # Search with original complaint
            results = self._embedding_service.search_knowledge(
                state.current_input, 
                n_results=5  # Get more, then filter
            )
            
            # Convert to hierarchy format for re-ranking
            docs_for_ranking = [
                {
                    "content": r.content,
                    "metadata": r.metadata,
                    "score": r.score
                }
                for r in results
            ]
            
            # Apply knowledge hierarchy (TSB > Manual > General)
            ranked_docs = hierarchy.rerank(docs_for_ranking, topic=state.current_input)
            
            # Store audit trail
            state.retrieval_audit = hierarchy.get_audit_trail(ranked_docs)
            
            # Take top 3 after re-ranking
            state.knowledge_context = [
                KnowledgeContext(
                    source=doc.get("metadata", {}).get("source", "unknown"),
                    title=doc.get("metadata", {}).get("title", ""),
                    content=doc.get("content", "")[:500],
                    relevance_score=doc.get("score", 0.5)
                )
                for doc in ranked_docs[:3]
            ]
            
            logger.info(f"Retrieved {len(state.knowledge_context)} knowledge chunks (re-ranked by hierarchy)")
            
        except Exception as e:
            logger.error(f"Knowledge retrieval failed: {e}")
        
        return state


class HistorianAgent:
    """
    Searches historical tickets for similar past cases.
    Provides evidence from successful resolutions.
    """
    
    def __init__(self):
        self._embedding_service = None
    
    def _ensure_service(self):
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
            # Index if empty
            if self._embedding_service.get_collection_stats()["tickets"] == 0:
                self._embedding_service.index_tickets()
    
    def process(self, state: AgentState) -> AgentState:
        """Find similar historical tickets with PII redacted."""
        try:
            self._ensure_service()
            
            # PII redaction for privacy
            from ..services.pii_redactor import get_pii_redactor
            redactor = get_pii_redactor()
            
            results = self._embedding_service.search_tickets(
                state.current_input,
                n_results=5
            )
            
            # Redact PII from all ticket fields before returning
            state.similar_tickets = []
            for r in results:
                # Redact content (complaint text)
                content_result = redactor.redact(r.content)
                # Redact resolution (may contain customer interaction notes)
                resolution = r.metadata.get("resolution", "")
                resolution_result = redactor.redact(resolution) if resolution else None
                
                state.similar_tickets.append(SimilarTicket(
                    ticket_id=r.metadata.get("ticket_id", r.id),
                    complaint=content_result.redacted_text,
                    system=r.metadata.get("system", ""),
                    component=r.metadata.get("component", ""),
                    failure_mode=r.metadata.get("failure_mode", ""),
                    resolution=resolution_result.redacted_text if resolution_result else "",
                    similarity_score=r.score
                ))
            
            logger.info(f"Found {len(state.similar_tickets)} similar tickets (PII redacted)")
            
        except Exception as e:
            logger.error(f"Ticket search failed: {e}")
        
        return state


class DiagnosisAgent:
    """
    Makes final diagnosis based on all gathered evidence.
    Uses LLM with RAG context and historical patterns.
    """
    
    SYSTEM_PROMPT = """You are an automotive diagnostic expert for Indian market (cars AND two-wheelers).

VALID CATEGORIES:
{ontology}

IMPORTANT RULES:
1. scooty/activa/bike/splendor/pulsar = use TWO_WHEELER system
2. CNG/LPG issues = use CNG_LPG system
3. "on nahi ho rahi" for two-wheeler = KICK_SELF_START failure

Provide diagnosis as JSON:
{{"system": "...", "component": "...", "failure_mode": "...", "confidence": 0.XX, "reasoning": "..."}}

Use ONLY values from valid categories. For two-wheelers, prefer TWO_WHEELER system over ELECTRICAL/POWERTRAIN."""

    def __init__(self, llm_client=None):
        self._llm = llm_client
        self._init_llm()
    
    def _init_llm(self):
        if self._llm is None:
            try:
                from ..config import config
                
                if getattr(config, "USE_GROQ", False) and config.GROQ_API_KEY:
                    # Use Groq
                    from langchain_groq import ChatGroq
                    self._llm = ChatGroq(
                        api_key=config.GROQ_API_KEY,
                        model_name=config.GROQ_MODEL,
                        temperature=0.1,
                        max_retries=2
                    )
                    logger.info("DiagnosisAgent using Groq")
                else:
                    # Use Ollama
                    from langchain_ollama import ChatOllama
                    self._llm = ChatOllama(model=config.OLLAMA_MODEL, temperature=0.1, format="json")
            except Exception as e:
                logger.warning(f"Could not init LLM: {e}")
    
    def process(self, state: AgentState) -> AgentState:
        """Generate final diagnosis."""
        
        # Build context from gathered evidence
        context_parts = []
        
        # Add complaint
        context_parts.append(f"COMPLAINT: {state.current_input}")
        
        # Add symptoms
        if state.symptoms:
            context_parts.append(f"EXTRACTED: keywords={state.symptoms.keywords}, location={state.symptoms.location}")
        
        # Add similar tickets
        if state.similar_tickets:
            tickets_str = "\n".join([
                f"- {t.complaint} → {t.system}/{t.component}/{t.failure_mode} (similarity: {t.similarity_score:.2f})"
                for t in state.similar_tickets[:3]
            ])
            context_parts.append(f"SIMILAR PAST CASES:\n{tickets_str}")
        
        # Add knowledge
        if state.knowledge_context:
            knowledge_str = "\n".join([
                f"- [{k.source}] {k.content[:200]}..."
                for k in state.knowledge_context[:2]
            ])
            context_parts.append(f"TECHNICAL KNOWLEDGE:\n{knowledge_str}")
        
        full_context = "\n\n".join(context_parts)
        
        if self._llm is None:
            return self._fallback_diagnosis(state)
        
        try:
            prompt = self.SYSTEM_PROMPT.format(ontology=get_ontology_formatted())
            
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Diagnose based on this evidence:\n\n{full_context}")
            ]
            
            response = self._llm.invoke(messages)
            data = json.loads(response.content)
            
            diagnosis = DiagnosisCandidate(
                system=data.get("system", "").upper(),
                component=data.get("component", "").upper(),
                failure_mode=data.get("failure_mode", "").upper(),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                supporting_evidence=[t.ticket_id for t in state.similar_tickets[:3]]
            )
            
            state.candidates.append(diagnosis)
            state.final_diagnosis = diagnosis
            state.overall_confidence = diagnosis.confidence
            state.is_complete = diagnosis.confidence >= 0.7
            
            logger.info(f"Diagnosis: {diagnosis.system}/{diagnosis.failure_mode} @ {diagnosis.confidence}")
            
        except Exception as e:
            logger.error(f"Diagnosis failed: {e}")
            return self._fallback_diagnosis(state)
        
        return state
    
    def _fallback_diagnosis(self, state: AgentState) -> AgentState:
        """Use historical tickets for diagnosis when LLM unavailable."""
        if state.similar_tickets:
            top = state.similar_tickets[0]
            diagnosis = DiagnosisCandidate(
                system=top.system,
                component=top.component,
                failure_mode=top.failure_mode,
                confidence=min(top.similarity_score, 0.85),
                reasoning=f"Based on similar ticket: {top.ticket_id}",
                supporting_evidence=[top.ticket_id]
            )
            state.final_diagnosis = diagnosis
            state.overall_confidence = diagnosis.confidence
            state.is_complete = True
        
        return state
