"""
Multi-Agent Orchestration using LangGraph
Coordinates agents in a workflow to diagnose automotive issues.
"""
import logging
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END

from .state import AgentState, Message, ChatRequest, ChatResponse
from .agents import (
    SymptomAnalystAgent,
    KnowledgeAgent,
    HistorianAgent,
    DiagnosisAgent
)

logger = logging.getLogger(__name__)


class DiagnosticOrchestrator:
    """
    Orchestrates the multi-agent diagnostic workflow.
    
    Flow:
    1. Router → decides which path
    2. Symptom Analyst → extracts symptoms
    3. Knowledge Agent → retrieves docs (parallel)
    4. Historian Agent → finds similar tickets (parallel)
    5. Diagnosis Agent → makes final diagnosis
    """
    
    def __init__(self):
        # Initialize agents
        self.symptom_agent = SymptomAnalystAgent()
        self.knowledge_agent = KnowledgeAgent()
        self.historian_agent = HistorianAgent()
        self.diagnosis_agent = DiagnosisAgent()
        
        # Build graph
        self.graph = self._build_graph()
        
        # Session storage
        self._sessions: Dict[str, AgentState] = {}
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create graph with dict state to avoid Pydantic issues
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("symptom_analyst", self._symptom_node)
        workflow.add_node("knowledge_retrieval", self._knowledge_node)
        workflow.add_node("historian", self._historian_node)
        workflow.add_node("diagnosis", self._diagnosis_node)
        workflow.add_node("respond", self._respond_node)
        
        # Set entry point
        workflow.add_node("refine_query", self._refine_node)
        
        # Set entry point
        workflow.set_entry_point("symptom_analyst")
        
        # Define edges with reflection loop
        workflow.add_edge("symptom_analyst", "historian")
        workflow.add_edge("historian", "knowledge_retrieval")
        workflow.add_edge("knowledge_retrieval", "diagnosis")
        
        # Conditional edge from diagnosis
        workflow.add_conditional_edges(
            "diagnosis",
            self._check_confidence,
            {
                "respond": "respond",
                "refine": "refine_query"
            }
        )
        
        workflow.add_edge("refine_query", "knowledge_retrieval")
        workflow.add_edge("respond", END)
        
        return workflow.compile()
    
    def _symptom_node(self, state: dict) -> dict:
        """Process symptoms."""
        logger.info("🔍 Symptom Analyst processing...")
        agent_state = AgentState(**state)
        agent_state.current_agent = "symptom_analyst"
        result = self.symptom_agent.process(agent_state)
        return result.model_dump()
    
    def _knowledge_node(self, state: dict) -> dict:
        """Retrieve knowledge."""
        logger.info("📚 Knowledge Agent retrieving...")
        agent_state = AgentState(**state)
        agent_state.current_agent = "knowledge"
        result = self.knowledge_agent.process(agent_state)
        return result.model_dump()
    
    def _historian_node(self, state: dict) -> dict:
        """Search historical tickets."""
        logger.info("📋 Historian searching...")
        agent_state = AgentState(**state)
        agent_state.current_agent = "historian"
        result = self.historian_agent.process(agent_state)
        return result.model_dump()
    
    def _diagnosis_node(self, state: dict) -> dict:
        """Make diagnosis."""
        logger.info("🔧 Diagnosis Agent analyzing...")
        agent_state = AgentState(**state)
        agent_state.current_agent = "diagnosis"
        result = self.diagnosis_agent.process(agent_state)
        return result.model_dump()
    
    def _check_confidence(self, state: dict) -> str:
        """Decide whether to loop back or respond."""
        agent_state = AgentState(**state)
        
        # Stop looping if confidence is high enough or max loops reached
        if agent_state.final_diagnosis and agent_state.final_diagnosis.confidence >= 0.7:
            return "respond"
        
        if agent_state.loop_count >= 1:  # Max 1 reflection loop to prevent infinite cycle
            logger.info("Max loops reached, finishing.")
            return "respond"
            
        logger.info(f"Confidence low ({agent_state.overall_confidence}), reflecting...")
        return "refine"

    def _refine_node(self, state: dict) -> dict:
        """Refine search query based on diagnosis feedback."""
        logger.info("🔄 Agent Reflecting: Refining search query...")
        agent_state = AgentState(**state)
        agent_state.loop_count += 1
        
        # Simple reflection logic: focus on component or failure mode if suspected
        candidates = agent_state.candidates
        new_query = agent_state.current_input
        
        if candidates:
            top = candidates[-1]
            # Try searching for the suspected problem specifically
            new_query = f"{top.system} {top.component} {top.failure_mode} symptoms"
            logger.info(f"Refined query: {new_query}")
        
        agent_state.refined_query = new_query
        # Prioritize refined query for next retrieval
        agent_state.current_input = new_query
        
        return agent_state.model_dump()
    
    def _respond_node(self, state: dict) -> dict:
        """Generate response message with parts dependencies."""
        agent_state = AgentState(**state)
        
        if agent_state.final_diagnosis:
            diag = agent_state.final_diagnosis
            
            # Parts Graph Lookup - get mandatory/recommended parts
            parts_info = self._lookup_parts(diag.failure_mode, diag.component)
            if parts_info:
                agent_state.parts_dependencies = parts_info
            
            response = (
                f"**Diagnosis:** {diag.system} → {diag.component} → {diag.failure_mode}\n"
                f"**Confidence:** {int(diag.confidence * 100)}%\n"
                f"**Reasoning:** {diag.reasoning}"
            )
            
            # Add parts recommendations if available
            if parts_info:
                mandatory = parts_info.get("mandatory", [])
                recommended = parts_info.get("recommended", [])
                if mandatory:
                    response += f"\n\n**Required Parts:** {', '.join(mandatory)}"
                if recommended:
                    response += f"\n**Also Recommended:** {', '.join(recommended)}"
                if parts_info.get("labor_note"):
                    response += f"\n💡 *{parts_info['labor_note']}*"
            
            if agent_state.similar_tickets:
                response += f"\n\n**Similar past cases:** {len(agent_state.similar_tickets)} found"
        else:
            response = "I couldn't determine a diagnosis. Could you provide more details about the issue?"
            agent_state.needs_clarification = True
        
        agent_state.messages.append(Message(
            role="assistant",
            content=response,
            agent="diagnosis"
        ))
        
        return agent_state.model_dump()
    
    def _lookup_parts(self, failure_mode: str, component: str) -> dict | None:
        """Lookup parts dependencies from graph."""
        import json
        from pathlib import Path
        
        try:
            graph_path = Path(__file__).parent.parent.parent.parent / "data" / "parts_graph.json"
            with open(graph_path) as f:
                graph = json.load(f)
            
            # Try matching by failure_mode or component
            key = failure_mode.lower().replace("_", "_")
            if key in graph:
                return graph[key]
            
            # Try component match
            comp_key = component.lower().replace(" ", "_")
            if comp_key in graph:
                return graph[comp_key]
            
            # Fuzzy match
            for part_key in graph:
                if part_key in failure_mode.lower() or failure_mode.lower() in part_key:
                    return graph[part_key]
                if part_key in comp_key or comp_key in part_key:
                    return graph[part_key]
            
        except Exception as e:
            logger.warning(f"Parts graph lookup failed: {e}")
        
        return None
    
    def process_message(self, request: ChatRequest) -> ChatResponse:
        """Process a user message through the agent workflow."""
        
        # Get or create session
        if request.session_id and request.session_id in self._sessions:
            state = self._sessions[request.session_id]
        else:
            state = AgentState()
        
        # Normalize noisy input (abbreviations, slang)
        from ..services.normalizer import get_normalizer
        normalizer = get_normalizer()
        normalized_text, norm_meta = normalizer.normalize(request.message)
        
        # Extract VIN if present
        from ..services.vin_decoder import get_vin_decoder
        import re
        vin_pattern = r'[A-HJ-NPR-Z0-9]{17}|[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}'
        vin_match = re.search(vin_pattern, request.message, re.IGNORECASE)
        if vin_match:
            decoder = get_vin_decoder()
            vehicle_info = decoder.decode(vin_match.group())
            if vehicle_info:
                state.vehicle_info = {
                    "vin": vehicle_info.vin,
                    "make": vehicle_info.make,
                    "year": vehicle_info.year,
                    "engine": vehicle_info.engine,
                    "filter_tags": vehicle_info.get_filter_tags()
                }
                logger.info(f"Extracted vehicle: {state.vehicle_info}")
        
        # Add user message (use normalized text)
        state.messages.append(Message(role="user", content=normalized_text))
        state.current_input = normalized_text
        
        # Detect language (simple)
        if any(c in request.message for c in "अआइईउऊएऐओऔकखगघ"):
            state.detected_language = "hi"
        else:
            state.detected_language = "en"
        
        # Run workflow
        try:
            # Convert Pydantic model to dict for LangGraph
            state_dict = state.model_dump()
            final_state = self.graph.invoke(state_dict)
            state = AgentState(**final_state)
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            state.messages.append(Message(
                role="assistant",
                content=f"Sorry, an error occurred: {str(e)}",
                agent="system"
            ))
        
        # Store session
        self._sessions[state.session_id] = state
        
        # Build response
        last_message = state.messages[-1] if state.messages else Message(role="assistant", content="")
        
        return ChatResponse(
            session_id=state.session_id,
            message=last_message.content,
            agent=last_message.agent or "system",
            is_complete=state.is_complete,
            diagnosis=state.final_diagnosis,
            similar_tickets=state.similar_tickets[:3],
            confidence=state.overall_confidence,
            needs_input=state.needs_clarification
        )
    
    def get_session(self, session_id: str) -> AgentState | None:
        """Get session state."""
        return self._sessions.get(session_id)
    
    def clear_session(self, session_id: str):
        """Clear a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]


# Singleton
_orchestrator: DiagnosticOrchestrator | None = None


def get_orchestrator() -> DiagnosticOrchestrator:
    """Get or create orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DiagnosticOrchestrator()
    return _orchestrator
