"""
Explanation Generator
Generates human-readable explanations for classification decisions.
Shows WHY a particular diagnosis was made.
"""
import logging
from typing import List, Optional, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ExplanationStep(BaseModel):
    """A single step in the explanation chain."""
    step: int
    agent: str
    action: str
    evidence: str
    contribution: str


class Explanation(BaseModel):
    """Full explanation for a classification decision."""
    summary: str
    confidence_reason: str
    steps: List[ExplanationStep]
    supporting_evidence: List[str]
    alternative_considerations: List[str]


class ExplanationGenerator:
    """
    Generates explanations for diagnostic decisions.
    Implements Chain-of-Thought explanation pattern.
    """
    
    def generate(
        self,
        input_text: str,
        diagnosis_system: str,
        diagnosis_component: str,
        diagnosis_failure_mode: str,
        confidence: float,
        similar_tickets: List[Dict] = None,
        knowledge_context: List[Dict] = None,
        symptoms: Dict = None
    ) -> Explanation:
        """Generate a human-readable explanation."""
        
        steps = []
        evidence = []
        
        # Step 1: Symptom extraction
        if symptoms:
            keywords = symptoms.get("keywords", [])
            steps.append(ExplanationStep(
                step=1,
                agent="Symptom Analyst",
                action="Extracted key symptoms from input",
                evidence=f"Keywords: {', '.join(keywords)}" if keywords else "No specific keywords",
                contribution=f"Identified potential system: {diagnosis_system}"
            ))
            evidence.append(f"Keywords detected: {keywords}")
        
        # Step 2: Historical ticket matching
        if similar_tickets:
            top_ticket = similar_tickets[0] if similar_tickets else {}
            steps.append(ExplanationStep(
                step=2,
                agent="Historian",
                action=f"Found {len(similar_tickets)} similar past cases",
                evidence=f"Most similar: {top_ticket.get('complaint', 'N/A')} (score: {top_ticket.get('similarity_score', 0):.2f})",
                contribution=f"Past cases support {diagnosis_failure_mode} diagnosis"
            ))
            evidence.append(f"Similar ticket: {top_ticket.get('ticket_id', 'N/A')}")
        
        # Step 3: Knowledge retrieval
        if knowledge_context:
            top_doc = knowledge_context[0] if knowledge_context else {}
            steps.append(ExplanationStep(
                step=3,
                agent="Knowledge Agent",
                action="Retrieved technical documentation",
                evidence=f"Source: {top_doc.get('source', 'N/A')}, Relevance: {top_doc.get('relevance_score', 0):.2f}",
                contribution=f"Documentation confirms {diagnosis_component} as likely affected component"
            ))
            evidence.append(f"Technical doc: {top_doc.get('title', 'N/A')}")
        
        # Step 4: Final diagnosis
        steps.append(ExplanationStep(
            step=len(steps) + 1,
            agent="Diagnosis Agent",
            action="Combined evidence to make final classification",
            evidence=f"Confidence score: {confidence:.0%}",
            contribution=f"Classified as {diagnosis_system}/{diagnosis_component}/{diagnosis_failure_mode}"
        ))
        
        # Generate summary
        summary = self._generate_summary(
            input_text, diagnosis_system, diagnosis_component, 
            diagnosis_failure_mode, confidence, len(similar_tickets or [])
        )
        
        # Confidence reasoning
        confidence_reason = self._explain_confidence(confidence, similar_tickets, knowledge_context)
        
        # Alternatives
        alternatives = self._generate_alternatives(diagnosis_system, diagnosis_failure_mode)
        
        return Explanation(
            summary=summary,
            confidence_reason=confidence_reason,
            steps=steps,
            supporting_evidence=evidence,
            alternative_considerations=alternatives
        )
    
    def _generate_summary(
        self, 
        input_text: str, 
        system: str, 
        component: str, 
        failure_mode: str,
        confidence: float,
        num_similar: int
    ) -> str:
        """Generate a one-sentence summary."""
        conf_word = "high" if confidence >= 0.8 else "moderate" if confidence >= 0.6 else "low"
        
        return (
            f"Based on the complaint, this appears to be a {failure_mode.replace('_', ' ').lower()} "
            f"issue in the {component.replace('_', ' ').lower()} ({system}). "
            f"Confidence is {conf_word} ({confidence:.0%}) with {num_similar} similar past cases found."
        )
    
    def _explain_confidence(
        self, 
        confidence: float,
        similar_tickets: List[Dict] = None,
        knowledge_context: List[Dict] = None
    ) -> str:
        """Explain why confidence is high or low."""
        reasons = []
        
        if similar_tickets:
            top_score = similar_tickets[0].get("similarity_score", 0) if similar_tickets else 0
            if top_score > 0.8:
                reasons.append("very similar historical cases found")
            elif top_score > 0.6:
                reasons.append("moderately similar past cases found")
        else:
            reasons.append("no historical match found")
        
        if knowledge_context:
            top_relevance = knowledge_context[0].get("relevance_score", 0) if knowledge_context else 0
            if top_relevance > 0.7:
                reasons.append("strong technical documentation match")
        
        if confidence >= 0.8:
            return f"Confidence is high because: {', '.join(reasons)}"
        elif confidence >= 0.6:
            return f"Confidence is moderate because: {', '.join(reasons)}"
        else:
            return f"Confidence is low because: {', '.join(reasons)}. Consider asking follow-up questions."
    
    def _generate_alternatives(self, system: str, failure_mode: str) -> List[str]:
        """Suggest alternative diagnoses to consider."""
        alternatives = {
            "BRAKES": [
                "If noise persists after pad replacement, check rotors for warping",
                "Consider checking brake fluid level and condition"
            ],
            "POWERTRAIN": [
                "If overheating, also check water pump and thermostat",
                "Misfires can be caused by ignition or fuel issues"
            ],
            "ELECTRICAL": [
                "Battery issues can mask alternator problems",
                "Check ground connections if multiple electrical issues"
            ],
            "HVAC": [
                "No cooling can be compressor or refrigerant leak",
                "Blower issues may be resistor or motor related"
            ]
        }
        
        return alternatives.get(system, ["Consider related components in this system"])


# Singleton
_generator: Optional[ExplanationGenerator] = None


def get_explanation_generator() -> ExplanationGenerator:
    global _generator
    if _generator is None:
        _generator = ExplanationGenerator()
    return _generator
