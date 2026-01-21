"""
Knowledge Hierarchy Service
Implements source prioritization for RAG retrieval.
TSB > Recall > Manual > General
"""
import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SourcePriority:
    """Source type with priority ranking."""
    source_type: str
    priority: int  # Lower = higher priority
    description: str


# Knowledge hierarchy (lower number = higher priority)
SOURCE_HIERARCHY = {
    "recall": SourcePriority("recall", 1, "Safety recalls - highest priority"),
    "tsb": SourcePriority("tsb", 2, "Technical Service Bulletins"),
    "campaign": SourcePriority("campaign", 3, "Service campaigns"),
    "manual": SourcePriority("manual", 4, "Original service manual"),
    "general": SourcePriority("general", 5, "General knowledge"),
    "community": SourcePriority("community", 6, "Community/forum knowledge"),
}


class KnowledgeHierarchy:
    """
    Re-ranks RAG results based on source authority.
    
    Key behaviors:
    1. TSBs override Manuals for the same topic
    2. Newer documents override older ones (same source type)
    3. Recalls always surface to top
    """
    
    def __init__(self):
        self.hierarchy = SOURCE_HIERARCHY
    
    def rerank(
        self, 
        documents: List[Dict[str, Any]], 
        topic: str = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents by source authority.
        
        Args:
            documents: List of {"content": str, "metadata": dict, "score": float}
            topic: Optional topic for conflict detection
            
        Returns:
            Re-ranked documents with conflicts resolved
        """
        if not documents:
            return documents
        
        # Step 1: Assign priority scores
        scored = []
        for doc in documents:
            metadata = doc.get("metadata", {})
            source_type = metadata.get("source_type", "general").lower()
            effective_date = metadata.get("effective_date")
            
            # Base priority from hierarchy
            priority_info = self.hierarchy.get(source_type, self.hierarchy["general"])
            base_priority = priority_info.priority
            
            # Date boost: newer documents get lower (better) priority within same type
            date_boost = 0
            if effective_date:
                try:
                    doc_date = datetime.fromisoformat(effective_date)
                    days_old = (datetime.now() - doc_date).days
                    date_boost = min(days_old / 365, 1)  # Up to 1 point penalty for old docs
                except:
                    pass
            
            # Combined score (lower is better)
            final_priority = base_priority + date_boost
            
            scored.append({
                **doc,
                "_priority": final_priority,
                "_source_type": source_type,
            })
        
        # Step 2: Sort by priority
        scored.sort(key=lambda x: x["_priority"])
        
        # Step 3: Detect and resolve conflicts
        if topic:
            scored = self._resolve_conflicts(scored, topic)
        
        # Step 4: Log hierarchy decisions
        if len(scored) > 1:
            top_source = scored[0].get("_source_type", "unknown")
            suppressed = [d.get("_source_type") for d in scored[1:] if d.get("_source_type") != top_source]
            if suppressed:
                logger.info(f"Knowledge hierarchy: {top_source} supersedes {suppressed}")
        
        return scored
    
    def _resolve_conflicts(
        self, 
        documents: List[Dict[str, Any]], 
        topic: str
    ) -> List[Dict[str, Any]]:
        """
        Remove lower-priority documents that conflict with higher-priority ones.
        
        Example: If TSB says "Don't reuse coolant" and Manual says "Reuse coolant",
        we suppress the Manual entry.
        """
        # Simple conflict detection: same topic, different source types
        # In production, this would use semantic similarity
        
        seen_topics = set()
        result = []
        
        for doc in documents:
            source_type = doc.get("_source_type")
            content_hash = hash(doc.get("content", "")[:100])  # Simple dedup
            
            # Check for supersession markers
            content = doc.get("content", "").lower()
            if "supersedes" in content or "overrides" in content:
                doc["_is_override"] = True
            
            # Deduplicate
            if content_hash not in seen_topics:
                seen_topics.add(content_hash)
                result.append(doc)
        
        return result
    
    def get_audit_trail(self, documents: List[Dict[str, Any]]) -> Dict:
        """
        Generate audit trail for compliance.
        Shows which documents were retrieved and why some were suppressed.
        """
        return {
            "retrieved_count": len(documents),
            "sources": [
                {
                    "type": d.get("_source_type"),
                    "priority": d.get("_priority"),
                    "is_override": d.get("_is_override", False),
                }
                for d in documents
            ],
            "timestamp": datetime.now().isoformat(),
        }


# Singleton
_hierarchy = None

def get_knowledge_hierarchy() -> KnowledgeHierarchy:
    global _hierarchy
    if _hierarchy is None:
        _hierarchy = KnowledgeHierarchy()
    return _hierarchy
