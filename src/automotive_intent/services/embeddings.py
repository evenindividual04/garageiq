"""
Embedding Service with Multilingual Support
Uses sentence-transformers for embeddings and ChromaDB for vector storage.
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from similarity search."""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class EmbeddingService:
    """
    Multilingual embedding service using sentence-transformers.
    Uses paraphrase-multilingual-MiniLM-L12-v2 for Hindi + English support.
    """
    
    # Multilingual model - understands Hindi, English, and 50+ languages
    MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, persist_directory: str = "./data/vectordb"):
        self._model = None
        self._chroma_client = None
        self._knowledge_collection = None
        self._tickets_collection = None
        self.persist_directory = persist_directory
        
        self._initialize()
    
    def _initialize(self):
        """Initialize embedding model and ChromaDB."""
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            from chromadb.config import Settings
            
            # Load multilingual model
            logger.info(f"Loading embedding model: {self.MODEL_NAME}")
            self._model = SentenceTransformer(self.MODEL_NAME)
            
            # Initialize ChromaDB with persistence
            self._chroma_client = chromadb.Client(Settings(
                anonymized_telemetry=False,
                is_persistent=True,
                persist_directory=self.persist_directory
            ))
            
            # Create collections
            self._knowledge_collection = self._chroma_client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            
            self._tickets_collection = self._chroma_client.get_or_create_collection(
                name="historical_tickets",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("Embedding service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model.encode(text).tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if self._model is None:
            raise RuntimeError("Model not initialized")
        return self._model.encode(texts).tolist()
    
    def index_knowledge_base(self, knowledge_dir: str = "./data/knowledge_base"):
        """Index all knowledge base documents."""
        kb_path = Path(knowledge_dir)
        if not kb_path.exists():
            logger.warning(f"Knowledge base directory not found: {knowledge_dir}")
            return
        
        documents = []
        metadatas = []
        ids = []
        
        for doc_file in kb_path.glob("*.md"):
            content = doc_file.read_text()
            
            # Split into chunks (by section)
            chunks = self._chunk_document(content, doc_file.stem)
            
            for i, (chunk_text, chunk_meta) in enumerate(chunks):
                doc_id = f"{doc_file.stem}_{i}"
                documents.append(chunk_text)
                metadatas.append(chunk_meta)
                ids.append(doc_id)
        
        if documents:
            embeddings = self.embed_texts(documents)
            
            # Upsert to ChromaDB
            self._knowledge_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Indexed {len(documents)} knowledge chunks")
    
    def _chunk_document(self, content: str, source: str) -> List[tuple]:
        """Split document into chunks with metadata."""
        chunks = []
        
        # Split by headers
        sections = content.split("\n## ")
        
        for section in sections:
            if not section.strip():
                continue
            
            # Get title from first line
            lines = section.strip().split("\n")
            title = lines[0].replace("#", "").strip()
            text = "\n".join(lines[1:]) if len(lines) > 1 else lines[0]
            
            if len(text) > 50:  # Only include meaningful chunks
                chunks.append((
                    text,
                    {"source": source, "title": title, "type": "knowledge"}
                ))
        
        return chunks
    
    def index_tickets(self, tickets_file: str = "./data/tickets/historical_tickets.json"):
        """Index historical tickets."""
        tickets_path = Path(tickets_file)
        if not tickets_path.exists():
            logger.warning(f"Tickets file not found: {tickets_file}")
            return
        
        with open(tickets_path) as f:
            tickets = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for ticket in tickets:
            # Combine English and Hindi complaints for better search
            combined = f"{ticket['complaint']} {ticket.get('complaint_hi', '')}"
            
            documents.append(combined)
            metadatas.append({
                "ticket_id": ticket["id"],
                "system": ticket["system"],
                "component": ticket["component"],
                "failure_mode": ticket["failure_mode"],
                "resolution": ticket["resolution"],
                "type": "ticket"
            })
            ids.append(ticket["id"])
        
        if documents:
            embeddings = self.embed_texts(documents)
            
            self._tickets_collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Indexed {len(documents)} historical tickets")
    
    def search_knowledge(self, query: str, n_results: int = 3) -> List[SearchResult]:
        """Search knowledge base for relevant context."""
        query_embedding = self.embed_text(query)
        
        results = self._knowledge_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                score = 1 - results["distances"][0][i]  # Convert distance to similarity
                search_results.append(SearchResult(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    score=score
                ))
        
        return search_results
    
    def search_tickets(self, query: str, n_results: int = 5) -> List[SearchResult]:
        """Search historical tickets for similar cases."""
        query_embedding = self.embed_text(query)
        
        results = self._tickets_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                score = 1 - results["distances"][0][i]
                search_results.append(SearchResult(
                    id=doc_id,
                    content=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                    score=score
                ))
        
        return search_results
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get stats about indexed collections."""
        return {
            "knowledge_chunks": self._knowledge_collection.count(),
            "tickets": self._tickets_collection.count()
        }


# Singleton
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
