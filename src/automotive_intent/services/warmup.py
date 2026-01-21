"""
Model warmup script for faster first request.
Run this on startup to preload models into memory.
"""
import logging
import time

logger = logging.getLogger(__name__)


def warmup_ollama():
    """Warm up Ollama model with a simple query."""
    try:
        import ollama
        from ..config import config
        
        logger.info(f"Warming up Ollama model: {config.OLLAMA_MODEL}")
        start = time.time()
        
        # Simple warmup query
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            options={
                "num_predict": 5,  # Very short response
                "temperature": 0
            }
        )
        
        elapsed = time.time() - start
        logger.info(f"Ollama warmup complete in {elapsed:.2f}s")
        return True
    except Exception as e:
        logger.warning(f"Ollama warmup failed: {e}")
        return False


def warmup_embeddings():
    """Warm up embedding model."""
    try:
        from ..services.embeddings import get_embedding_service
        
        logger.info("Warming up embedding model...")
        start = time.time()
        
        service = get_embedding_service()
        # Generate a test embedding
        _ = service.model.encode("test warmup query")
        
        elapsed = time.time() - start
        logger.info(f"Embedding warmup complete in {elapsed:.2f}s")
        return True
    except Exception as e:
        logger.warning(f"Embedding warmup failed: {e}")
        return False


def warmup_all():
    """Warm up all models on startup."""
    logger.info("=== Model Warmup Starting ===")
    total_start = time.time()
    
    results = {
        "ollama": warmup_ollama(),
        "embeddings": warmup_embeddings()
    }
    
    total_elapsed = time.time() - total_start
    logger.info(f"=== Warmup Complete in {total_elapsed:.2f}s ===")
    
    return results
