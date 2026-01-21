"""
Configuration module for GarageIQ
Environment-based settings for production vs development.
"""
import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    """Application configuration."""
    
    # Environment
    ENV: Literal["production", "development", "testing"] = "production"
    
    # LLM Settings
    USE_OLLAMA: bool = False  # Disabled - using Groq instead
    OLLAMA_MODEL: str = "phi3:mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_TEMPERATURE: float = 0.1  # Low for consistency
    LLM_TIMEOUT: int = 30  # seconds
    
    # Groq Cloud Settings (FAST!)
    USE_GROQ: bool = True
    GROQ_API_KEY: str = ""  # Set via GROQ_API_KEY env var
    GROQ_MODEL: str = "llama-3.1-8b-instant"  # ~500 tokens/sec
    
    # Translation Settings
    USE_NLLB: bool = True
    NLLB_MODEL: str = "facebook/nllb-200-distilled-600M"
    
    # Classification Thresholds (PRD §5.4)
    CONFIDENCE_THRESHOLD: float = 0.70
    AMBIGUITY_DELTA: float = 0.10
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list = None
    
    def __post_init__(self):
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = ["*"]
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            ENV=os.getenv("AMI_ENV", "production"),
            USE_OLLAMA=os.getenv("AMI_USE_OLLAMA", "true").lower() == "true",
            OLLAMA_MODEL=os.getenv("AMI_OLLAMA_MODEL", "mistral"),
            OLLAMA_BASE_URL=os.getenv("AMI_OLLAMA_URL", "http://localhost:11434"),
            USE_NLLB=os.getenv("AMI_USE_NLLB", "true").lower() == "true",
            NLLB_MODEL=os.getenv("AMI_NLLB_MODEL", "facebook/nllb-200-distilled-600M"),
            API_HOST=os.getenv("AMI_HOST", "0.0.0.0"),
            API_PORT=int(os.getenv("AMI_PORT", "8000")),
            GROQ_API_KEY=os.getenv("GROQ_API_KEY", ""),  # Required: set GROQ_API_KEY env var
            USE_GROQ=os.getenv("AMI_USE_GROQ", "true").lower() == "true",
        )


# Global config instance
config = Config.from_env()
