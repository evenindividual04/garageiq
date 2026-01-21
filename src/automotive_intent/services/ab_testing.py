"""
A/B Testing Framework
Enables comparing different prompts, models, and configurations.
"""
import logging
import random
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from pydantic import BaseModel, Field
import hashlib

logger = logging.getLogger(__name__)


class ExperimentVariant(BaseModel):
    """A single variant in an A/B test."""
    name: str
    config: Dict[str, Any]
    weight: float = 0.5  # Traffic allocation


class ExperimentResult(BaseModel):
    """Result of a single experiment trial."""
    experiment_id: str
    variant: str
    input_hash: str  # For reproducibility
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # Metrics
    confidence: float
    classification_status: str
    response_time_ms: float
    
    # Optional feedback
    user_feedback: Optional[str] = None
    was_correct: Optional[bool] = None


class Experiment(BaseModel):
    """An A/B test experiment."""
    id: str
    name: str
    description: str
    variants: List[ExperimentVariant]
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    is_active: bool = True
    
    # Results storage
    results: List[ExperimentResult] = []


class ABTestingService:
    """
    Manages A/B testing experiments for:
    - Different prompt templates
    - Different models (mistral vs others)
    - Different confidence thresholds
    - RAG vs no-RAG comparisons
    """
    
    def __init__(self, experiments_file: str = "./data/experiments.json"):
        self.experiments_file = Path(experiments_file)
        self._experiments: Dict[str, Experiment] = {}
        self._load()
    
    def _load(self):
        """Load experiments from file."""
        if self.experiments_file.exists():
            try:
                with open(self.experiments_file) as f:
                    data = json.load(f)
                for exp_data in data.get("experiments", []):
                    exp = Experiment(**exp_data)
                    self._experiments[exp.id] = exp
                logger.info(f"Loaded {len(self._experiments)} experiments")
            except Exception as e:
                logger.warning(f"Could not load experiments: {e}")
    
    def _save(self):
        """Save experiments to file."""
        self.experiments_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "experiments": [e.model_dump() for e in self._experiments.values()]
        }
        with open(self.experiments_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def create_experiment(
        self,
        name: str,
        description: str,
        variants: List[Dict]
    ) -> Experiment:
        """Create a new A/B test experiment."""
        exp_id = f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        experiment = Experiment(
            id=exp_id,
            name=name,
            description=description,
            variants=[ExperimentVariant(**v) for v in variants]
        )
        
        self._experiments[exp_id] = experiment
        self._save()
        
        logger.info(f"Created experiment: {exp_id}")
        return experiment
    
    def get_variant(self, experiment_id: str, user_id: str = None) -> Optional[ExperimentVariant]:
        """
        Get variant assignment for a user.
        Uses consistent hashing so same user always gets same variant.
        """
        experiment = self._experiments.get(experiment_id)
        if not experiment or not experiment.is_active:
            return None
        
        if not experiment.variants:
            return None
        
        # Consistent assignment based on user_id
        if user_id:
            hash_input = f"{experiment_id}:{user_id}"
            hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
            normalized = (hash_value % 1000) / 1000
        else:
            normalized = random.random()
        
        # Select variant based on weight
        cumulative = 0.0
        for variant in experiment.variants:
            cumulative += variant.weight
            if normalized <= cumulative:
                return variant
        
        return experiment.variants[-1]
    
    def record_result(
        self,
        experiment_id: str,
        variant_name: str,
        input_text: str,
        confidence: float,
        status: str,
        response_time_ms: float
    ) -> ExperimentResult:
        """Record the result of an experiment trial."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment not found: {experiment_id}")
        
        # Hash input for privacy and reproducibility
        input_hash = hashlib.md5(input_text.encode()).hexdigest()[:12]
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant=variant_name,
            input_hash=input_hash,
            confidence=confidence,
            classification_status=status,
            response_time_ms=response_time_ms
        )
        
        experiment.results.append(result)
        self._save()
        
        return result
    
    def get_stats(self, experiment_id: str) -> Dict:
        """Get aggregated statistics for an experiment."""
        experiment = self._experiments.get(experiment_id)
        if not experiment:
            return {}
        
        stats_by_variant: Dict[str, Dict] = {}
        
        for variant in experiment.variants:
            variant_results = [r for r in experiment.results if r.variant == variant.name]
            
            if variant_results:
                confidences = [r.confidence for r in variant_results]
                times = [r.response_time_ms for r in variant_results]
                confirmed = sum(1 for r in variant_results if r.classification_status == "CONFIRMED")
                
                stats_by_variant[variant.name] = {
                    "count": len(variant_results),
                    "avg_confidence": sum(confidences) / len(confidences),
                    "avg_response_time_ms": sum(times) / len(times),
                    "confirmed_rate": confirmed / len(variant_results)
                }
            else:
                stats_by_variant[variant.name] = {"count": 0}
        
        return {
            "experiment_id": experiment_id,
            "name": experiment.name,
            "total_trials": len(experiment.results),
            "variants": stats_by_variant
        }
    
    def list_experiments(self) -> List[Dict]:
        """List all experiments."""
        return [
            {
                "id": e.id,
                "name": e.name,
                "is_active": e.is_active,
                "variants": len(e.variants),
                "results": len(e.results)
            }
            for e in self._experiments.values()
        ]


# Singleton
_ab_service: Optional[ABTestingService] = None


def get_ab_testing_service() -> ABTestingService:
    global _ab_service
    if _ab_service is None:
        _ab_service = ABTestingService()
    return _ab_service
