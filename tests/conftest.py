"""
Shared test fixtures and configuration.
"""
import pytest
import os
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set test environment
os.environ["AMI_ENV"] = "testing"
os.environ["AMI_USE_OLLAMA"] = "false"
os.environ["AMI_USE_NLLB"] = "false"


@pytest.fixture
def sample_complaints():
    """Sample complaints for testing."""
    return {
        "english": [
            "My car won't start, just clicks",
            "Brake noise when stopping",
            "AC not cooling properly",
            "Engine overheating on highway",
        ],
        "hindi": [
            "Gaadi start nahi ho rahi",
            "Brake lagane par awaaz aa rahi hai",
            "AC thanda nahi kar raha",
        ],
        "noisy": [
            "Cus sts frt lft thumping noise",
            "chk eng lite on, rgh idle",
            "brk pedal spongy, frt rt pull",
        ],
        "with_pii": [
            "Customer Rahul called on +91 9876543210 about brakes",
            "Email from test@example.com regarding engine issue",
            "Mr. Sharma at Plot 42, Sector 15 complained about AC",
        ],
        "with_vin": [
            "VIN: 1HGCM82633A123456 brake noise",
            "MH12AB1234 engine overheating",
        ],
    }


@pytest.fixture
def sample_diagnosis():
    """Sample diagnosis result."""
    return {
        "system": "BRAKES",
        "component": "PADS_ROTORS",
        "failure_mode": "SQUEALING",
        "confidence": 0.92,
    }


@pytest.fixture
def mock_pipeline(monkeypatch):
    """Mock pipeline for fast tests."""
    from automotive_intent.core.schemas import ServiceTicket, Intent, RequestMeta
    
    def mock_process(request):
        return ServiceTicket(
            classification_status="CONFIRMED",
            meta=RequestMeta(
                original_text=request.text,
                detected_language="en",
                timestamp_utc="2024-01-01T00:00:00Z"
            ),
            intent=Intent(
                system="BRAKES",
                component="PADS_ROTORS",
                failure_mode="SQUEALING",
                confidence=0.92
            )
        )
    
    return mock_process
