from typing import Dict, List, Set

# --- Core Service Ontology (Closed World) ---
# PRD Section 7 — Expanded with SUSPENSION, STEERING, EXHAUST

SERVICE_ONTOLOGY: Dict[str, Dict[str, List[str]]] = {
    "POWERTRAIN": {
        "ENGINE": [
            "NO_START",
            "ROUGH_IDLE",
            "OVERHEATING",
            "OIL_LEAK",
            "MISFIRING",
            "STALLING"
        ],
        "TRANSMISSION": [
            "HARD_SHIFT",
            "SLIPPING",
            "DELAYED_ENGAGEMENT",
            "GRINDING_GEAR",
            "FLUID_LEAK"
        ]
    },
    "ELECTRICAL": {
        "BATTERY": [
            "LOW_VOLTAGE",
            "DEAD_CELL",
            "CORROSION",
            "NOT_CHARGING"
        ],
        "STARTER_MOTOR": [
            "NO_CRANK",
            "SOLENOID_CLICK",
            "GRINDING_NOISE",
            "INTERMITTENT_START"
        ],
        "ALTERNATOR": [
            "NOT_CHARGING",
            "WHINING_NOISE",
            "WARNING_LIGHT"
        ]
    },
    "HVAC": {
        "COMPRESSOR": [
            "CLUTCH_FAILURE",
            "NOISY_OPERATION",
            "LEAK",
            "NOT_ENGAGING"
        ],
        "BLOWER_MOTOR": [
            "NO_AIRFLOW",
            "NOISE_VIBRATION",
            "INTERMITTENT"
        ],
        "HEATER_CORE": [
            "NO_HEAT",
            "COOLANT_SMELL",
            "FOGGING_WINDOWS"
        ]
    },
    "BRAKES": {
        "PADS_ROTORS": [
            "SQUEALING",
            "GRINDING",
            "VIBRATION_ON_BRAKE",
            "PULSATING_PEDAL"
        ],
        "BRAKE_FLUID": [
            "LOW_LEVEL",
            "CONTAMINATED",
            "SPONGY_PEDAL"
        ],
        "ABS": [
            "WARNING_LIGHT",
            "PULSATING",
            "NOT_ENGAGING"
        ]
    },
    "SUSPENSION": {
        "SHOCKS_STRUTS": [
            "BOUNCY_RIDE",
            "LEAKING",
            "CLUNKING_NOISE",
            "UNEVEN_TIRE_WEAR"
        ],
        "BALL_JOINTS": [
            "CLUNKING",
            "WANDERING",
            "VIBRATION"
        ],
        "CONTROL_ARMS": [
            "WORN_BUSHINGS",
            "CLUNKING",
            "ALIGNMENT_ISSUES"
        ]
    },
    "STEERING": {
        "POWER_STEERING": [
            "WHINING_NOISE",
            "STIFF_STEERING",
            "FLUID_LEAK",
            "JERKY_MOVEMENT"
        ],
        "STEERING_RACK": [
            "LOOSE_STEERING",
            "CLUNKING",
            "FLUID_LEAK"
        ],
        "TIE_RODS": [
            "LOOSE",
            "CLUNKING",
            "UNEVEN_TIRE_WEAR"
        ]
    },
    "EXHAUST": {
        "CATALYTIC_CONVERTER": [
            "ROTTEN_EGG_SMELL",
            "CHECK_ENGINE_LIGHT",
            "REDUCED_POWER",
            "RATTLING"
        ],
        "MUFFLER": [
            "LOUD_EXHAUST",
            "RATTLING",
            "RUST_HOLES"
        ],
        "EXHAUST_MANIFOLD": [
            "TICKING_NOISE",
            "EXHAUST_SMELL",
            "LEAK"
        ]
    },
    "TIRES_WHEELS": {
        "TIRES": [
            "PUNCTURE",
            "FLAT",
            "LOW_PRESSURE",
            "WORN_TREAD",
            "BULGE",
            "UNEVEN_WEAR"
        ],
        "WHEELS": [
            "BENT_RIM",
            "VIBRATION",
            "LOOSE_LUG_NUTS",
            "CRACKED"
        ],
        "TPMS": [
            "WARNING_LIGHT",
            "SENSOR_FAILURE",
            "INACCURATE_READING"
        ]
    },
    # India-specific: CNG/LPG systems (very common in India)
    "CNG_LPG": {
        "CNG_KIT": [
            "LEAK",
            "LOW_PRESSURE",
            "CYLINDER_EMPTY",
            "REGULATOR_FAILURE",
            "INJECTOR_ISSUE",
            "NOT_SWITCHING",
            "FILLING_PROBLEM"
        ],
        "LPG_KIT": [
            "LEAK",
            "VAPORIZER_FAILURE",
            "SOLENOID_ISSUE",
            "MIXER_ADJUSTMENT",
            "TANK_ISSUE"
        ],
        "GAS_SENSOR": [
            "WARNING_LIGHT",
            "FALSE_ALARM",
            "SENSOR_FAILURE"
        ]
    },
    # Two-wheelers (huge market in India)
    "TWO_WHEELER": {
        "KICK_SELF_START": [
            "KICK_NOT_WORKING",
            "SELF_START_FAILURE",
            "STARTER_MOTOR_ISSUE"
        ],
        "CHAIN_SPROCKET": [
            "CHAIN_LOOSE",
            "CHAIN_NOISE",
            "SPROCKET_WORN",
            "CHAIN_BROKEN"
        ],
        "CLUTCH_LEVER": [
            "HARD_CLUTCH",
            "SLIPPING",
            "FREE_PLAY_ISSUE",
            "CABLE_BROKEN"
        ],
        "CARBURETOR": [
            "STARTING_ISSUE",
            "IDLING_PROBLEM",
            "FUEL_OVERFLOW",
            "MILEAGE_DROP"
        ],
        "DISC_DRUM_BRAKE": [
            "SQUEALING",
            "NOT_GRIPPING",
            "LEVER_LOOSE",
            "DRUM_WORN"
        ]
    }
}

# --- Helper Sets for Validation Speed ---

VALID_SYSTEMS: Set[str] = set(SERVICE_ONTOLOGY.keys())

def get_valid_components(system: str) -> Set[str]:
    """Return valid components for a given system."""
    if system not in SERVICE_ONTOLOGY:
        return set()
    return set(SERVICE_ONTOLOGY[system].keys())

def get_valid_failure_modes(system: str, component: str) -> Set[str]:
    """Return valid failure modes for a given system and component."""
    if system not in SERVICE_ONTOLOGY:
        return set()
    if component not in SERVICE_ONTOLOGY[system]:
        return set()
    return set(SERVICE_ONTOLOGY[system][component])

def validate_ontology_path(system: str, component: str, failure_mode: str) -> bool:
    """
    Strict boolean check if a path exists in the ontology.
    Used by Pydantic validators.
    """
    if system not in SERVICE_ONTOLOGY:
        return False
    if component not in SERVICE_ONTOLOGY[system]:
        return False
    return failure_mode in SERVICE_ONTOLOGY[system][component]


def get_ontology_formatted() -> str:
    """Return ontology as formatted string for LLM prompts."""
    lines = []
    for system, components in SERVICE_ONTOLOGY.items():
        lines.append(f"SYSTEM: {system}")
        for component, modes in components.items():
            lines.append(f"  COMPONENT: {component}")
            lines.append(f"    FAILURE_MODES: {', '.join(modes)}")
    return "\n".join(lines)
