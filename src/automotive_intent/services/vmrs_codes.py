"""
VMRS Code Mapping
Maps automotive systems/components to Vehicle Maintenance Reporting Standards codes.
Industry standard for fleet management and enterprise automotive data.
"""
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass  
class VMRSCode:
    """VMRS Code structure."""
    code: str  # Format: XXX-XXX-XXX (System-Assembly-Component)
    description: str
    system_name: str


# VMRS System Codes (First 3 digits)
VMRS_SYSTEMS = {
    "BRAKES": "013",
    "ELECTRICAL": "034", 
    "POWERTRAIN": "045",
    "HVAC": "083",
    "SUSPENSION": "014",
    "STEERING": "015",
    "EXHAUST": "043",
    "TIRES_WHEELS": "017",
    "FUEL": "044",
    "COOLING": "042",
}

# VMRS Assembly/Component Codes
VMRS_COMPONENTS = {
    # Brakes
    ("BRAKES", "PADS_ROTORS"): "013-001",
    ("BRAKES", "BRAKE_FLUID"): "013-002",
    ("BRAKES", "CALIPERS"): "013-003",
    ("BRAKES", "ABS"): "013-004",
    ("BRAKES", "BRAKE_LINES"): "013-005",
    
    # Electrical
    ("ELECTRICAL", "BATTERY"): "034-001",
    ("ELECTRICAL", "ALTERNATOR"): "034-002",
    ("ELECTRICAL", "STARTER_MOTOR"): "034-003",
    ("ELECTRICAL", "WIRING"): "034-004",
    ("ELECTRICAL", "FUSES"): "034-005",
    
    # Powertrain
    ("POWERTRAIN", "ENGINE"): "045-001",
    ("POWERTRAIN", "TRANSMISSION"): "045-002",
    ("POWERTRAIN", "CLUTCH"): "045-003",
    ("POWERTRAIN", "DRIVESHAFT"): "045-004",
    
    # HVAC
    ("HVAC", "COMPRESSOR"): "083-001",
    ("HVAC", "CONDENSER"): "083-002",
    ("HVAC", "EVAPORATOR"): "083-003",
    ("HVAC", "BLOWER_MOTOR"): "083-004",
    ("HVAC", "HEATER_CORE"): "083-005",
    
    # Suspension
    ("SUSPENSION", "SHOCKS_STRUTS"): "014-001",
    ("SUSPENSION", "BALL_JOINTS"): "014-002",
    ("SUSPENSION", "CONTROL_ARMS"): "014-003",
    ("SUSPENSION", "SPRINGS"): "014-004",
    
    # Steering
    ("STEERING", "POWER_STEERING"): "015-001",
    ("STEERING", "STEERING_RACK"): "015-002",
    ("STEERING", "TIE_RODS"): "015-003",
    ("STEERING", "STEERING_COLUMN"): "015-004",
    
    # Exhaust
    ("EXHAUST", "CATALYTIC_CONVERTER"): "043-001",
    ("EXHAUST", "MUFFLER"): "043-002",
    ("EXHAUST", "EXHAUST_MANIFOLD"): "043-003",
    ("EXHAUST", "O2_SENSORS"): "043-004",
    
    # Tires/Wheels
    ("TIRES_WHEELS", "TIRES"): "017-001",
    ("TIRES_WHEELS", "WHEELS"): "017-002",
    ("TIRES_WHEELS", "TPMS"): "017-003",
}

# Failure Mode Suffixes
VMRS_FAILURE_MODES = {
    # Common failure modes
    "SQUEALING": "001",
    "GRINDING": "002",
    "VIBRATION": "003",
    "NOISE": "004",
    "LEAK": "005",
    "NO_START": "006",
    "OVERHEATING": "007",
    "WARNING_LIGHT": "008",
    "DEAD_CELL": "009",
    "CORROSION": "010",
    "NOT_CHARGING": "011",
    "ROUGH_IDLE": "012",
    "MISFIRING": "013",
    "STALLING": "014",
    "HARD_SHIFT": "015",
    "SLIPPING": "016",
    "CLUTCH_FAILURE": "017",
    "NO_AIRFLOW": "018",
    "NO_HEAT": "019",
    "BOUNCY_RIDE": "020",
    "CLUNKING": "021",
    "WANDERING": "022",
    "PUNCTURE": "023",
    "FLAT": "024",
}


class VMRSMapper:
    """
    Maps GarageIQ classifications to VMRS codes.
    
    VMRS (Vehicle Maintenance Reporting Standards) is the industry-standard
    coding system used by fleets, OEMs, and service networks.
    """
    
    def __init__(self):
        self.systems = VMRS_SYSTEMS
        self.components = VMRS_COMPONENTS
        self.failure_modes = VMRS_FAILURE_MODES
    
    def get_vmrs_code(
        self, 
        system: str, 
        component: str, 
        failure_mode: str
    ) -> Optional[VMRSCode]:
        """
        Get VMRS code for a classification.
        
        Args:
            system: e.g., "BRAKES"
            component: e.g., "PADS_ROTORS"  
            failure_mode: e.g., "SQUEALING"
            
        Returns:
            VMRSCode with full code like "013-001-001"
        """
        # Get system code
        system_code = self.systems.get(system)
        if not system_code:
            return None
        
        # Get component code
        component_key = (system, component)
        component_code = self.components.get(component_key)
        if not component_code:
            # Fallback to system-level
            component_code = f"{system_code}-000"
        
        # Get failure mode suffix
        failure_suffix = self.failure_modes.get(failure_mode, "099")
        
        # Build full code
        full_code = f"{component_code}-{failure_suffix}"
        
        return VMRSCode(
            code=full_code,
            description=f"{system} {component} {failure_mode}",
            system_name=system
        )
    
    def get_all_system_codes(self) -> Dict[str, str]:
        """Return all VMRS system codes."""
        return self.systems.copy()


# Singleton
_mapper: VMRSMapper | None = None

def get_vmrs_mapper() -> VMRSMapper:
    global _mapper
    if _mapper is None:
        _mapper = VMRSMapper()
    return _mapper
